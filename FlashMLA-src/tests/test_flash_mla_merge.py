"""
FlashMLA Merge Kernel 性能测试

测试 sparse attention merge kernel 在不同 hit rate 下的性能表现。
merge kernel 通过复用预测的 QK 分数来加速 sparse attention。

主要测试内容：
1. 不同 hit rate 下的性能对比
2. 与标准 sparse kernel 的加速比测量
3. 正确性验证
"""

import math
import time
from typing import Tuple, Optional
import random
import dataclasses

import torch
import torch.nn.functional as F
import triton

from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_merge_fwd
from lib import check_is_allclose

@dataclasses.dataclass
class TestParam:
    """测试参数配置类"""
    s_q: int              # Query 序列长度
    s_kv: int             # KV 缓存序列长度
    topk: int             # 每个 query 的 top-k 索引数
    h_q: int = 128        # Query 的头数
    h_kv: int = 1         # KV 的头数 (MLA 通常为 1)
    d_qk: int = 576       # QK 的维度
    d_v: int = 512        # V 的维度
    hit_rate: float = 0.5 # 预测索引的命中率目标值 (0.0-1.0)
    seed: int = 0         # 随机种子
    check_correctness: bool = True  # 是否进行正确性检查
    benchmark: bool = True          # 是否进行性能基准测试

@dataclasses.dataclass
class Testcase:
    """测试用例数据类"""
    t: TestParam                           # 测试参数配置
    q: torch.Tensor                        # Query 张量 [s_q, h_q, d_qk]
    kv: torch.Tensor                       # KV 张量 [s_kv, h_kv, d_qk]
    actual_indices: torch.Tensor           # 实际的 sparse 索引 [s_q, h_kv, topk]
    predicted_indices: Optional[torch.Tensor]  # 预测的索引 [s_q, h_kv, topk]
    predicted_qk: Optional[torch.Tensor]   # 预测的 QK 分数 [s_q, h_q, topk]
    hit_mask: Optional[torch.Tensor]       # hit/miss 掩码 [s_q, h_kv, topk]

def _resolve_kv_and_trim(
    indices: torch.Tensor,          # [s_q, topk], 可能用 -1 填充
    kv_pool: torch.Tensor,          # [pool_size, 1, d_qk] KV 池
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """处理并裁剪索引，去除填充并确保边界安全

    NSA indexer 将 topk 索引填充到 2048 的倍数，用 -1 表示无效索引。
    所有有效索引都是 KV pool 的槽位索引（分页式）。

    返回值:
        safe_idx:    [s_q, K]   限制在 [0, pool_sz-1] 范围内，无效槽位设为 0
        valid_mask:  [s_q, K]   原始索引满足 0 <= idx < pool_sz 的位置为 True
        K:           int        裁剪后的列数
    """
    # 去除尾部的无效填充（仅按非负判断用于裁剪长度）
    valid_full = indices >= 0                                # [s_q, topk]
    K = max(int(valid_full.sum(dim=-1).max().item()), 1)    # 找到最大有效长度
    trimmed = indices[:, :K]                                 # [s_q, K] 裁剪到有效长度

    # 将索引限制在 pool 边界内（无效 → 0，超出边界 → pool_sz-1）
    pool_sz = kv_pool.shape[0]
    valid_mask = (trimmed >= 0) & (trimmed < pool_sz)        # [s_q, K] 严格有效性掩码
    safe_idx = trimmed.clamp(min=0, max=pool_sz - 1)        # [s_q, K]

    return safe_idx, valid_mask, K

def _remap_predicted_to_actual(
    predicted_indices: torch.Tensor,   # [s_q, pred_topk] 预测的索引
    actual_idx: torch.Tensor,          # [s_q, K] 实际的索引
    h_q: int,                          # Query 头数
) -> torch.Tensor:
    """为每个实际位置找到其在预测索引中的位置

    返回可用于 torch.gather(predicted_qk, 2, ...) 的索引张量 [s_q, h_q, K]
    """
    # 克隆预测索引，将无效值设为 -2（哨兵值）
    safe_pred = predicted_indices.clone()
    safe_pred[safe_pred < 0] = -2

    # 对预测索引排序，得到排序后的值和原始位置的映射
    sorted_pred, sort_perm = safe_pred.sort(dim=-1)

    # 为每个实际索引在排序后的预测索引中查找位置
    search_vals = actual_idx.clamp(min=0)  # 确保搜索值非负
    pos_in_sorted = torch.searchsorted(sorted_pred, search_vals)
    pos_in_sorted = pos_in_sorted.clamp(max=sorted_pred.shape[-1] - 1)

    # 从排序映射中恢复原始位置
    original_pos = torch.gather(sort_perm, 1, pos_in_sorted)  # [s_q, K]

    # 扩展到 head 维度，返回 [s_q, h_q, K]
    return original_pos.unsqueeze(1).expand(-1, h_q, -1)

def generate_testcase(t: TestParam) -> Testcase:
    """生成测试用例数据，按照 attn_cache.py 的数据处理流程"""
    # 设置随机种子确保可重复性
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)

    device = torch.device("cuda:0")

    # 生成输入张量：Q 和 KV
    q = torch.randn((t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn((t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16, device=device) / 10

    # 限制数值范围，避免数值问题
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    # ===== 生成实际的 sparse 索引（ground truth）=====
    actual_indices = torch.full((t.s_q, t.h_kv, t.topk), t.s_kv, dtype=torch.int32, device=device)

    for s in range(t.s_q):  # 遍历每个 query 位置
        for h in range(t.h_kv):  # 遍历每个 KV 头（通常为 1）
            # 生成现实的索引分布：大部分接近序列末尾，少部分随机分布
            near_mask = torch.randint(0, 32, (min(t.topk, t.s_kv),), device=device) < 31  # 31/32 的概率选择近端
            cur_indices = torch.randperm(t.s_kv, device=device)[:t.topk]  # 随机选择 topk 个索引

            # 将近端掩码对应的索引替换为接近末尾的索引
            cur_indices[near_mask] = torch.randint(max(0, t.s_kv - 20000), t.s_kv - 1,
                                                   (near_mask.sum().item(),), device=device)

            # 如果生成的索引不够，用大数值填充（会被当作无效索引处理）
            if len(cur_indices) < t.topk:
                cur_indices = torch.cat([cur_indices, torch.full((t.topk - len(cur_indices),),
                                                                2147480000, device=device)])

            # 随机打乱顺序
            cur_indices = cur_indices[torch.randperm(t.topk, device=device)]
            actual_indices[s, h] = cur_indices

    # ===== 生成预测索引和 hit mask 以达到目标命中率 =====
    predicted_indices = None
    predicted_qk = None
    hit_mask = None

    if t.hit_rate > 0:
        # 初始化预测索引（用无效值填充）
        predicted_indices = torch.full((t.s_q, t.h_kv, t.topk), t.s_kv, dtype=torch.int32, device=device)

        # 创建 hit mask，控制命中率
        hit_mask = torch.rand((t.s_q, t.h_kv, t.topk), device=device) < t.hit_rate

        for s in range(t.s_q):
            for h in range(t.h_kv):
                actual_row = actual_indices[s, h]  # 当前位置的实际索引
                hit_positions = hit_mask[s, h]     # 当前位置的命中位置

                # 对于命中位置：使用实际索引
                predicted_indices[s, h, hit_positions] = actual_row[hit_positions]

                # 对于 miss 位置：生成随机索引
                miss_positions = ~hit_positions
                if miss_positions.any():
                    num_miss = miss_positions.sum().item()
                    # 简单地生成随机索引（允许重复，因为这是测试）
                    miss_indices = torch.randint(0, t.s_kv, (num_miss,), device=device, dtype=torch.int32)
                    predicted_indices[s, h, miss_positions] = miss_indices

        # ===== 计算预测的 QK 分数 =====
        # 按照 attn_cache.py 的 compute_qk_scores_pytorch 逻辑计算
        # predicted_qk 应该基于 predicted_indices 计算，而不是 actual_indices
        predicted_qk = torch.zeros((t.s_q, t.h_q, t.topk), dtype=torch.float32, device=device)

        if t.hit_rate > 0:  # 只有在有命中时才计算
            for s in range(t.s_q):  # 遍历每个 query 位置
                pred_indices = predicted_indices[s, 0]  # [topk] 预测索引

                # 为每个预测索引计算 QK 分数
                q_s = q[s:s+1].float()  # [1, h_q, d_qk] 当前 query

                for i, idx in enumerate(pred_indices):
                    if idx >= 0 and idx < t.s_kv:
                        # 有效索引：计算 QK 分数
                        kv_selected = kv[idx:idx+1].float()  # [1, h_kv, d_qk]
                        qk_score = q_s @ kv_selected.transpose(1, 2)  # [1, h_q, 1]
                        predicted_qk[s, :, i:i+1] = qk_score[0]  # raw Q@K^T, kernel 内部统一乘 sm_scale*log2(e)
                    else:
                        # 无效索引：设为 -inf
                        predicted_qk[s, :, i] = float('-inf')

    return Testcase(
        t=t,
        q=q,
        kv=kv,
        actual_indices=actual_indices,
        predicted_indices=predicted_indices,
        predicted_qk=predicted_qk,
        hit_mask=hit_mask
    )

def get_flop(p: TestParam) -> float:
    """Calculate FLOPs for the sparse attention computation"""
    # For merge kernel: only miss positions need QK computation
    miss_rate = 1.0 - p.hit_rate
    qk_flops = 2 * p.h_q * p.d_qk * p.topk * miss_rate
    pv_flops = 2 * p.h_q * p.d_v * p.topk
    total_flops = (qk_flops + pv_flops) * p.s_q
    return total_flops

def run_merge_kernel_test(p: TestParam) -> bool:
    """运行 merge kernel 性能测试，对比标准 sparse kernel

    按照 attn_cache.py 的数据处理流程：
    1. Resolve & trim indices
    2. Sort indices (miss first, hit next, invalid last)
    3. Remap predicted_qk to sorted order
    4. Pad and format data for kernel
    5. Call kernel and measure performance
    """
    print("================")
    print(f"测试 sparse_attn_merge_fwd_kernel，命中率: {p.hit_rate*100:.1f}%")
    print(f"参数配置: {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    sm_scale = 1 / math.sqrt(p.d_qk)
    device = t.q.device
    torch.cuda.synchronize()

    # ===== 按照 attn_cache.py 的数据处理流程 =====
    s_q, h_q, d_qk = t.q.shape
    kv_lora_rank = p.d_v

    # ---- 步骤1: Resolve & trim ----
    # 处理实际索引，去除填充，确保边界安全
    safe_idx, valid_mask, K = _resolve_kv_and_trim(t.actual_indices.squeeze(1), t.kv.squeeze(1))

    # 裁剪 hit_mask 到有效长度
    if t.hit_mask is not None and t.hit_mask.shape[-1] > K:
        hit_mask_trimmed = t.hit_mask[:, :, :K]
    else:
        hit_mask_trimmed = t.hit_mask

    # 检查是否存在命中情况
    has_hit = (
        hit_mask_trimmed is not None and hit_mask_trimmed.any()
        and t.predicted_qk is not None
        and t.predicted_indices is not None
    )

    # ---- 步骤2: Sort indices (miss first, hit next, invalid last) ----
    # 创建排序键：0=miss, 1=hit, 2=invalid
    sort_key = torch.zeros(s_q, K, dtype=torch.long, device=device)
    if has_hit:
        # 命中且有效的索引设为 1
        sort_key[hit_mask_trimmed.squeeze(1) & valid_mask] = 1
    # 无效索引设为 2（排序时排在最后）
    sort_key[~valid_mask] = 2

    # 排序得到重排列的索引
    _, perm = sort_key.sort(dim=-1, stable=True)           # [s_q, K] 排序后的位置映射

    # 根据排序重新排列所有张量
    sorted_idx   = torch.gather(safe_idx, 1, perm)        # [s_q, K] 排序后的索引
    sorted_valid = torch.gather(valid_mask, 1, perm)       # [s_q, K] 排序后的有效性掩码
    sorted_hit   = torch.gather(
        hit_mask_trimmed.squeeze(1) if has_hit else torch.zeros_like(valid_mask),
        1, perm,
    )                                                       # [s_q, K] 排序后的命中掩码

    # ---- 步骤3: Remap predicted_qk to sorted order ----
    if has_hit:
        # 获取排序后的实际索引，用于 remap
        sorted_orig = torch.gather(t.actual_indices.squeeze(1)[:, :K], 1, perm)

        # 计算 remap 位置：每个排序后的实际索引在预测索引中的位置
        remap_pos = _remap_predicted_to_actual(
            t.predicted_indices.squeeze(1), sorted_orig, h_q,
        )                                                   # [s_q, h_q, K]

        # 根据 remap 位置重新排列 predicted_qk
        remapped_qk = torch.gather(t.predicted_qk, 2, remap_pos)

        # 对于 miss/invalid 位置设为 0（kernel 会忽略这些位置）
        not_hit = ~(sorted_hit & sorted_valid).unsqueeze(1).expand(-1, h_q, -1)
        remapped_qk = remapped_qk.masked_fill(not_hit, 0.0)
    else:
        # 没有命中时，QK 分数全部为 0
        remapped_qk = torch.zeros(
            s_q, h_q, K, dtype=torch.float32, device=device,
        )

    # ---- 步骤4: Pad data to kernel format ----

    # 将 topk 填充到 128 的倍数（2 * B_TOPK）
    B_TOPK_2 = 128
    pad_len = (B_TOPK_2 - (K % B_TOPK_2)) % B_TOPK_2
    if pad_len > 0:
        sorted_idx   = F.pad(sorted_idx,   (0, pad_len), value=0)   # 填充索引为 0
        sorted_valid = F.pad(sorted_valid,  (0, pad_len), value=False)  # 填充有效性为 False
        # 填充位置标记为 "hit"，让 kernel 跳过这些位置的 QK GEMM 计算
        sorted_hit   = F.pad(sorted_hit.long(), (0, pad_len), value=1).bool()
        remapped_qk  = F.pad(remapped_qk,  (0, pad_len), value=0.0)  # 填充 QK 为 0
    topk_padded = sorted_idx.shape[-1]

    # ---- 步骤5: Build kernel-format tensors ----
    # 构建符合 kernel 要求格式的张量

    # indices: [s_q, h_kv=1, topk_padded], int32
    # 无效位置设为 -1，让 kernel 标记 is_kv_valid=False
    kernel_indices = sorted_idx.clone()
    kernel_indices[~sorted_valid] = -1
    indices_3d  = kernel_indices.unsqueeze(1).to(torch.int32).contiguous()

    # hit_mask: [s_q, h_kv=1, topk_padded], int32 (1=hit, 0=miss)
    hit_mask_3d = sorted_hit.unsqueeze(1).to(torch.int32).contiguous()

    # predicted_qk: [s_q, h_q, topk_padded], float32
    remapped_qk = remapped_qk.contiguous()

    # 将 h_q 填充到 64 的倍数（WGMMA atom M-dimension）
    B_H_KERNEL = 64
    pad_h = (B_H_KERNEL - (h_q % B_H_KERNEL)) % B_H_KERNEL
    if pad_h > 0:
        # 用 0 填充 Q → Q@K^T = 0 对填充的头是无害的
        q_kernel = F.pad(t.q, (0, 0, 0, pad_h)).contiguous()         # [s_q, h_q+pad_h, d_qk]
        qk_kernel = F.pad(remapped_qk, (0, 0, 0, pad_h)).contiguous() # [s_q, h_q+pad_h, topk_padded]
    else:
        q_kernel = t.q.contiguous()
        qk_kernel = remapped_qk

    # ===== 调用 kernel 进行测试 =====

    # 调用 merge kernel（新实现）
    def run_merge_kernel():
        output, _max_logits, _lse = flash_mla_sparse_merge_fwd(
            q=q_kernel,
            kv=t.kv.contiguous(),
            actual_indices=indices_3d,
            predicted_qk=qk_kernel,
            hit_mask=hit_mask_3d,
            sm_scale=sm_scale,
            d_v=kv_lora_rank,
        )
        # 裁剪回原始的 h_q 维度
        if pad_h > 0:
            output = output[:, :h_q, :].contiguous()
        return output, _max_logits, _lse

    # 调用标准 sparse kernel（基准）
    def run_sparse_kernel():
        return flash_mla_sparse_fwd(
            q=t.q,
            kv=t.kv,
            indices=t.actual_indices,
            sm_scale=sm_scale,
            d_v=p.d_v,
        )

    # Benchmark merge kernel
    merge_out, merge_max_logits, merge_lse = run_merge_kernel()
    torch.cuda.synchronize()

    if p.benchmark:
        flop = get_flop(p)
        merge_time: float = triton.testing.do_bench(run_merge_kernel, warmup=10, rep=20) / 1000  # Convert to ms
        merge_flops = flop / merge_time / 1e12
        print(f"Merge kernel:  {merge_time * 1e6:4.0f} us, {merge_flops:.3f} TFlops")

        torch.cuda.synchronize()
        sparse_time: float = triton.testing.do_bench(run_sparse_kernel, warmup=10, rep=20) / 1000  # Convert to ms
        sparse_flops = flop / sparse_time / 1e12  # Note: using same FLOP count for fair comparison
        print(f"Sparse kernel: {sparse_time * 1e6:4.0f} us, {sparse_flops:.3f} TFlops")

        speedup = sparse_time / merge_time
        print(f"Speedup: {speedup:.2f}x")

    if p.check_correctness:
        torch.cuda.synchronize()

        # Get reference result from sparse kernel
        ref_out, ref_max_logits, ref_lse = run_sparse_kernel()
        torch.cuda.synchronize()

        # Debug: Print some statistics for 100% hit rate
        if p.hit_rate == 1.0:
            print(f"Debug 100% hit rate:")
            print(f"  merge_out mean: {merge_out.mean().item():.6f}, std: {merge_out.std().item():.6f}")
            print(f"  ref_out mean: {ref_out.mean().item():.6f}, std: {ref_out.std().item():.6f}")
            print(f"  merge_max_logits mean: {merge_max_logits.mean().item():.6f}, std: {merge_max_logits.std().item():.6f}")
            print(f"  ref_max_logits mean: {ref_max_logits.mean().item():.6f}, std: {ref_max_logits.std().item():.6f}")

        # Check correctness one by one to save memory
        is_correct = True

        # Check output with relaxed cosine tolerance for numerical precision
        is_correct &= check_is_allclose("merge_out", merge_out, ref_out, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=1e-4)
        del ref_out  # Free memory
        torch.cuda.empty_cache()

        is_correct &= check_is_allclose("merge_max_logits", merge_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01 / 65536)
        del ref_max_logits  # Free memory
        torch.cuda.empty_cache()

        # Check lse with relaxed tolerance
        is_correct &= check_is_allclose("merge_lse", merge_lse, ref_lse, abs_tol=1e-1, rel_tol=1.0, cos_diff_tol=1e-4)
        del ref_lse  # Free memory
        torch.cuda.empty_cache()

        print(f"is_correct: {is_correct}")

        return is_correct

    return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    # ===== 测试用例配置 =====

    # 精度测试 (benchmark=False → check_correctness 不被覆盖)
    # merge kernel vs sparse kernel，不同 hit rate
    accuracy_tests = [
        TestParam(s_q=100, s_kv=2048, topk=2048, hit_rate=rate,
                 check_correctness=True, benchmark=True)
        for rate in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    ]

    # ===== 运行所有测试 =====
    all_tests = accuracy_tests

    failed_cases = []
    for test in all_tests:
        if test.benchmark:
            time.sleep(0.2)  # 测试间冷却，避免热量积累
        is_correct = run_merge_kernel_test(test)
        if not is_correct:
            failed_cases.append(test)

    # ===== 输出测试结果 =====
    total = len(all_tests)  
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {total} 个测试用例失败:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
    else:
        print(f"\033[32m\033[1m所有 {total} 个测试用例通过！\033[0m")