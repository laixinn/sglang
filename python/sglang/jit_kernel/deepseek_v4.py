from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

try:
    import tilelang
    import tilelang.language as T

    _tilelang_available = True
except ImportError:
    _tilelang_available = False

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name(f"common"),
        cuda_files=[f"deepseek_v4/common.cuh"],
        cuda_wrappers=[("plan_compress_prefill", "plan_compress_prefill")],
    )


@cache_once
def _jit_compress_128_online_plan_module() -> Module:
    """Host-side plan generator for online compress 128 (no template args)."""
    return load_jit(
        make_name("compress_128_online_plan"),
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("plan_compress_online_prefill", "plan_compress_online_prefill"),
        ],
    )


@cache_once
def _jit_compress_128_online_module(head_dim: int) -> Module:
    """Online compress 128 kernel: ring_size=1, per-index (max, sum, kv) state."""
    args = make_cpp_args(head_dim, is_arch_support_pdl())
    kernel_class = f"FlashCompress128OnlineKernel<{args}>"
    return load_jit(
        make_name("compress_128_online"),
        *args,
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk"),
        *args,
        cuda_files=["deepseek_v4/topk.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK512Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk1024_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk1024"),
        *args,
        cuda_files=["deepseek_v4/topk_1024.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK1024Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module(topk: int) -> Module:
    return load_jit(
        make_name("topk_v2"),
        str(topk),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "CombinedTopKKernel::transform"),
            ("topk_plan", "CombinedTopKKernel::plan"),
        ],
        extra_cuda_cflags=[f"-DSGL_TOPK={topk}"],
    )


@cache_once
def _jit_mask_topk_module() -> Module:
    return load_jit(
        make_name("mask_topk"),
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("run", "MaskKernel::run")],
    )


@cache_once
def _jit_hash_topk_module() -> Module:
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_compress_module_v2_defensive(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress128Kernel<{args}>"
    return load_jit(
        make_name("compress_128_v2_defensive"),
        *args,
        cuda_files=["deepseek_v4/c128_v2.cuh"],
        cuda_wrappers=[
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_rmsnorm_head_module(head_dim: int, dtype: torch.dtype):
    args = make_cpp_args(head_dim, dtype, is_arch_support_pdl())
    kernel_class = f"RMSNormKernel<{args}>"
    return load_jit(
        make_name("rmsnorm_head"),
        *args,
        cuda_files=["deepseek_v4/rmsnorm.cuh"],
        cuda_wrappers=[("run_self", f"{kernel_class}::run_self")],
    )


@cache_once
def _jit_fused_rope_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name(f"fused_norm_rope"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
) -> Module:
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[("run", f"{kernel_class}::run")],
    )


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


@cache_once
def _jit_silu_mul_quant_varlen_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
) -> Module:
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_varlen"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
) -> Module:
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


# ---------------------------------------------------------------------------
# Byte-equal fallbacks: when SGLANG_OPT_FIX_MEGA_MOE_MEMORY is off, route
# silu_and_mul_masked_post_quant / silu_and_mul_clamp through these _tmp
# modules, which load a copy of the optimize-branch kernel (different
# precision behavior ??? bf16 silu roundtrip, expf, fp32 clamp).
# ---------------------------------------------------------------------------


@cache_once
def _jit_silu_mul_quant_tmp_module(
    quant_group_size: int, scale_ue8m0: bool, apply_swiglu_limit: bool
) -> Module:
    args = make_cpp_args(
        quant_group_size, scale_ue8m0, is_arch_support_pdl(), apply_swiglu_limit
    )
    return load_jit(
        make_name("silu_mul_quant_tmp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
    )


@cache_once
def _jit_silu_and_mul_clamp_tmp_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp_tmp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
    )


@cache_once
def _jit_mega_moe_pre_dispatch_module(quant_group_size: int) -> Module:
    args = make_cpp_args(quant_group_size, is_arch_support_pdl())
    return load_jit(
        make_name("mega_moe_pre_dispatch"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


@cache_once
def _jit_hisparse_transfer_module() -> Module:
    return load_jit(
        make_name("hisparse_transfer"),
        cuda_files=["deepseek_v4/hisparse_transfer.cuh"],
        cuda_wrappers=[("hisparse_transfer", "hisparse_transfer")],
    )


def hisparse_offload_to_host(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    module = _jit_hisparse_transfer_module()
    module.hisparse_transfer(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if out_page_indices.shape[1] == 512:
        module = _jit_topk_module()
    else:
        module = _jit_topk1024_module()
    module.topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


_WORKSPACE_INTS_PER_BATCH = 2 + 1024 * 2
_PLAN_METADATA_INTS_PER_BATCH = 4


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    module = _jit_topk_v2_module(512)  # does not matter
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    module = _jit_topk_v2_module(out_page_indices.shape[1])
    bs = scores.shape[0]
    workspace = seq_lens.new_empty(bs, _WORKSPACE_INTS_PER_BATCH)
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        workspace,
        metadata,
    )


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    module = _jit_hash_topk_module()
    module.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor):
    return _jit_mask_topk_module().run(topk_ids, num_token_non_padded)


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

    def copy_(self, other: CompressorPrefillPlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.compress_plan.copy_(other.compress_plan)
        self.write_plan.copy_(other.write_plan)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        from sglang.srt.environ import envs

        # Online c128 keeps the same NamedTuple shape (compress_plan, write_plan)
        # so call sites that splat `*plan[1:]` continue to work, but the C++
        # plan struct semantics differ (last-token coords + window_len).
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            return CompressorPrefillPlan._generate_online(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
                device=device,
                use_cuda_graph=use_cuda_graph,
            )
        assert seq_lens.device == extend_lens.device
        seq_lens = seq_lens.to(torch.int64)
        extend_lens = extend_lens.to(torch.int64)
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device=seq_lens.device,
            pin_memory=seq_lens.is_cpu,
        )
        module = _jit_common_module()
        is_overlap = compress_ratio == 4
        plan_lens = module.plan_compress_prefill(
            extend_lens,
            seq_lens,
            plan_tensor[0],
            plan_tensor[1],
            compress_ratio,
            is_overlap,
            use_cuda_graph,
        )
        return CompressorPrefillPlan(
            compress_ratio,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )

    @staticmethod
    def _generate_online(
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool,
    ) -> CompressorPrefillPlan:
        # Online plan host-side path: only CPU/cuda-host implemented today.
        # Move inputs to CPU pinned memory then bounce the result to device.
        seq_lens_cpu = seq_lens.detach().to(torch.int64).cpu()
        extend_lens_cpu = extend_lens.detach().to(torch.int64).cpu()
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        module = _jit_compress_128_online_plan_module()
        plan_lens = module.plan_compress_online_prefill(
            extend_lens_cpu,
            seq_lens_cpu,
            plan_tensor[0],
            plan_tensor[1],
            use_cuda_graph,
        )
        return CompressorPrefillPlan(
            128,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )

    @property
    def is_decode(self) -> bool:
        return False


class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    seq_lens: torch.Tensor

    def copy_(self, other: CompressorDecodePlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.seq_lens.copy_(other.seq_lens)

    @property
    def is_decode(self) -> bool:
        return True


def compress_plan(
    compress_ratio: Literal[4, 128],
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    device: torch.device,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if extend_lens is not None:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            device,
        )
    else:
        assert num_q_tokens == len(seq_lens)
        seq_lens = seq_lens.to(device, non_blocking=True)
        return CompressorDecodePlan(compress_ratio, seq_lens)


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan, None] = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    if plan is None:
        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    # Online c128: separate JIT module, fp32 state, no compile-time dtypes.
    if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        online_module = _jit_compress_128_online_module(head_dim=head_dim)
        F = online_module.decode if plan.is_decode else online_module.prefill
        F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
        return out
    module = _jit_compress_module(
        head_dim,
        kv_score_input.dtype,
        out.dtype,
        compress_ratio,
    )
    if plan.is_decode:
        F = module.decode
    elif compress_ratio == 128 and _should_use_c128_prefill_defensive():
        F = _jit_compress_module_v2_defensive(
            head_dim,
            kv_score_input.dtype,
            out.dtype,
        ).prefill
    else:
        F = module.prefill
    F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
    return out


def _should_use_c128_prefill_defensive() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_HANDLE_C128_PREFILL_KERNEL.get()


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],
        freq_cis,
        int(plan.is_decode),
        eps,
        plan.compress_ratio,
    )


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        positions,
        freq_cis,
        2,
        eps,
        0,
    )


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


@cache_once
def _tilelang_make_swa_indices_kernel(swa_window_size: int, threads: int = 128) -> Any:
    import tilelang
    import tilelang.language as T

    batch_size = T.dynamic("batch_size")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_q_tokens = T.dynamic("num_q_tokens")
    num_warps = threads // 32
    assert swa_window_size % 32 == 0

    @tilelang.jit
    def make_swa_prefill_indices(
        seq_lens_k: T.Tensor[(batch_size,), T.int32],
        seq_lens_q: T.Tensor[(batch_size,), T.int32],
        cu_seqlens_q: T.Tensor[(batch_size_plus_1,), T.int32],
        swa_indices: T.Tensor[(num_q_tokens, swa_window_size), T.int32],
    ):
        _ = batch_size_plus_1
        with T.Kernel(T.ceildiv(num_q_tokens, num_warps), threads=threads) as bx:
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx % 32
            s_batch_id = T.alloc_shared((num_warps,), dtype=T.int32)

            token_id = warp_id + bx * num_warps
            if token_id >= num_q_tokens:
                return
            for i in T.serial(0, batch_size, step=32):
                j = i + lane_id
                if cu_seqlens_q[j] <= token_id < cu_seqlens_q[j + 1]:
                    s_batch_id[warp_id] = j
            T.sync_warp()

            seq_idx = s_batch_id[warp_id]
            kv_len = seq_lens_k[seq_idx]
            qo_len = seq_lens_q[seq_idx]
            cum_qo_len = cu_seqlens_q[seq_idx]
            prefix_len = kv_len - qo_len
            curr_seq_qo_idx = token_id - cum_qo_len
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = T.max(end_abs_pos - swa_window_size, 0)
            old_kv_start = seq_idx * swa_window_size
            new_kv_start = batch_size * swa_window_size + cum_qo_len

            for i in T.unroll(0, swa_window_size, step=32):
                j = i + lane_id
                abs_pos = start_abs_pos + j
                swa_indices[token_id, j] = T.if_then_else(
                    abs_pos < end_abs_pos,
                    T.if_then_else(
                        abs_pos < prefix_len,
                        old_kv_start + abs_pos % swa_window_size,
                        new_kv_start + (abs_pos - prefix_len),
                    ),
                    -1,
                )

    return make_swa_prefill_indices


def tilelang_make_swa_prefill_indices(
    seq_lens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
    swa_window_size = swa_indices.shape[1]
    kernel = _tilelang_make_swa_indices_kernel(swa_window_size)
    kernel(seq_lens_k, seq_lens_q, cu_seqlens_q, swa_indices)
    return swa_indices


# ---------------------------------------------------------------------------
# FP4 quantization — mirrors HF DeepSeek-V4-Pro inference/kernel.py exactly.
#
# NOTE on "from __future__ import annotations" (PEP-563):
# This module uses PEP-563 which converts ALL annotations to strings at
# definition time.  tilelang's get_type_hints() tries to evaluate those strings
# and fails when they resolve to tvm.tir.Buffer objects rather than Python
# types.  The workaround: define @T.prim_func kernels WITHOUT annotation
# syntax; instead, manually assign __annotations__ after the function def
# (as dict values that are already evaluated tvm.tir.Buffer objects).  The
# dict assignment is never subject to PEP-563 stringification.
# ---------------------------------------------------------------------------

_FP4 = "float4_e2m1fn"
_FE8M0 = "float8_e8m0fnu"
_BF16 = "bfloat16"
_FP32 = "float32"

_FP4_PASS_CONFIGS = (
    {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if _tilelang_available
    else {}
)


def _fp4_fast_round_scale(amax, fp4_max_inv):
    """Power-of-2 scale via IEEE-754 bit ops (matches HF kernel.py fast_round_scale)."""
    bits_x = T.reinterpret("uint32", amax * fp4_max_inv)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    log2_ceil = T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))
    return T.reinterpret("float32", (log2_ceil + 127) << 23)


def _fp4_build_prim_func(M, N, group_size, in_dtype, out_dtype, scale_dtype, compute_dtype, fp4_max, fp4_max_inv, blk_m, inplace):
    """Build and return a T.prim_func for FP4 quantization.

    This function lives OUTSIDE @tilelang.jit so the tilelang AST tracer
    never sees the __annotations__ assignment.  The annotation dict contains
    already-evaluated tvm.tir.Buffer objects, bypassing PEP-563 stringification
    that would otherwise turn them into unresolvable ForwardRef strings.
    """
    def fp4_quant_kernel_(X, Y, S):
        with T.Kernel(
            T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128
        ) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    s_local[i] = _fp4_fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(
                                compute_dtype,
                                T.Cast(_FP4, T.clamp(
                                    x_local[i, j] / s_local[i],
                                    -fp4_max, fp4_max,
                                )),
                            ) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    # Dict assignment is NOT subject to PEP-563 stringification; the values
    # are tvm.tir.Buffer objects that tilelang's get_type_hints can use directly.
    fp4_quant_kernel_.__annotations__ = {
        "X": T.Tensor[(M, N), in_dtype],
        "Y": T.Tensor[(M, N), out_dtype],
        "S": T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    }
    return T.prim_func(fp4_quant_kernel_)


if _tilelang_available:

    @tilelang.jit(pass_configs=_FP4_PASS_CONFIGS)
    def _fp4_quant_kernel(
        N, block_size=32, in_dtype=_BF16, scale_dtype=_FE8M0, inplace=False
    ):
        """Block-wise FP4 quantization (MXFP4, E2M1, FE8M0 power-of-2 scale).
        inplace=True: fused quant+dequant back to BF16 (QAT simulation)."""
        M = T.symbolic("M")
        fp4_max = 6.0
        fp4_max_inv = 1.0 / fp4_max
        blk_m = 32
        group_size = block_size
        compute_dtype = _FP32
        out_dtype = in_dtype if inplace else _FP4
        return _fp4_build_prim_func(
            M, N, group_size, in_dtype, out_dtype, scale_dtype,
            compute_dtype, fp4_max, fp4_max_inv, blk_m, inplace,
        )


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise MXFP4 quantization (matches HF DeepSeek-V4-Pro inference/kernel.py).

    Returns (fp4_packed, fe8m0_scales) unless inplace=True, which does fused
    quant+dequant back to BF16 (QAT simulation) and returns x.
    """
    if not _tilelang_available:
        raise RuntimeError(
            "fp4_act_quant requires TileLang. Install with: pip install tilelang"
        )
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = (
        torch.empty_like(z)
        if inplace
        else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    )
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    _fp4_quant_kernel(N, block_size, inplace=inplace)(
        z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size)
    )
    if inplace:
        x.copy_(y)
        return x
    return y, s


_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _fp4_build_paged_mqa_prim_func(
    N, L, S, C, B, D, H, D_fp4, S_fe8m0, d_0, d_1, fp4_block_size, clear_accum, lut
):
    """Build T.prim_func for FP4 paged MQA logits.

    Lives OUTSIDE @tilelang.jit so the tilelang AST tracer never sees the
    __annotations__ assignment.  Annotation dict holds already-evaluated
    tvm.tir.Buffer / StridedBuffer objects, bypassing PEP-563 stringification.

    Implements the PR #23686 idea: dequant K from MXFP4 → BF16 on-the-fly,
    then compute BF16 × BF16 GEMM for logits.
    """
    def fp4_paged_mqa_logits_(q, kvcache_fp4, kvcache_fe8m0, weight, seq_lens, page_table, o):
        _ = N, L, S, C, D, H, B, D_fp4, S_fe8m0, d_0, d_1
        with T.Kernel(N) as bx:
            seq_len = seq_lens[bx]

            q_fp8_smem = T.alloc_shared((H, D), T.float8_e4m3)
            q_smem = T.alloc_shared((H, D), T.bfloat16)
            q_s_frag = T.alloc_fragment((H,), T.float32)
            T.copy(q[bx, 0, 0], q_fp8_smem)
            T.copy(weight[bx, 0], q_s_frag)
            for h, d in T.Parallel(H, D):
                q_smem[h, d] = T.cast(q_fp8_smem[h, d], T.bfloat16)

            for i in T.Pipelined(T.ceildiv(seq_len, B), num_stages=2):
                page = page_table[bx, i]

                k_fp4_smem = T.alloc_shared((B, D_fp4), T.uint8)
                k_fe8m0_smem = T.alloc_shared((B, S_fe8m0), T.uint8)
                T.copy(kvcache_fp4[page, 0, 0], k_fp4_smem)
                T.copy(kvcache_fe8m0[page, 0, 0], k_fe8m0_smem)

                # Dequant FP4 → BF16 (PR #23686 style: unpack nibbles + FE8M0 scale)
                k_bf16_smem = T.alloc_shared((B, D), T.bfloat16)
                for j, d2 in T.Parallel(B, D_fp4):
                    packed = T.cast(k_fp4_smem[j, d2], T.int32)
                    lo_nibble = packed & 0x0F
                    hi_nibble = (packed >> 4) & 0x0F
                    scale_idx = (2 * d2) // fp4_block_size
                    fe8m0_val = T.cast(k_fe8m0_smem[j, scale_idx], T.int32)
                    scale_f32 = T.exp2(T.cast(fe8m0_val - 127, T.float32))
                    scale_bf16 = T.cast(scale_f32, T.bfloat16)
                    lo_val = T.if_then_else(lo_nibble == 0, T.cast(lut[0], T.bfloat16),
                             T.if_then_else(lo_nibble == 1, T.cast(lut[1], T.bfloat16),
                             T.if_then_else(lo_nibble == 2, T.cast(lut[2], T.bfloat16),
                             T.if_then_else(lo_nibble == 3, T.cast(lut[3], T.bfloat16),
                             T.if_then_else(lo_nibble == 4, T.cast(lut[4], T.bfloat16),
                             T.if_then_else(lo_nibble == 5, T.cast(lut[5], T.bfloat16),
                             T.if_then_else(lo_nibble == 6, T.cast(lut[6], T.bfloat16),
                             T.if_then_else(lo_nibble == 7, T.cast(lut[7], T.bfloat16),
                             T.if_then_else(lo_nibble == 8, T.cast(lut[8], T.bfloat16),
                             T.if_then_else(lo_nibble == 9, T.cast(lut[9], T.bfloat16),
                             T.if_then_else(lo_nibble == 10, T.cast(lut[10], T.bfloat16),
                             T.if_then_else(lo_nibble == 11, T.cast(lut[11], T.bfloat16),
                             T.if_then_else(lo_nibble == 12, T.cast(lut[12], T.bfloat16),
                             T.if_then_else(lo_nibble == 13, T.cast(lut[13], T.bfloat16),
                             T.if_then_else(lo_nibble == 14, T.cast(lut[14], T.bfloat16),
                             T.cast(lut[15], T.bfloat16))))))))))))))))
                    hi_val = T.if_then_else(hi_nibble == 0, T.cast(lut[0], T.bfloat16),
                             T.if_then_else(hi_nibble == 1, T.cast(lut[1], T.bfloat16),
                             T.if_then_else(hi_nibble == 2, T.cast(lut[2], T.bfloat16),
                             T.if_then_else(hi_nibble == 3, T.cast(lut[3], T.bfloat16),
                             T.if_then_else(hi_nibble == 4, T.cast(lut[4], T.bfloat16),
                             T.if_then_else(hi_nibble == 5, T.cast(lut[5], T.bfloat16),
                             T.if_then_else(hi_nibble == 6, T.cast(lut[6], T.bfloat16),
                             T.if_then_else(hi_nibble == 7, T.cast(lut[7], T.bfloat16),
                             T.if_then_else(hi_nibble == 8, T.cast(lut[8], T.bfloat16),
                             T.if_then_else(hi_nibble == 9, T.cast(lut[9], T.bfloat16),
                             T.if_then_else(hi_nibble == 10, T.cast(lut[10], T.bfloat16),
                             T.if_then_else(hi_nibble == 11, T.cast(lut[11], T.bfloat16),
                             T.if_then_else(hi_nibble == 12, T.cast(lut[12], T.bfloat16),
                             T.if_then_else(hi_nibble == 13, T.cast(lut[13], T.bfloat16),
                             T.if_then_else(hi_nibble == 14, T.cast(lut[14], T.bfloat16),
                             T.cast(lut[15], T.bfloat16))))))))))))))))
                    k_bf16_smem[j, 2 * d2]     = lo_val * scale_bf16
                    k_bf16_smem[j, 2 * d2 + 1] = hi_val * scale_bf16

                # BF16 K × BF16 Q^T → FP32 logits[B, H]
                logits = T.alloc_fragment((B, H), T.float32)
                T.gemm(
                    k_bf16_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                # ReLU + weight scaling + reduce across heads
                for h, j in T.Parallel(H, B):
                    logits[j, h] = T.max(logits[j, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), T.float32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.copy(logits_sum, o[bx, i * B])

    fp4_paged_mqa_logits_.__annotations__ = {
        "q":           T.Tensor[(N, H, D), T.float8_e4m3],
        "kvcache_fp4": T.StridedTensor[(C, B, D_fp4), (d_0, D_fp4, 1), T.uint8],
        "kvcache_fe8m0": T.StridedTensor[(C, B, S_fe8m0), (d_1, S_fe8m0, 1), T.uint8],
        "weight":      T.Tensor[(N, H), T.float32],
        "seq_lens":    T.Tensor[(N,), T.int32],
        "page_table":  T.Tensor[(N, L), T.int32],
        "o":           T.Tensor[(N, S), T.float32],
    }
    return T.prim_func(fp4_paged_mqa_logits_)


if _tilelang_available:

    @tilelang.jit(pass_configs=_FP4_PASS_CONFIGS)
    def _fp4_paged_mqa_logits_kernel(
        head_dim: int = 128,
        num_heads: int = 64,
        block_size: int = 64,
        fp4_block_size: int = 32,
        clear_accum: bool = True,
    ):
        """FP4 paged MQA logits kernel (PR #23686 style: K dequant FP4→BF16, BF16×BF16 GEMM).

        Q: (N, H, D) FP8-e4m3  — converted to BF16 in smem
        K: MXFP4 packed (FP4 nibbles + FE8M0 scales) — dequanted to BF16 on-the-fly
        output: (N, S) FP32  — ReLU(K @ Q^T) * weight, summed across heads
        """
        N = T.dynamic("batch_size")
        L = T.dynamic("max_table_length")
        S = T.dynamic("max_seq_len")
        C = T.dynamic("num_blocks")
        B = block_size
        D = head_dim
        H = num_heads
        D_fp4 = D // 2
        S_fe8m0 = D // fp4_block_size
        d_0, d_1 = T.dynamic("d_0"), T.dynamic("d_1")

        assert D == 128
        assert H % 4 == 0

        return _fp4_build_paged_mqa_prim_func(
            N, L, S, C, B, D, H, D_fp4, S_fe8m0, d_0, d_1,
            fp4_block_size, clear_accum, _FP4_LUT,
        )


def tilelang_fp4_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp4: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """FP4 K-cache paged MQA logits (W4A16-style: K dequant FP4->BF16, BF16xBF16 wgmma).

    Args:
        q_fp8: (batch, 1, num_heads, head_dim) FP8
        kvcache_fp4: (num_pages, block_size, 1, 68) uint8
                     68 = 64 bytes FP4-packed K + 4 bytes FE8M0 scale
    Returns:
        logits: (batch, max_seq_len) FP32
    """
    if not _tilelang_available:
        raise RuntimeError(
            "tilelang_fp4_paged_mqa_logits requires TileLang. "
            "Install with: pip install tilelang"
        )
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError(
            "tilelang_fp4_paged_mqa_logits requires SM90 (Hopper) or later. "
            f"Current device: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
        )
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    # kvcache_fp4: (num_pages, block_size, 1, token_bytes) uint8
    num_pages = kvcache_fp4.shape[0]
    block_size = kvcache_fp4.shape[1]
    fp4_block_size = 32
    k_fp4_bytes = head_dim // 2      # 64 bytes of packed FP4 K per token
    fe8m0_bytes = head_dim // fp4_block_size  # 4 bytes of FE8M0 scales per token
    token_bytes = k_fp4_bytes + fe8m0_bytes   # 68

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp4.shape[1:] == (block_size, 1, token_bytes)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    # Flatten to (num_pages, block_size, token_bytes) for stride computation
    # Memory layout: for each (page, tok), the token_bytes are laid out as
    #   [fp4[0..63] | fe8m0[0..3]]  contiguously.
    # Stride for the page dimension = block_size * token_bytes.
    kv_flat = kvcache_fp4.view(num_pages, block_size, token_bytes)  # (C, B, 68)
    page_stride = block_size * token_bytes  # stride in bytes across pages

    # Create StridedTensors pointing into the interleaved buffer with explicit strides.
    # TileLang StridedTensor[(C, B, D_fp4), (d_0, D_fp4, 1)] means:
    #   element[c, b, d] = base_ptr + c * d_0 + b * D_fp4 + d * 1
    # For FP4 part: base = start of kv_flat, d_0 = page_stride, second stride = D_fp4 (= 64)
    # For FE8M0 part: base = kv_flat offset by k_fp4_bytes per token, d_0 = page_stride, second stride = S_fe8m0 (= 4)
    kv_fp4 = kv_flat[:, :, :k_fp4_bytes].contiguous()   # (C, B, 64) uint8, fp4 bytes
    kv_fe8m0 = kv_flat[:, :, k_fp4_bytes:].contiguous() # (C, B, 4)  uint8, fe8m0 bytes

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    kernel = _fp4_paged_mqa_logits_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=block_size,
        fp4_block_size=fp4_block_size,
        clear_accum=clean_logits,
    )
    q_flat = q_fp8.view(batch_size, num_heads, head_dim)
    kernel(q_flat, kv_fp4, kv_fe8m0, weight, seq_lens, page_table, logits)
    return logits


@triton.jit
def create_paged_compress_data_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_index_mapping_ptr,
    out_0_ptr,
    out_1_ptr,
    batch_size,
    stride_req_to_token_0,
    stride_req_to_token_1: tl.constexpr,
    stride_out_1_0,
    stride_out_1_1: tl.constexpr,
    compress_ratio: tl.constexpr,
    is_overlap: tl.constexpr,
    swa_page_size: tl.constexpr,
    ring_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size

    rid = tl.load(req_pool_indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    extend_len = tl.load(extend_seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr
    v0 = tl.zeros([BLOCK], tl.int32)
    v1 = tl.zeros([BLOCK], tl.int32)
    v2 = tl.zeros([BLOCK], tl.int32)
    v3 = tl.zeros([BLOCK], tl.int32)

    for i in tl.static_range(4):
        if i == 0:
            pos = load_pos
        elif i == 1:
            pos = write_pos
        elif i == 2:
            pos = load_overlap_pos
        else:
            pos = write_overlap_pos
        pos = tl.maximum(pos, 0)
        loc = tl.load(
            req_to_token_ptr
            + rid.to(tl.int64) * stride_req_to_token_0
            + pos.to(tl.int64) * stride_req_to_token_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc, mask=mask, other=0).to(
            tl.int32
        )
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        if i == 0:
            v0 = state_loc
        elif i == 1:
            v1 = state_loc
        elif i == 2:
            v2 = state_loc
        else:
            v3 = state_loc

    tl.store(out_0_ptr + offs, v1, mask=mask)

    if is_overlap:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v2, mask=mask)
        tl.store(base + 1 * stride_out_1_1, v0, mask=mask)
        tl.store(base + 2 * stride_out_1_1, v3, mask=mask)
        tl.store(base + 3 * stride_out_1_1, write_pos.to(tl.int32), mask=mask)
    else:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v0, mask=mask)


_mmap_dumper = None


def _get_mmap_dumper():
    global _mmap_dumper
    if _mmap_dumper is None:
        from sglang.srt.debug_utils.mmap_dumper import MmapDumper
        from sglang.srt.environ import envs

        dump_dir = envs.SGLANG_HACK_DEBUG_DUMP_CREATE_PAGED_COMPRESS_DATA.get()
        _mmap_dumper = MmapDumper(dump_dir or None)
    return _mmap_dumper


_dumped_static_meta_once = False


def _maybe_dump_create_paged_compress_data_inputs(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int,
) -> None:
    d = _get_mmap_dumper()
    if not d.is_active():
        return

    # Print static config (constant after server init) once per process.
    global _dumped_static_meta_once
    if not _dumped_static_meta_once:
        print(
            f"[c128_dump_static] swa_page_size={swa_page_size} ring_size={ring_size} "
            f"block={block} req_to_token_shape={tuple(req_to_token.shape)} "
            f"full_to_swa_shape={tuple(full_to_swa_index_mapping.shape)}",
            flush=True,
        )
        _dumped_static_meta_once = True

    # Per-ratio dump (small): req_pool_indices / seq_lens / extend_seq_lens.
    # These are forward_batch fields, identical across c4 and c128 within the
    # same forward — but small (KB-level) so dumping twice is cheap.
    p = f"c{compress_ratio}_plan"
    d.dump(
        {
            f"{p}_compress_ratio": compress_ratio,
            f"{p}_is_overlap": is_overlap,
            f"{p}_req_pool_indices": req_pool_indices,
            f"{p}_seq_lens": seq_lens,
            f"{p}_extend_seq_lens": extend_seq_lens,
        }
    )

    # Global tensors shared between c4 and c128 (multi-MB-GB). Only dump on
    # the first call per forward to avoid 2x GPU->CPU copy (~184 MB + ~33 MB).
    # Backends call create_paged_compressor_data with c4 first, then c128
    # (deepseek_v4_backend_radix.py:457-458), so dump on c4 only.
    if compress_ratio == 4:
        cols = min(10000, req_to_token.shape[1])
        req_to_token_partial = req_to_token[:, :cols].contiguous()
        d.dump(
            {
                "global_req_to_token_dumped_cols": cols,
                "global_req_to_token_partial": req_to_token_partial,
                "global_full_to_swa_index_mapping": full_to_swa_index_mapping,
            }
        )


def _maybe_dump_create_paged_compress_data_outputs(
    *,
    compress_ratio: int,
    out_0: torch.Tensor,
    out_1: torch.Tensor,
) -> None:
    d = _get_mmap_dumper()
    if not d.is_active():
        return
    p = f"c{compress_ratio}_plan"
    d.dump(
        {
            f"{p}_out_0": out_0,
            f"{p}_out_1": out_1,
            f"{p}_out_0_shape": list(out_0.shape),
            f"{p}_out_1_shape": list(out_1.shape),
        }
    )


_printed_buffer_shape_once: dict = {}


def maybe_dump_compress_metadata_extras(
    *,
    compress_ratio: int,
    kv_score_buffer_shape: Tuple[int, ...],
    kv_score_buffer_dtype: torch.dtype,
    plan_compress_plan: torch.Tensor,
    plan_write_plan: torch.Tensor,
) -> None:
    """Public helper to be called from compressor.py at metadata-prepare time
    (once per forward per ratio, not per layer). Dumps the prefill kernel's
    real bound (kv_score_buffer.shape) plus the actual plan tensors that get
    fed to flash_c{ratio}_prefill.
    """
    d = _get_mmap_dumper()
    if not d.is_active():
        return

    # Print kv_score_buffer.shape once per ratio (constant after init).
    if compress_ratio not in _printed_buffer_shape_once:
        print(
            f"[c128_dump_static] c{compress_ratio} "
            f"kv_score_buffer_shape={tuple(kv_score_buffer_shape)} "
            f"dtype={kv_score_buffer_dtype}",
            flush=True,
        )
        _printed_buffer_shape_once[compress_ratio] = True

    p = f"c{compress_ratio}_meta"
    d.dump(
        {
            f"{p}_plan_compress_plan": plan_compress_plan,
            f"{p}_plan_write_plan": plan_write_plan,
            f"{p}_plan_compress_count": int(plan_compress_plan.shape[0]),
            f"{p}_plan_write_count": int(plan_write_plan.shape[0]),
        }
    )


def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _should_dump = bool(envs.SGLANG_HACK_DEBUG_DUMP_CREATE_PAGED_COMPRESS_DATA.get())
    if _should_dump:
        torch.cuda.synchronize()
        _maybe_dump_create_paged_compress_data_inputs(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=full_to_swa_index_mapping,
            block=block,
        )

    batch_size = req_pool_indices.shape[0]
    out_dim = 4 if is_overlap else 1
    device_args: dict = dict(device=req_pool_indices.device, dtype=torch.int32)
    out_0 = torch.empty((batch_size,), **device_args)
    out_1 = torch.empty((batch_size, out_dim), **device_args)
    grid = (triton.cdiv(batch_size, block),)
    create_paged_compress_data_kernel[grid](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        req_to_token,
        full_to_swa_index_mapping,
        out_0,
        out_1,
        batch_size=batch_size,
        stride_req_to_token_0=req_to_token.stride(0),
        stride_req_to_token_1=req_to_token.stride(1),
        stride_out_1_0=out_1.stride(0),
        stride_out_1_1=out_1.stride(1),
        compress_ratio=compress_ratio,
        is_overlap=1 if is_overlap else 0,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        BLOCK=block,
    )

    if _should_dump:
        torch.cuda.synchronize()
        _maybe_dump_create_paged_compress_data_outputs(
            compress_ratio=compress_ratio, out_0=out_0, out_1=out_1
        )

    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    module = _jit_fused_store_module(
        name=type,
        input_dtype=input.dtype,
        index_dtype=indices.dtype,
        page_size=page_size,
    )
    module.run(input, cache, indices)


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    # Fallback path is hacky on purpose: when the mega-moe-memory flag is off
    # we must be bitwise-identical to the optimize branch, which used the
    # pre-refactor kernel.
    from sglang.srt.environ import envs

    deepseek_v4_moe_code_path_checker.observed += 1
    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        module = _jit_silu_and_mul_clamp_module(input.dtype)
    else:
        module = _jit_silu_and_mul_clamp_tmp_module(input.dtype)
    module.run(input, output, float(swiglu_limit))


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    if apply_swiglu_limit:
        deepseek_v4_moe_code_path_checker.observed += 1
    if swizzle:
        module = _jit_silu_mul_quant_varlen_module(
            quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
        )
    else:
        module = _jit_silu_mul_quant_tmp_module(
            quant_group_size, scale_ue8m0, apply_swiglu_limit
        )
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def silu_and_mul_contig_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    if apply_swiglu_limit:
        deepseek_v4_moe_code_path_checker.observed += 1
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    module = _jit_mega_moe_pre_dispatch_module(quant_group_size)
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    assert page_size == 64
    seq_lens = seq_lens.view(-1).to(torch.int32)
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    module = _jit_metadata_module()
    module.run(seq_lens, metadata)
    return metadata


def rmsnorm_self(q: torch.Tensor, eps: float) -> torch.Tensor:
    module = _jit_rmsnorm_head_module(q.shape[-1], q.dtype)
    out = q.new_empty(q.shape)
    module.run_self(q, out, eps)
    return out



def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sglang.srt.environ import envs

    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

    if algo == "auto":
        from sglang.srt.layers.linear_bf16_fp32.selector import pick_backend

        algo = pick_backend(m=x.size(0), n=y.size(0), k=x.size(1))

    return _dispatch_bf16_fp32_backend(x, y, algo=algo)


def _dispatch_bf16_fp32_backend(
    x: torch.Tensor, y: torch.Tensor, *, algo: str
) -> torch.Tensor:
    if algo == "cublas":
        # cuBLAS BF16xBF16 -> FP32 GEMM via PyTorch native API (torch >= 2.9).
        # Bit-exact and matches the previous JIT cublasGemmEx kernel.
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    elif algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    else:
        return torch.nn.functional.linear(x.float(), y.float())


def _compile_one(*input_tuple) -> None:
    name, job_fn, *args = input_tuple
    print(f"Compiling {name}...", flush=True)
    job_fn(*args)
    print(f"Finished compiling {name}.", flush=True)


def compile_aot():
    c_dtype = torch.float32
    jobs = [
        ("common", _jit_common_module),
        ("mask_topk", _jit_mask_topk_module),
        ("topk", _jit_topk_module),
        ("topk_v2", _jit_topk_v2_module),
        ("hash_topk", _jit_hash_topk_module),
        ("rope", _jit_fused_rope_module),
        ("metadata", _jit_metadata_module),
        (
            "compress_128_4",
            _jit_compress_module,
            128,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_4",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_128",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            128,
        ),
        (
            "norm_rope_128_64",
            _jit_norm_rope_module,
            c_dtype,
            128,
            64,
        ),
        (
            "norm_rope_512_64",
            _jit_norm_rope_module,
            c_dtype,
            512,
            64,
        ),
        (
            "store_flashmla_bf16_swa_256",
            _jit_fused_store_module,
            "flashmla",
            torch.bfloat16,
            torch.int32,
            256,
        ),
        (
            "store_flashmla_fp32_c4_64",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "store_flashmla_fp32_c128_2",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            2,
        ),
        (
            "store_indexer_fp32_c4_64",
            _jit_fused_store_module,
            "indexer",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "rmsnorm_head_512_bf16",
            _jit_rmsnorm_head_module,
            512,
            torch.bfloat16,
        ),
    ]
    import multiprocessing

    max_parallel_jobs = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        pool.starmap(_compile_one, jobs)


if __name__ == "__main__":
    compile_aot()
