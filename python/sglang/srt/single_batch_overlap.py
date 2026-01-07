from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import os

from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLOutput
from sglang.srt.layers.moe.utils import is_sbo_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_int_env_var

# USC cache modes:
# SGLANG_ENABLE_USC_CACHE=1: Enable full hit/miss splitting 
def is_usc_cache_enabled():
    return os.environ.get("SGLANG_ENABLE_USC_CACHE", "0") == "1"

# Global hit rate statistics accumulator
class USCHitRateStats:
    """Calculate recall, output summary periodically.
    
    Definitions:
    - True Positive (TP): predicted expert is actually activated
    - False Negative (FN): activated expert is NOT predicted
    
    Recall = TP / (TP + FN) = correct predictions / total actual activations
    
    Statistics computed:
    - Overall recall (aggregated TP/FN across all calls)
    - Per-call recall: min / max / median / average
    """
    def __init__(self):
        self.total_tp = 0  # True Positive  (TP)
        self.total_fn = 0  # False Negative (FN)
        self.call_count = 0
        self.recall_history = []  # 记录每次调用的 recall 值
        self.log_interval = int(os.environ.get("SGLANG_USC_LOG_INTERVAL", "100"))

    def update(self, tp: int, fn: int, rank: int = 0):
        self.total_tp += tp
        self.total_fn += fn
        self.call_count += 1
        
        # 计算本次调用的 recall 并记录
        total = tp + fn
        recall = tp / total if total > 0 else 0.0
        self.recall_history.append(recall)
        
        # output summary periodically (only on rank 0)
        if rank == 0 and self.call_count % self.log_interval == 0:
            total_actual = self.total_tp + self.total_fn
            overall_recall = self.total_tp / total_actual if total_actual > 0 else 0.0
            
            # 计算 min/max/median/average
            if self.recall_history:
                sorted_recalls = sorted(self.recall_history)
                min_recall = sorted_recalls[0]
                max_recall = sorted_recalls[-1]
                avg_recall = sum(self.recall_history) / len(self.recall_history)
                n = len(sorted_recalls)
                median_recall = (sorted_recalls[n // 2] + sorted_recalls[(n - 1) // 2]) / 2
            else:
                min_recall = max_recall = avg_recall = median_recall = 0.0
            
            print(f"\n{'='*60}", flush=True)
            print(f"[USC Stats Summary] After {self.call_count} calls:", flush=True)
            print(f"  Overall: TP={self.total_tp}, FN={self.total_fn}, Recall={overall_recall*100:.2f}%", flush=True)
            print(f"  Per-call Recall: min={min_recall*100:.2f}%, max={max_recall*100:.2f}%, median={median_recall*100:.2f}%, avg={avg_recall*100:.2f}%", flush=True)
            print(f"{'='*60}", flush=True)

# Global instance
_usc_hit_rate_stats = USCHitRateStats()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.usc_moe_cache import USCMoECache 
    from sglang.srt.layers.moe.index_estimator import MoEIndexPredictor


class SboFlags:
    # TODO may have: "enable_dispatch_shared_one_stream_overlap", "enable_dispatch_gateup_gemm_two_stream_overlap", ...

    @classmethod
    def enable_combine_down_gemm_two_stream_overlap(cls):
        return (
            is_sbo_enabled()
            # currently only cutedsl backend supports it
            and get_moe_runner_backend().is_flashinfer_cutedsl()
        )

    @classmethod
    def enable_combine_shared_two_stream_overlap(cls):
        return is_sbo_enabled()

    @classmethod
    def fuse_shared_experts_inside_sbo(cls):
        # TODO after antgroup's PR, should be `... or cls.enable_dispatch_shared_one_stream_overlap()`
        return cls.enable_combine_shared_two_stream_overlap()


@dataclass
class CombineOverlapArgs:
    # this "overlap" flag means overlapping with down gemm, not the general two-stream overlap
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: int
    signal: Optional[torch.Tensor] = None
    threshold: int = -1


@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event


def execute_sbo_with_usc_cache(
    experts: "DeepEPMoE",
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    forward_batch: ForwardBatch,
    usc_cache: "USCMoECache",
    moe_index_predictor:"MoEIndexPredictor",
    predicted_topk_idx: Optional[torch.Tensor] = None,
    predicted_topk_weights: Optional[torch.Tensor] = None,
    next_layer_gate: Optional[torch.nn.Module] = None,
    next_layer_topk: Optional[Any] = None,
    layer_id: int = 0,
    forward_shared_experts: Optional[Callable[[], Any]] = None,
    alt_stream: Optional = None,      
):
    """ 
    Execute Single Batch Overlap with USC (Unified Sparse Cache) optimization.
    
    Two modes:
    1. SGLANG_ENABLE_USC_CACHE=1: Full USC mode with hit/miss splitting
       - Pre-dispatch using PREDICTED expert indices
       - Predict next layer's expert index 
       - Compare predicted vs actual to get hit/miss mask
       - For HIT tokens: use predicted dispatch result
       - For MISS tokens: re-dispatch with actual indices
       - Merge results using hit_mask
    
    2. SGLANG_ENABLE_USC_CACHE=0: 
       - Normal SBO flow (dispatch -> moe -> combine)
    
    Args:
        experts: DeepEPMoE layer
        hidden_states: Input hidden states [num_tokens, hidden_size]
        topk_idx: Actual expert indices computed by gate [num_tokens, top_k]
        topk_weights: Actual expert weights [num_tokens, top_k]
        forward_batch: Forward batch info
        usc_cache: USCMoECache instance 
        moe_index_predictor: MoEIndexPredictor instance
        predicted_topk_idx: Predicted expert indices from previous layer (optional)
        predicted_topk_weights: Predicted expert weights from previous layer (optional)
        next_layer_gate: Gate module of the next MoE layer (for index prediction)
        next_layer_topk: TopK module of the next MoE layer (for index prediction)
        layer_id: Current layer ID (default=0)
        forward_shared_experts: Function to forward shared experts
        alt_stream: Alternative CUDA stream
        
    Returns:
        Tuple of (final_hidden_states, shared_output)
    """
    import torch.distributed as dist
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    num_tokens = hidden_states.shape[0]
    
    # Track if we have a real prediction (not just using actual indices)
    has_real_prediction = (
        predicted_topk_idx is not None 
        and predicted_topk_weights is not None
        and predicted_topk_idx.shape[0] == num_tokens
        and predicted_topk_idx.shape[1] == topk_idx.shape[1]
    )
    
    # For USC mode, if no prediction available, use actual indices as "prediction"
    if not has_real_prediction:
        predicted_topk_idx = topk_idx.clone()
        predicted_topk_weights = topk_weights.clone()
        
    """
    _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """
    num_tokens = hidden_states.shape[0]
    max_dispatch_tokens = 200  # less than 256 for DeepEP low latency mode
    
    local_use_full_usc = (
        is_usc_cache_enabled()
        and usc_cache is not None
        and forward_batch.forward_mode.is_decode()
        and num_tokens <= max_dispatch_tokens  # token 数不能超过 DeepEP 限制
    )
    
    # CRITICAL: All ranks must agree on whether to use USC Full mode
    # If any rank can't use USC (e.g. usc_cache is None), all ranks must skip it
    # Otherwise we get deadlock in distributed communication
    if dist.is_initialized():
        # Use all_reduce with MIN to ensure all ranks agree (1 if all True, 0 if any False)
        use_full_usc_tensor = torch.tensor([1 if local_use_full_usc else 0], 
                                            dtype=torch.int32, device=hidden_states.device)
        dist.all_reduce(use_full_usc_tensor, op=dist.ReduceOp.MIN)
        use_full_usc = bool(use_full_usc_tensor.item())
    else:
        use_full_usc = local_use_full_usc

    shared_output = None

    if use_full_usc:
        # ========== Full USC Mode using usc_moe_cache ==========
        # 1. dispatch_decode(predicted) - uses predicted indices
        # 2. get_dispatch_result() - compares, splits hit/miss, dispatches miss
        # 2.1 predict next layer's expert index 
        # 3. hit stream: moe_impl + combine_decode(hit)
        # 4. miss stream: moe_impl + combine_decode(miss)
        # 5. hit + miss
        
        barrier_group = None
        if dist.is_initialized():
            barrier_group = getattr(usc_cache, "device_group", None)
            if barrier_group is None:
                dist_group = getattr(dist, "group", None)
                barrier_group = getattr(dist_group, "WORLD", None) if dist_group is not None else None

        # Step 1: Dispatch with predicted indices (all ranks do this)
        usc_cache.dispatch_decode(
            hidden_states, predicted_topk_idx, predicted_topk_weights, forward_batch
        )
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Step 1 - dispatch_decode done", flush=True)

        # Step 2.1: Predict next layer's expert index 
        hidden_states_for_prediction = hidden_states.clone()
        if next_layer_gate is not None and next_layer_topk is not None and hidden_states.shape[0] > 0:
            moe_index_predictor.predict_and_store(
            hidden_states=hidden_states_for_prediction,
            next_layer_gate=next_layer_gate,
            next_layer_topk=next_layer_topk,
            layer_id=layer_id,
            forward_batch=forward_batch,
        )

        # Step 2: Get dispatch result - compares predicted vs actual, splits hit/miss
        usc_cache.get_dispatch_result(
            hidden_states, topk_idx, topk_weights, forward_batch
        )
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Step 2 - get_dispatch_result done", flush=True)
        
        empty_hidden = hidden_states.new_zeros(hidden_states.shape) if hidden_states.numel() > 0 else hidden_states

        # Step 3: Process HIT stream
        hit_hidden_states, hit_topk_idx, hit_topk_weights, hit_masked_m, hit_expected_m, hit_topk_m, hit_shuffle_idx, hit_handle = \
            usc_cache.hit_global_results
        
        hit_dispatch_output = DeepEPLLOutput(
            hidden_states_fp8=hit_hidden_states,
            topk_idx=hit_topk_idx,
            topk_weights=hit_topk_weights,
            masked_m=hit_masked_m,
            expected_m=hit_expected_m,
        )
        
        hit_has_tokens = hit_expected_m > 0
        hit_combined = empty_hidden
        
        # 如果有 token，调用 moe_impl
        # 如果没有 token，创建 contiguous 的 placeholder
        if hit_has_tokens:
            hit_moe_output = experts.moe_impl(hit_dispatch_output, down_gemm_overlap_args=None)
        else:
            # 先释放不需要的内存
            del hit_dispatch_output
            torch.cuda.empty_cache()
            # shape 必须是 [num_local_experts, M, hidden]，且必须 contiguous
            hit_fp8 = hit_hidden_states[0] if isinstance(hit_hidden_states, tuple) else hit_hidden_states
            num_experts, M, _ = hit_fp8.shape
            hidden_size = hidden_states.shape[-1]
            # 必须分配 contiguous 内存，无法避免
            hit_moe_output = torch.zeros(
                (num_experts, M, hidden_size), device=hidden_states.device, dtype=torch.bfloat16
            )
        
        if dist.is_initialized():
            dist.barrier(group=barrier_group)
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Barrier before hit combine (has_tokens={hit_has_tokens}, group={group_label})", flush=True)
        
        # 所有 rank 都必须调用 combine_decode（DeepEP combine 是分布式通信）
        hit_combined = usc_cache.combine_decode(
            hit_moe_output,
            hit_topk_idx,
            hit_topk_weights,
            forward_batch,
            hit_handle,
        )
        
        # 释放内存（hit_dispatch_output 可能在 else 分支已删除）
        del hit_moe_output, hit_hidden_states
        if hit_has_tokens:
            del hit_dispatch_output
        
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Step 3 - hit combine_decode done", flush=True)

        # Step 4: Process MISS stream
        miss_hidden_states, miss_topk_idx, miss_topk_weights, miss_masked_m, miss_expected_m, miss_topk_m, _, miss_handle = \
            usc_cache.miss_global_results
        
        miss_has_tokens = miss_expected_m > 0

        miss_dispatch_output = DeepEPLLOutput(
            hidden_states_fp8=miss_hidden_states,
            topk_idx=miss_topk_idx,
            topk_weights=miss_topk_weights,
            masked_m=miss_masked_m,
            expected_m=miss_expected_m,
        )
        
        miss_combined = empty_hidden
        
        # 如果有 token，调用 moe_impl
        # 如果没有 token，创建 contiguous 的 placeholder
        if miss_has_tokens:
            miss_moe_output = experts.moe_impl(miss_dispatch_output, down_gemm_overlap_args=None)
        else:
            del miss_dispatch_output
            torch.cuda.empty_cache()
            miss_fp8 = miss_hidden_states[0] if isinstance(miss_hidden_states, tuple) else miss_hidden_states
            num_experts, M, _ = miss_fp8.shape
            hidden_size = hidden_states.shape[-1]
            miss_moe_output = torch.zeros(
                (num_experts, M, hidden_size), device=hidden_states.device, dtype=torch.bfloat16
            )
        
        if dist.is_initialized():
            dist.barrier(group=barrier_group)
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Barrier before miss combine (has_tokens={miss_has_tokens}, group={group_label})", flush=True)
        
        # 所有 rank 都必须调用 combine_decode（DeepEP combine 是分布式通信）
        miss_combined = usc_cache.combine_decode(
            miss_moe_output,
            miss_topk_idx,
            miss_topk_weights,
            forward_batch,
            miss_handle,
        )
        
        # 释放内存（miss_dispatch_output 可能在 else 分支已删除）
        del miss_moe_output, miss_hidden_states
        if miss_has_tokens:
            del miss_dispatch_output
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: Step 4 - miss combine_decode done", flush=True)

        # Step 5: add hit + miss
        final_hidden_states = hit_combined + miss_combined
        
        # Step 6: Calculate and log recall (使用 _verify_cache 中计算的值)
        tp = getattr(usc_cache, '_last_tp', 0)  # TP = hit count
        fn = getattr(usc_cache, '_last_fn', 0)  # FN = miss count
        if tp + fn > 0:
            # update global statistics
            _usc_hit_rate_stats.update(tp, fn, rank)
                
        # Handle shared experts
        if SboFlags.enable_combine_shared_two_stream_overlap() and forward_shared_experts is not None:
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                meta_overlap_args["compute_num_sms"]
            ):
                shared_output = forward_shared_experts()
        # print(f"[USC Full] Rank {rank} Layer {layer_id}: USC e2e flow completed", flush=True)
        
    else:
        # ========== Normal Mode (execute_sbo function) ==========
        dispatch_output = experts.dispatch(
            hidden_states, topk_idx, topk_weights, forward_batch
        )

        combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
            _compute_overlap_args(dispatch_output, alt_stream)
        )

        moe_output = experts.moe_impl(
            dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
        )
        if (e := meta_overlap_args.get("record_event_after_down")) is not None:
            e.record()

        if SboFlags.enable_combine_shared_two_stream_overlap() and forward_shared_experts is not None:
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                meta_overlap_args["compute_num_sms"]
            ):
                shared_output = forward_shared_experts()

        final_hidden_states = experts.combine(
            moe_output,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            forward_batch,
            overlap_args=combine_overlap_args,
        )

    
    return final_hidden_states, shared_output


def execute_sbo(
    forward_shared_experts: Callable[[], Any],
    experts: "DeepEPMoE",
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    forward_batch: ForwardBatch,
    alt_stream: Optional = None,
):
    shared_output = None

    dispatch_output = experts.dispatch(
        hidden_states, topk_idx, topk_weights, forward_batch
    )

    combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
        _compute_overlap_args(dispatch_output, alt_stream)
    )

    hidden_states = experts.moe_impl(
        dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
    )
    if (e := meta_overlap_args.get("record_event_after_down")) is not None:
        e.record()

    if SboFlags.enable_combine_shared_two_stream_overlap():
        with deep_gemm_wrapper.configure_deep_gemm_num_sms(
            meta_overlap_args["compute_num_sms"]
        ):
            shared_output = forward_shared_experts()

    hidden_states = experts.combine(
        hidden_states,
        dispatch_output.topk_idx,
        dispatch_output.topk_weights,
        forward_batch,
        overlap_args=combine_overlap_args,
    )

    return hidden_states, shared_output


def _compute_overlap_args(dispatch_output, alt_stream):
    if not (
        SboFlags.enable_combine_down_gemm_two_stream_overlap()
        or SboFlags.enable_combine_shared_two_stream_overlap()
    ):
        return None, None, {}

    hidden_states = dispatch_output.hidden_states_fp8
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    num_local_experts, num_tokens_static, hidden_dim = hidden_states.shape

    total_num_sms = torch.cuda.get_device_properties(
        device="cuda"
    ).multi_processor_count
    communicate_num_sms = get_int_env_var("SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS", 32)
    compute_num_sms = total_num_sms - communicate_num_sms

    assert alt_stream is not None
    combine_wait_event = torch.cuda.Event()
    combine_overlap_args = CombineOverlapArgs(
        overlap=False,
        num_sms=communicate_num_sms,
        stream=alt_stream,
        wait_event=combine_wait_event,
    )
    meta_overlap_args = dict(
        compute_num_sms=compute_num_sms,
    )
    down_gemm_overlap_args = None

    if SboFlags.enable_combine_down_gemm_two_stream_overlap():
        combine_signal = torch.zeros(
            num_local_experts, dtype=torch.uint32, device=hidden_states.device
        )

        down_gemm_overlap_args = DownGemmOverlapArgs(
            signal=combine_signal,
            start_event=combine_wait_event,
            num_sms=compute_num_sms,
        )
        combine_overlap_args.overlap = True
        combine_overlap_args.signal = combine_signal
        combine_overlap_args.threshold = compute_num_sms
    else:
        meta_overlap_args |= dict(
            record_event_after_down=combine_wait_event,
        )

    return combine_overlap_args, down_gemm_overlap_args, meta_overlap_args
