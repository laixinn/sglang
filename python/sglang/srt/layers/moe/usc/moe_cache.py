from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, Tuple
from contextlib import contextmanager

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.layers.moe.usc.decode_cache import DecodeCache

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe_post_sum
from sglang.srt.layers.moe.fused_moe_triton import override_config
from sgl_kernel import moe_usc_hit_replace

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import (
        TopKOutput, 
    )
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe import MoeRunner
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.layers.moe.token_dispatcher.standard import IntermediateDispatchOutput


@contextmanager
def override_moe_config(runner: MoeRunner, config: MoeRunnerConfig):
    old_config = runner.config
    runner.config = config
    try:
        yield
    finally:
        runner.config = old_config

# fork from https://github.com/chang-l/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/multi_stream_utils.py#L35
def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = aux_stream is not None

    if multi_stream:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


@triton.jit
def _verify_cache_mask_kernel(
    cache_topk_idx_ptr,
    topk_idx_ptr,
    hit_token_mask_ptr,
    miss_token_mask_ptr,
    stride_cache_m,
    stride_cache_k,
    stride_topk_m,
    stride_topk_k,
    stride_hit_mask_m,
    stride_hit_mask_k,
    stride_miss_mask_m,
    stride_miss_mask_k,
    K: int,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute hit and miss masks for cache verification.
    
    For each token i:
    - hit_token_mask[i, :] checks if cache_topk_idx[i, :] elements are in topk_idx[i, :]
    - miss_token_mask[i, :] checks if topk_idx[i, :] elements are NOT in cache_topk_idx[i, :]
    """
    pid_m = tl.program_id(0)
    
    # Load all values for this token
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < K
    
    cache_vals = tl.load(
        cache_topk_idx_ptr + pid_m * stride_cache_m + k_offsets * stride_cache_k,
        mask=k_mask,
        other=-2
    )
    topk_vals = tl.load(
        topk_idx_ptr + pid_m * stride_topk_m + k_offsets * stride_topk_k,
        mask=k_mask,
        other=-3
    )

    intersect_mask = (cache_vals[:, None] == topk_vals[None, :])

    hit_mask = tl.sum(intersect_mask, axis=0) > 0

    miss_mask = tl.sum(intersect_mask, axis=1) == 0
    
    # Store results
    tl.store(
        hit_token_mask_ptr + pid_m * stride_hit_mask_m + k_offsets * stride_hit_mask_k,
        hit_mask,
        mask=k_mask
    )
    tl.store(
        miss_token_mask_ptr + pid_m * stride_miss_mask_m + k_offsets * stride_miss_mask_k,
        miss_mask,
        mask=k_mask
    )


def verify_cache_mask_triton(
    cache_topk_idx: torch.Tensor,
    topk_idx: torch.Tensor,
):
    """
    Triton kernel wrapper for cache verification mask computation.
    
    Args:
        cache_topk_idx: [M, K] tensor of cached topk indices
        topk_idx: [M, K] tensor of current topk indices
        hit_token_mask: [M, K] output tensor for hit mask
        miss_token_mask: [M, K] output tensor for miss mask
    """
    M, K = cache_topk_idx.shape

    hit_token_mask = torch.zeros((M, K), dtype=torch.bool, device=cache_topk_idx.device)
    miss_token_mask = torch.zeros((M, K), dtype=torch.bool, device=cache_topk_idx.device)
    
    # Determine block sizes
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    
    grid = (M,)
    
    _verify_cache_mask_kernel[grid](
        cache_topk_idx,
        topk_idx,
        hit_token_mask,
        miss_token_mask,
        cache_topk_idx.stride(0),
        cache_topk_idx.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        hit_token_mask.stride(0),
        hit_token_mask.stride(1),
        miss_token_mask.stride(0),
        miss_token_mask.stride(1),
        K,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return hit_token_mask, miss_token_mask


class StreamTensorWrapper:
    def __init__(self, tensor: torch.Tensor, event: torch.cuda.Event, sync: bool = False):
        self.tensor = tensor
        self.event = event
        self.sync = sync

        if self.sync:
            self.event.wait()

    def get_tensor(self):
        if not self.sync:
            self.event.wait()
        return self.tensor


class USCTPMoECache(DecodeCache):
    def __init__(self, experts: FusedMoE):
        super().__init__()

        # hit moe runner, miss moe can support topk = -1 filtering
        self.hit_moe_runner_config = replace(
            experts.moe_runner_config, 
            no_combine=True, 
            inplace=False, 
            down_num_warps=4,
        )
        self.miss_moe_runner_config = replace(
            experts.moe_runner_config,
            up_num_warps=4,
        )
        self.experts = experts

        self.alt_stream = torch.cuda.Stream()
        self.alt_event0 = torch.cuda.Event()
        self.alt_event1 = torch.cuda.Event()

    def _verify_cache(
        self,
        topk_output: TopKOutput,
        cache_topk_output: Optional[TopKOutput] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx = topk_output.topk_ids

        if cache_topk_output is not None:
            cache_topk_idx = cache_topk_output.topk_ids

            hit_token_mask, miss_token_mask = verify_cache_mask_triton(cache_topk_idx, topk_idx)
        else:
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            miss_token_mask = torch.ones_like(topk_idx, dtype=torch.bool)

        return hit_token_mask, miss_token_mask

    def _get_miss_cache(self, topk_output: TopKOutput, miss_mask: torch.Tensor):
        # ensure topk_idx is no longer used outside
        miss_topk_idx = topk_output.topk_ids
        # miss_topk_weights = topk_output.topk_weights
        # miss_router_logits = topk_output.router_logits

        miss_topk_idx.masked_fill_(~miss_mask, -1)
        # miss_topk_weights.masked_fill_(~miss_mask, 0.0)
        # miss_router_logits.masked_fill_(~miss_mask, 0.0)

        # hit mask is separated
        return StandardTopKOutput(
            topk_weights=topk_output.topk_weights,
            topk_ids=miss_topk_idx,
            router_logits=topk_output.router_logits,
        )

    def _get_hit_cache(
        self, 
        estimated_topk: TopKOutput, 
        grounded_topk: TopKOutput,
        hit_mask: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> TopKOutput:
        # TODO: this kernel temporarily considers small topk, optimize this
        moe_usc_hit_replace(grounded_topk.topk_weights, miss_mask, estimated_topk.topk_weights, hit_mask, inplace=True)

        return estimated_topk

    def _async_execute(self, fn: Callable):
        self.alt_event0.record()
        with torch.cuda.stream(self.alt_stream):
            self.alt_event0.wait()
            result = fn()
            self.alt_event1.record()
        return StreamTensorWrapper(
            result,
            self.alt_event1
        )
    
    def index_estimate_a(
        self, 
        index_estimate_fn: Callable,
        **kwargs,
    ):
        return self._async_execute(
            lambda: index_estimate_fn(**kwargs)
        )

    def index_estimate_b(self, tensor_wrapper: StreamTensorWrapper):
        return tensor_wrapper.get_tensor()

    def hit_forward_a(
        self,
        experts: FusedMoE,
        hidden_states: torch.Tensor,
        estimated_topk: TopKOutput,
    ):
        def _fn():
            with override_moe_config(experts.quant_method.runner, self.hit_moe_runner_config):
                hit_varlen_combine_output = experts(
                    hidden_states=hidden_states,
                    topk_output=estimated_topk,
                )
            return hit_varlen_combine_output
        return self._async_execute(_fn)

    def hit_forward_stage_a(
        self,
        experts: Callable,
        hidden_states: torch.Tensor,
        estimated_topk: TopKOutput,
        stage: int,
        dispatch_output: Optional[IntermediateDispatchOutput] = None,
    ) -> IntermediateDispatchOutput:
        hit_varlen_intermediate_output = experts(
            hidden_states=hidden_states,
            topk_output=estimated_topk,
            stage=stage,
            dispatch_output=dispatch_output,
        )
        
        return hit_varlen_intermediate_output

    def hit_forward_b(self, wrapped_tensor: StreamTensorWrapper):
        return wrapped_tensor.get_tensor()

    def miss_forward(
        self, 
        experts: FusedMoE,
        hidden_states: torch.Tensor,
        grounded_topk: TopKOutput,
        miss_mask: torch.Tensor,
    ):
        topk_output = self._get_miss_cache(
            grounded_topk, miss_mask
        )

        # support varlen topk moe
        # standard combine output with topk=-1 positions excluded from combine
        with override_moe_config(experts.quant_method.runner, self.miss_moe_runner_config):
            miss_varlen_combine_output = experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )

        return miss_varlen_combine_output

    def reduce(
        self,
        hit_varlen_combine_output: torch.Tensor,
        miss_varlen_combine_output: torch.Tensor,
        hit_mask: torch.Tensor,
        miss_mask: torch.Tensor,
        routed_scaling_factor: float,
        estimated_topk: TopKOutput,
        grounded_topk: TopKOutput,
    ):
        # correct hit weights
        moe_usc_hit_replace(
            grounded_topk.topk_weights, 
            miss_mask, 
            estimated_topk.topk_weights, 
            hit_mask,
            inplace=True
        )
        hit_varlen_combine_output *= estimated_topk.topk_weights.unsqueeze(2)

        # get hit results, no need to apply miss mask
        hit_varlen_combine_output.masked_fill_(~hit_mask.unsqueeze(2), 0)

        hit_results = fused_moe_post_sum(
            hit_varlen_combine_output,
            routed_scaling_factor,
        )

        # reduce
        return hit_results + miss_varlen_combine_output


class USCEPMoECache(DecodeCache):
    def __init__(self, experts):
        super().__init__()

        self.experts = experts
        self.hit_tbo_index = 0
        self.miss_tbo_index = 1

        self.alt_stream = torch.cuda.Stream()
        self.alt_event0 = torch.cuda.Event()
        self.alt_event1 = torch.cuda.Event()

    def _async_execute(self, fn: Callable):
        self.alt_event0.record()
        with torch.cuda.stream(self.alt_stream):
            self.alt_event0.wait()
            result = fn()
            self.alt_event1.record()
        return StreamTensorWrapper(
            result,
            self.alt_event1
        )

    def index_estimate_a(
        self, 
        index_estimate_fn: Callable,
        **kwargs,
    ):
        return self._async_execute(
            lambda: index_estimate_fn(**kwargs)
        )

    def index_estimate_b(self, tensor_wrapper: StreamTensorWrapper):
        return tensor_wrapper.get_tensor()
    
    def forward_expert_a(self, dispatch_output):
        def _fn():
            return self.experts.run_moe_core(
                dispatch_output=dispatch_output,
            )
        
        return self._async_execute(_fn)
    
    def forward_expert_b(self, wrapped_tensor: StreamTensorWrapper):
        return wrapped_tensor.get_tensor()


    # hit operations
    def hit_dispatch_a(
        self,
        hidden_states: torch.Tensor,
        estimated_topk: TopKOutput,
    ):
        self.experts.dispatcher.dispatch_a(
            hidden_states=hidden_states,
            topk_output=estimated_topk,
            tbo_subbatch_index=self.hit_tbo_index,
        )

    def hit_dispatch_b(self):
        return self.experts.dispatcher.dispatch_b(
            tbo_subbatch_index=self.hit_tbo_index,
        )

    def hit_combine_a(
        self,
        hit_varlen_combine_output: torch.Tensor,
        miss_varlen_combine_output: torch.Tensor,
        hit_mask: torch.Tensor,
        miss_mask: torch.Tensor,
        routed_scaling_factor: float,
        estimated_topk: TopKOutput,
        grounded_topk: TopKOutput,
    ):
        # correct hit weights
        moe_usc_hit_replace(
            grounded_topk.topk_weights, 
            miss_mask, 
            estimated_topk.topk_weights, 
            hit_mask,
            inplace=True
        )
        hit_varlen_combine_output *= estimated_topk.topk_weights.unsqueeze(2)

        # get hit results, no need to apply miss mask
        hit_varlen_combine_output.masked_fill_(~hit_mask.unsqueeze(2), 0)

        # pack combine input
        self.experts.dispatcher.combine_a(
            hidden_states=hit_varlen_combine_output,
            tbo_subbatch_index=self.hit_tbo_index,
        )

    def hit_combine_b(self):
        return self.experts.dispatcher.combine_b(
            tbo_subbatch_index=self.hit_tbo_index,
        )


    # miss operations
    def miss_dispatch_a(
        self, 
        hidden_states: torch.Tensor,
        grounded_topk: TopKOutput,
        miss_mask: torch.Tensor,
    ):
        # TODO: implement EP _get_miss_cache
        topk_output = self._get_miss_cache(
            grounded_topk, miss_mask
        )

        self.experts.dispatcher.dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
            tbo_subbatch_index=self.miss_tbo_index,
        )

    def miss_dispatch_b(self):
        return self.experts.dispatcher.dispatch_b(
            tbo_subbatch_index=self.miss_tbo_index,
        )
    
    def miss_combine_a(
        self,
        miss_varlen_expert_output,
    ):
        self.experts.dispatcher.combine_a(
            hidden_states=miss_varlen_expert_output,
            tbo_subbatch_index=self.miss_tbo_index,
        )

    def miss_combine_b(self, hit_varlen_combine_output: torch.Tensor):
        return self.experts.dispatcher.combine_b(
            tbo_subbatch_index=self.miss_tbo_index,
        )


    def _verify_cache(
        self,
        topk_output: TopKOutput,
        cache_topk_output: Optional[TopKOutput] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dispatch_result is not None:
            # expert-centric
            cache_shuffle_idx = self.dispatch_result[6]
            cache_masked_m = self.dispatch_result[3]

            # TODO: optimize this
            # max_m = cache_masked_m.max()
            max_m = topk_idx.shape[0] * self.world_size

            topk_m = torch.full((topk_idx.shape[0],), topk_idx.shape[1], device=topk_idx.device, dtype=topk_idx.dtype)
            # TODO: optimize this, do shuffle when topk calculation
            shuffle_index, masked_m = self._shuffle_topk_idx(topk_idx, max_m, topk_m)

            hit_expert_mask = torch.zeros_like(cache_shuffle_idx, dtype=torch.bool)
            miss_expert_mask = torch.zeros_like(shuffle_index, dtype=torch.bool)

            for i in range(self.num_local_experts):
                # apply on cache shuffle_idx
                hit_expert_mask[i, :cache_masked_m[i]] = torch.isin(cache_shuffle_idx[i, :cache_masked_m[i]], shuffle_index[i, :masked_m[i]], assume_unique=True)
                # apply on new index
                miss_expert_mask[i, :masked_m[i]] = torch.isin(shuffle_index[i, :masked_m[i]], cache_shuffle_idx[i, :cache_masked_m[i]], assume_unique=True, invert=True)

            # token-centric
            cache_topk_idx = self.dispatch_result[1]

            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            miss_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)

            M = cache_topk_idx.shape[0]
            for i in range(M):
                # apply on cache topk_idx
                hit_token_mask[i, :] = torch.isin(cache_topk_idx[i, :], topk_idx[i, :], assume_unique=True)
                # apply on new index
                miss_token_mask[i, :] = torch.isin(topk_idx[i, :], cache_topk_idx[i, :], assume_unique=True, invert=True)
        else:
            hit_expert_mask = torch.zeros((self.num_experts, topk_idx.shape[0] * self.world_size), dtype=torch.bool, device='cuda')
            miss_expert_mask = torch.ones((self.num_local_experts, topk_idx.shape[0] * self.world_size), dtype=torch.bool, device='cuda')
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool, device='cuda')
            miss_token_mask = torch.ones_like(topk_idx, dtype=torch.bool, device='cuda')

        return (hit_expert_mask, hit_token_mask), (miss_expert_mask, miss_token_mask)

    def _shuffle_topk_idx(self, topk_idx: torch.Tensor, max_m: int, topk_m: torch.Tensor):
        '''
        Shuffle the topk_idx [num_experts, num_topk] to [num_experts, num_tokens]
        '''

        assert topk_m is not None

        # gather topk_idx
        num_tokens = topk_idx.shape[0]
        num_topk = topk_idx.shape[1]
        total_tokens = self.world_size * num_tokens
        all_topk_idx = torch.empty((self.world_size, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=self.device_group)
        all_topk_idx = all_topk_idx.view(total_tokens, num_topk)

        # gather topk_m
        all_topk_m = torch.empty((self.world_size, num_tokens), dtype=topk_m.dtype, device='cuda')
        dist.all_gather_into_tensor(all_topk_m, topk_m, group=self.device_group)
        all_topk_m = all_topk_m.view(total_tokens)

        shuffle_idx = torch.full((self.num_local_experts, max_m), -1, device=topk_idx.device, dtype=topk_idx.dtype)
        masked_m = torch.zeros(self.num_local_experts, device=topk_idx.device, dtype=torch.int32)

        for i in range(self.num_local_experts):
            expert_id = i + self.rank * self.num_local_experts
            for j in range(total_tokens):
                _m = min(all_topk_m[j], topk_idx.shape[1])
                cnt = (all_topk_idx[j, :_m] == expert_id).any()
                if cnt > 0:
                    shuffle_idx[i, masked_m[i]] = j
                    masked_m[i] += 1

        return shuffle_idx, masked_m