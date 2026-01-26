from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, Tuple
import torch
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.layers.moe.usc.decode_cache import DecodeCache

from sglang.srt.batch_overlap.operations import _StateDict
from sglang.srt.utils import is_non_idle_and_non_empty
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import VarlenTopKOutput

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import (
        TopK,
        TopKOutput, 
    )
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

class USCMoECache(DecodeCache):
    def __init__(
        self,
        index_estimator: Callable,
        rank: int,
        world_size: int,
        device_group: torch.distributed.ProcessGroup,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = True,
        return_recv_hook: bool = True,
    ):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // world_size
        self.device_group = device_group

        self.deepep_dispatcher = DeepEPDispatcher(
            group=device_group,
            router_topk=top_k,
            permute_fusion=True,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            async_finish=True,  # TODO
            return_recv_hook=True,
        )

        # cache params
        self.dispatch_result = None

        # next layer estimator
        self.index_estimator = index_estimator
        self.next_layer_args = None

        self.debug = 0


    def dispatch_decode(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk_m: Optional[torch.Tensor] = None,
    ):
        '''
        raw output: hidden_states (hidden, scale), topk_idx, topk_weights, masked_m (packed_recv_count), expected_m
        dispatch output: hidden_states (hidden, scale), topk_idx, topk_weights, masked_m (packed_recv_count), expected_m, topk_m, shuffle_idx, handle
        '''
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
            
        dispatch_result = self.deepep_dispatcher.dispatch(hidden_states, None, topk_idx, topk_weights, forward_batch)

        if topk_m is None:
            topk_m = torch.full((topk_idx.shape[0],), topk_idx.shape[1], device=topk_idx.device, dtype=topk_idx.dtype)

        shuffle_idx = self._get_shuffle_idx(
            self.deepep_dispatcher._low_latency_dispatcher.handle, 
            topk_idx.shape[0]
        )
        # Tuple[Tensor, Tensor, int, int, int]
        handle = tuple(item for item in self.deepep_dispatcher._low_latency_dispatcher.handle)

        self.dispatch_result = dispatch_result + (topk_m, shuffle_idx, handle)
        

    def combine_decode(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
        handle: Tuple[torch.Tensor, torch.Tensor, int, int, int] = None,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        # Get the correct dispatcher based on forward_batch mode
        impl = self.deepep_dispatcher._get_impl(forward_batch)
        if handle is not None:
            impl.handle = handle
        return self.deepep_dispatcher.combine(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_batch=forward_batch,
                overlap_args=overlap_args,
            )


    # def overlap_combine(self, args):
    #     with torch.cuda.stream(self.combine_stream):
    #         output = self.combine_decode(args)

    #     # dispatch the next layer
    #     if self.indexer_inputs is not None:
    #         # skip the first layer
    #         with torch.cuda.stream(self.dispatch_stream):
    #             next_layer_args = self.index_estimator(*self.indexer_inputs)
    #             self.dispatch_decode(*next_layer_args)

    #     return output


    def get_dispatch_result(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # verify
        hit_mask, miss_mask = self._verify_cache(topk_idx)
        if dist.is_initialized():
            dist.barrier(group=self.device_group)
            # print(f"[USC get_dispatch_result] Rank {rank}: _verify_cache done", flush=True)

        self.hit_global_results = self._get_hit_cache(hit_mask)
        
        miss_local_results = self._get_miss_cache(hidden_states, topk_idx, topk_weights, forward_batch, miss_mask)

        # correction - this calls dispatch_decode for miss tokens (distributed communication!)
        self.dispatch_decode(*miss_local_results)
        
        # 等待 DeepEP dispatch 通信完成
        torch.cuda.current_stream().synchronize()
        # print(f"[USC get_dispatch_result] Rank {rank}: dispatch_decode(miss) done", flush=True)
        
        self.miss_global_results = tuple(item for item in self.dispatch_result)
        # TODO: combine after computation
        scatter_results = self._correction(self.hit_global_results)

        if self.debug == 1:
            self.miss_local_results = miss_local_results

        self.indexer_inputs = (hidden_states, topk_idx, topk_weights, forward_batch)

        # clean up and return
        self._empty_cache()
        return scatter_results

    def _verify_cache(self, topk_idx: torch.Tensor):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # reset recall
        self._last_tp = 0
        self._last_fn = 0
        
        if self.dispatch_result is not None:
            cache_shuffle_idx = self.dispatch_result[6]
            if isinstance(cache_shuffle_idx, tuple):
                cache_shuffle_idx = cache_shuffle_idx[0]
            cache_masked_m = self.dispatch_result[3]

            max_m = topk_idx.shape[0] * self.world_size
            topk_m = torch.full((topk_idx.shape[0],), topk_idx.shape[1], device=topk_idx.device, dtype=topk_idx.dtype)
            # print(f"[USC _verify_cache] Rank {rank}: _shuffle_topk_idx start", flush=True)
            shuffle_index, masked_m = self._shuffle_topk_idx(topk_idx, max_m, topk_m)
            # print(f"[USC _verify_cache] Rank {rank}: _shuffle_topk_idx done", flush=True)

            max_cache_tokens = cache_shuffle_idx.shape[1] if cache_shuffle_idx.ndim > 1 else 0
            cache_masked_m = cache_masked_m.to(torch.int64).clamp(min=0, max=max_cache_tokens)
            num_cache_experts = cache_shuffle_idx.shape[0]

            hit_expert_mask = torch.zeros(
                (self.num_local_experts, max_cache_tokens),
                dtype=torch.bool,
                device=cache_shuffle_idx.device,
            )
            miss_expert_mask = torch.zeros_like(shuffle_index, dtype=torch.bool)

            max_iters = min(self.num_local_experts, num_cache_experts)
            for i in range(max_iters):
                cache_len = cache_masked_m[i].item() if i < cache_masked_m.shape[0] else 0
                mask_len = masked_m[i].item() if i < masked_m.shape[0] else 0
                cache_len = min(cache_len, max_cache_tokens)
                mask_len = min(mask_len, shuffle_index.shape[1])

                if cache_len > 0 and mask_len > 0:
                    hit_expert_mask[i, :cache_len] = torch.isin(
                        cache_shuffle_idx[i, :cache_len],
                        shuffle_index[i, :mask_len],
                        assume_unique=True,
                    )
                    miss_expert_mask[i, :mask_len] = torch.isin(
                        shuffle_index[i, :mask_len],
                        cache_shuffle_idx[i, :cache_len],
                        assume_unique=True,
                        invert=True,
                    )
                elif mask_len > 0:
                    miss_expert_mask[i, :mask_len] = torch.isin(
                        shuffle_index[i, :mask_len],
                        cache_shuffle_idx[i, :cache_len],
                        assume_unique=True,
                        invert=True,
                    )

            cache_topk_idx = self.dispatch_result[1]
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            miss_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)

            M = cache_topk_idx.shape[0]
            for i in range(M):
                hit_token_mask[i, :] = torch.isin(cache_topk_idx[i, :], topk_idx[i, :], assume_unique=True)
                miss_token_mask[i, :] = torch.isin(topk_idx[i, :], cache_topk_idx[i, :], assume_unique=True, invert=True)
            
            # calculate recall
            tp = hit_token_mask.sum().item()  # an activated expert is predicted
            fn = miss_token_mask.sum().item() # anactivated expert is not predicted
            self._last_tp = tp
            self._last_fn = fn
        else:
            hit_expert_mask = torch.zeros((self.num_experts, topk_idx.shape[0] * self.world_size), dtype=torch.bool, device='cuda')
            miss_expert_mask = torch.ones((self.num_local_experts, topk_idx.shape[0] * self.world_size), dtype=torch.bool, device='cuda')
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool, device='cuda')
            miss_token_mask = torch.ones_like(topk_idx, dtype=torch.bool, device='cuda')

        return (hit_expert_mask, hit_token_mask), (miss_expert_mask, miss_token_mask)


    def _get_hit_cache(self, hit_mask: Tuple[torch.Tensor, torch.Tensor]):
        # result from DeepEP, so hidden_states is of shape [num_experts, M, hidden_size]
        hidden_states, topk_idx, _, _, _, _, shuffle_idx, handle = self.dispatch_result

        hit_expert_mask, hit_token_mask = hit_mask

        # expert-centric
        hit_expert_topk_m = hit_expert_mask.sum(dim=1)

        hit_shuffle_idx = torch.full_like(shuffle_idx, -1)
        num_tokens = topk_idx.shape[0]

        hit_hidden_states = torch.zeros_like(hidden_states[0])
        hit_hidden_scale = torch.zeros_like(hidden_states[1])
        hidden_num_tokens = hit_expert_mask.shape[1]
        cache_hidden_states = hidden_states[0][:, :hidden_num_tokens]
        cache_hidden_scale = hidden_states[1][:, :hidden_num_tokens]

        hit_layout_range = torch.zeros_like(handle[1])
        int_mask = (2**32) - 1

        for e in range(self.num_local_experts):
            new_begin_idx = torch.tensor(0, dtype=torch.int64, device=hit_layout_range.device)
            for i in range(self.world_size):
                # get layout
                begin_idx = (handle[1][e, i] >> 32)
                count = (handle[1][e, i] & int_mask)
                # new layout
                rank_mask = hit_expert_mask[e, begin_idx:begin_idx+count]
                new_count = rank_mask.sum().to(torch.int64)
                if new_count > 0:
                    # update shuffle_idx
                    hit_shuffle_idx[e, new_begin_idx:new_begin_idx+new_count] = shuffle_idx[e, begin_idx:begin_idx+count][rank_mask] % num_tokens
                    # update hidden_states
                    hit_hidden_states[e, new_begin_idx:new_begin_idx+new_count] = cache_hidden_states[e, begin_idx:begin_idx+count][rank_mask]
                    hit_hidden_scale[e, new_begin_idx:new_begin_idx+new_count] = cache_hidden_scale[e, begin_idx:begin_idx+count][rank_mask]
                # update layout
                hit_layout_range[e, i] = (new_begin_idx << 32) | new_count
                new_begin_idx += new_count

        handle = (hit_shuffle_idx, hit_layout_range, *handle[2:])
        hit_hidden_tuple = (hit_hidden_states, hit_hidden_scale)
        
        # token-centric
        M = topk_idx.shape[0]
        hit_token_topk_m = hit_token_mask.sum(dim=1)

        hit_topk_idx = torch.full_like(topk_idx, -1)
        for i in range(M):
            if hit_token_topk_m[i] > 0:
                hit_topk_idx[i, :hit_token_topk_m[i]] = topk_idx[i, hit_token_mask[i]]

        return (
            hit_hidden_tuple,
            hit_topk_idx,
            None, # topk_weights must be updated
            hit_expert_topk_m.to(torch.int32), # masked_m
            hit_expert_topk_m.float().mean().ceil().long().item(), # expected_m
            hit_token_topk_m, # topk_m
            hit_shuffle_idx, 
            handle,
        )


    def _get_miss_cache(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, forward_batch: ForwardBatch, miss_mask: Tuple[torch.Tensor, torch.Tensor]):
        miss_token_mask = miss_mask[1]

        miss_token_topk_m = miss_token_mask.sum(dim=1)

        miss_topk_idx = torch.full_like(topk_idx, -1)
        miss_topk_weights = torch.zeros_like(topk_weights)
        for i in range(topk_idx.shape[0]):
            if miss_token_topk_m[i] > 0:
                miss_topk_idx[i, :miss_token_topk_m[i]] = topk_idx[i, miss_token_mask[i]]
                miss_topk_weights[i, :miss_token_topk_m[i]] = topk_weights[i, miss_token_mask[i]]

        # update hit_topk_weights
        hit_topk_weights = torch.zeros_like(topk_weights)
        hit_token_topk_m = self.hit_global_results[5]
        for i in range(topk_idx.shape[0]):
            if hit_token_topk_m[i] > 0:
                hit_topk_weights[i, :hit_token_topk_m[i]] = topk_weights[i, ~miss_token_mask[i]]
        self.hit_global_results = (*self.hit_global_results[:2], hit_topk_weights, *self.hit_global_results[3:])

        return (
            hidden_states,
            miss_topk_idx,
            miss_topk_weights,
            forward_batch,
            miss_token_topk_m,
        )
        
        
    def _correction(self, hit_results):
        # TODO: merge handle?
        cache_hidden_states, cache_topk_idx, cache_topk_weights, cache_masked_m, _, cache_topk_m, cache_shuffle_idx, cache_handle = self.dispatch_result

        hit_hidden_states, hit_topk_idx, hit_topk_weights, hit_masked_m, _, hit_topk_m, hit_shuffle_idx, hit_handle = hit_results

        num_topk = cache_topk_idx.shape[1]

        assert torch.allclose(cache_topk_m, num_topk - hit_topk_m)

        masked_m = cache_masked_m + hit_masked_m

        # merge hidden_states
        hidden_states_fp8 = torch.zeros_like(cache_hidden_states[0])
        hidden_states_scale = torch.zeros_like(cache_hidden_states[1])
        for i in range(self.num_local_experts):
            if cache_masked_m[i] > 0:
                hidden_states_fp8[i, :cache_masked_m[i]] = cache_hidden_states[0][i, :cache_masked_m[i]]
                hidden_states_scale[i, :cache_masked_m[i]] = cache_hidden_states[1][i, :cache_masked_m[i]]
            if hit_masked_m[i] > 0:
                hidden_states_fp8[i, cache_masked_m[i]:masked_m[i]] = hit_hidden_states[0][i, :hit_masked_m[i]]
                hidden_states_scale[i, cache_masked_m[i]:masked_m[i]] = hit_hidden_states[1][i, :hit_masked_m[i]]
        hidden_states = (hidden_states_fp8.contiguous(), hidden_states_scale.contiguous())

        # # TODO: merge handle
        # shuffle_idx = torch.full_like(cache_shuffle_idx, shuffle_idx[0, -1])
        # num_tokens = topk_idx.shape[0]
        # for i in range(self.num_local_experts):
        #     if hit_expert_topk_m[i] > 0:
        #         hit_shuffle_idx[i, :hit_expert_topk_m[i]] = shuffle_idx[i, hit_expert_mask[i]] % num_tokens

        # hit_layout_range = torch.zeros_like(handle[1])
        # int_mask = (2**32) - 1
        # for e in range(self.num_local_experts):
        #     new_begin_idx = torch.tensor(0, dtype=torch.int64, device=hit_layout_range.device)
        #     for i in range(self.world_size):
        #         begin_idx = (handle[1][e, i] >> 32)
        #         count = (handle[1][e, i] & int_mask)
                
        #         new_count = hit_expert_mask[e, begin_idx:begin_idx+count].sum().to(torch.int64)
        #         hit_layout_range[e, i] = (new_begin_idx << 32) | new_count
                
        #         new_begin_idx += new_count

        # handle = (hit_shuffle_idx, hit_layout_range, *handle[2:])

        # merge topk_idx and topk_weights
        M = cache_topk_idx.shape[0]
        topk_idx = torch.zeros_like(cache_topk_idx)
        topk_weights = torch.zeros_like(cache_topk_weights)
        for i in range(M):
            if cache_topk_m[i] > 0:
                topk_idx[i, :cache_topk_m[i]] = cache_topk_idx[i, :cache_topk_m[i]]
                topk_weights[i, :cache_topk_m[i]] = cache_topk_weights[i, :cache_topk_m[i]]
            if hit_topk_m[i] > 0:
                topk_idx[i, cache_topk_m[i]:] = hit_topk_idx[i, :hit_topk_m[i]]
                topk_weights[i, cache_topk_m[i]:] = hit_topk_weights[i, :hit_topk_m[i]]

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            masked_m,
        )


    def _empty_cache(self):
        self.dispatch_result = None


    def _get_shuffle_idx(self, handle: Tuple[torch.Tensor, torch.Tensor], num_tokens: int):
        recv_src_info = handle[0]
        recv_layout_range = handle[1]
        int_mask = (2**32) - 1
        actual_num_experts = recv_src_info.shape[0]
        actual_world_size = recv_layout_range.shape[1] if recv_layout_range.ndim > 1 else self.world_size
        max_m = recv_src_info.shape[1]
        shuffle_idx = torch.full((actual_num_experts, max_m), -1, device=recv_src_info.device, dtype=recv_src_info.dtype)

        for i in range(actual_world_size):
            begin_idx = (recv_layout_range[:, i] >> 32)
            count = (recv_layout_range[:, i] & int_mask)
            num_experts_to_iter = min(actual_num_experts, begin_idx.shape[0])
            for e in range(num_experts_to_iter):
                start = begin_idx[e].item()
                cnt = count[e].item()
                if start < 0 or cnt <= 0:
                    continue
                end = min(start + cnt, max_m)
                if start >= max_m:
                    continue
                shuffle_idx[e, start:end] = recv_src_info[e, start:end] + i * num_tokens

        return shuffle_idx


    def _shuffle_topk_idx(self, topk_idx: torch.Tensor, max_m: int, topk_m: torch.Tensor):
        '''
        Shuffle the topk_idx [num_experts, num_topk] to [num_experts, num_tokens]
        '''

        assert topk_m is not None

        # gather topk_idx with padding to the global max num_tokens
        num_tokens = topk_idx.shape[0]
        num_topk = topk_idx.shape[1]
        
        # Debug: check topk_m shape and dtype
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: topk_idx.shape={topk_idx.shape}, topk_m.shape={topk_m.shape}, topk_m.dtype={topk_m.dtype}", flush=True)

        group_world_size = dist.get_world_size(group=self.device_group) if dist.is_initialized() else 1
        dist.barrier(group=self.device_group)
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_reduce start", flush=True)

        max_tokens_tensor = torch.tensor([num_tokens], device=topk_idx.device, dtype=torch.int32)
        if dist.is_initialized() and group_world_size > 1:
            dist.all_reduce(max_tokens_tensor, op=dist.ReduceOp.MAX, group=self.device_group)
        max_tokens = int(max_tokens_tensor.item())
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_reduce done", flush=True)

        # 统一 dtype，避免不同 rank dtype 不一致导致 all_gather 死锁
        unified_idx_dtype = torch.int64
        unified_m_dtype = torch.int32
        topk_idx = topk_idx.to(unified_idx_dtype)
        topk_m = topk_m.to(unified_m_dtype)
        
        # If some ranks have fewer tokens, pad to max_tokens so shapes match for all_gather
        if num_tokens < max_tokens:
            padded_topk_idx = torch.full((max_tokens, num_topk), -1, dtype=unified_idx_dtype, device=topk_idx.device)
            padded_topk_idx[:num_tokens] = topk_idx
            padded_topk_m = torch.zeros((max_tokens,), dtype=unified_m_dtype, device=topk_idx.device)
            padded_topk_m[:num_tokens] = topk_m
        else:
            padded_topk_idx = topk_idx.contiguous()
            padded_topk_m = topk_m.contiguous()

        total_tokens = group_world_size * max_tokens

        dist.barrier(group=self.device_group)
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_gather topk_idx start", flush=True)
        all_topk_idx = torch.empty((group_world_size, max_tokens, num_topk), dtype=unified_idx_dtype, device=topk_idx.device)
        dist.all_gather_into_tensor(all_topk_idx, padded_topk_idx, group=self.device_group)
        all_topk_idx = all_topk_idx.view(total_tokens, num_topk)
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_gather topk_idx done", flush=True)

        dist.barrier(group=self.device_group)
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_gather topk_m start, padded_topk_m.shape={padded_topk_m.shape}, dtype={padded_topk_m.dtype}", flush=True)
        all_topk_m = torch.empty((group_world_size, max_tokens), dtype=unified_m_dtype, device=topk_idx.device)
        dist.all_gather_into_tensor(all_topk_m, padded_topk_m, group=self.device_group)
        all_topk_m = all_topk_m.view(total_tokens)
        # print(f"[USC _shuffle_topk_idx] Rank {self.rank}: all_gather topk_m done", flush=True)

        # 确保 max_m 至少为 1，避免创建空 tensor 导致越界
        safe_max_m = max(max_m, 1)
        shuffle_idx = torch.full((self.num_local_experts, safe_max_m), -1, device=topk_idx.device, dtype=unified_idx_dtype)
        masked_m = torch.zeros(self.num_local_experts, device=topk_idx.device, dtype=torch.int32)

        # 如果 max_m = 0 或 total_tokens = 0，直接返回空结果
        if max_m == 0 or total_tokens == 0:
            return shuffle_idx, masked_m

        for i in range(self.num_local_experts):
            expert_id = i + self.rank * self.num_local_experts
            for j in range(total_tokens):
                _m = min(int(all_topk_m[j].item()), topk_idx.shape[1])
                if _m <= 0:
                    continue
                cnt = (all_topk_idx[j, :_m] == expert_id).any()
                if cnt > 0:
                    # 边界检查
                    if masked_m[i] < safe_max_m:
                        shuffle_idx[i, masked_m[i]] = j
                        masked_m[i] += 1
        return shuffle_idx, masked_m

class USCTPMoECache(DecodeCache):
    def __init__(self):
        super().__init__()

        self.alt_stream = torch.cuda.Stream()

    def _verify_cache(
        self,
        topk_output: TopKOutput,
        cache_topk_output: Optional[TopKOutput] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx = topk_output.topk_ids

        # reset recall
        # TODO: CHECK THIS
        self._last_tp = 0
        self._last_fn = 0

        if cache_topk_output is not None:
            cache_topk_idx = cache_topk_output.topk_ids
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            miss_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)

            M = cache_topk_idx.shape[0]
            for i in range(M):
                hit_token_mask[i, :] = torch.isin(cache_topk_idx[i, :], topk_idx[i, :], assume_unique=True)
                miss_token_mask[i, :] = torch.isin(topk_idx[i, :], cache_topk_idx[i, :], assume_unique=True, invert=True)
            
            # calculate recall
            tp = hit_token_mask.sum().item()  # an activated expert is predicted
            fn = miss_token_mask.sum().item() # anactivated expert is not predicted
            self._last_tp = tp
            self._last_fn = fn
        else:
            hit_token_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            miss_token_mask = torch.ones_like(topk_idx, dtype=torch.bool)

        return hit_token_mask, miss_token_mask

    def index_estimate_a(
        self, 
        state: _StateDict, 
        has_usc_estimation: bool,
        layer_id: int,
        topk: TopK,
        next_layer_gate: Callable,
        next_layer_topk: Callable,
    ):
        with torch.cuda.stream(self.alt_stream):
            if is_non_idle_and_non_empty(
                state.forward_batch.forward_mode, state.hidden_states_mlp_input
            ) and has_usc_estimation:
                # router_logits: (num_tokens, n_experts)
                state.estimated_router = next_layer_gate(state.hidden_states_mlp_input)
                with get_global_expert_distribution_recorder().with_current_layer(
                    layer_id
                ):
                    state.estimated_topk = next_layer_topk(
                        hidden_states=state.hidden_states_mlp_input,
                        router_logits=state.estimated_router,
                        num_token_non_padded=state.forward_batch.num_token_non_padded,
                        expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                            layer_id=layer_id,
                        ),
                    )
            else:
                state.estimated_router = None
                state.estimated_topk = topk.empty_topk_output(state.hidden_states_mlp_input.device)

    def index_estimate_b(self):
        torch.cuda.current_stream().wait_stream(self.alt_stream)

    def _get_hit_cache(self, cache_topk_output: TopKOutput, hit_mask: torch.Tensor):
        topk_idx = cache_topk_output.topk_ids

        hit_token_mask = hit_mask

        M = topk_idx.shape[0]
        hit_topk_m = hit_token_mask.sum(dim=1)

        hit_topk_idx = torch.full_like(topk_idx, -1)
        for i in range(M):
            if hit_topk_m[i] > 0:
                hit_topk_idx[i, :hit_topk_m[i]] = topk_idx[i, hit_token_mask[i]]

        return VarlenTopKOutput(
            topk_weights=cache_topk_output.topk_weights,
            topk_ids=hit_topk_idx,
            router_logits=cache_topk_output.router_logits,
            topk_m=hit_topk_m,
        )

    def _get_miss_cache(self, topk_output: TopKOutput, miss_mask: torch.Tensor):
        topk_idx = topk_output.topk_ids

        miss_token_mask = miss_mask
        miss_topk_m = miss_token_mask.sum(dim=1)

        miss_topk_idx = torch.full_like(topk_idx, -1)
        for i in range(topk_idx.shape[0]):
            if miss_topk_m[i] > 0:
                miss_topk_idx[i, :miss_topk_m[i]] = topk_idx[i, miss_token_mask[i]]

        return VarlenTopKOutput(
                topk_weights=topk_output.topk_weights,
                topk_ids=miss_topk_idx,
                router_logits=topk_output.router_logits,
                topk_m=miss_topk_m,
            )

    def hit_forward_a(
        self, 
        state: _StateDict,
        experts: FusedMoE,
    ):
        with torch.cuda.stream(self.alt_stream):
            dispatch_output = experts.dispatcher.dispatch(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.estimated_topk,
            )
            # TODO: no_combine output with all topk computed
            combine_input = experts.run_moe_core(
                dispatch_output=dispatch_output,
            )
            # varlen combine output
            state.hit_varlen_combine_output = experts.dispatcher.combine(
                combine_input=combine_input,
            )

    def hit_forward_b(self):
        torch.cuda.current_stream().wait_stream(self.alt_stream)

    def miss_forward(
        self, 
        state: _StateDict,
        experts: FusedMoE,
        num_fused_shared_experts: int,
        shared_experts: torch.nn.Module,
    ):
        topk_output = self._get_miss_cache(
            state.estimated_topk, state.pop("miss_mask")
        )
        dispatch_output = experts.dispatcher.dispatch(
            hidden_states=state.hidden_states_mlp_input,
            topk_output=topk_output,
        )
        # support varlen topk moe
        combine_input = experts.run_moe_core(
            dispatch_output=dispatch_output,
        )
        # standard combine output with topk=-1 positions excluded from combine
        state.miss_varlen_combine_output = experts.dispatcher.combine(
            combine_input=combine_input,
        )
        if (num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            state.shared_output = shared_experts(state.hidden_states_mlp_input)
        else:
            state.shared_output = None

    def reduce(self, state: _StateDict):
        # TODO: get hit ones, miss ones, then reduce
        hit_combine = state.hit_varlen_combine_output
        miss_combine = state.miss_varlen_combine_output

        # TODO: get hit and miss ones


        # reduce
        return hit_results + miss_results