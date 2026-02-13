from typing import Tuple, Optional, Callable, List, TYPE_CHECKING
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_USC_DEBUG_SYNC = os.environ.get("USC_DEBUG_SYNC", "0") == "1"

# Try to import the fused sparse-merge kernel (miss-only QK + hit load + softmax + PV).
# Prefer flash_mla (FlashMLA-src build) which is already compiled;
# fall back to sgl_kernel once it is rebuilt with the new kernel.
_HAS_SPARSE_MERGE_KERNEL = False
_sparse_merge_fwd = None
try:
    from flash_mla import flash_mla_sparse_merge_fwd as _sparse_merge_fwd
    _HAS_SPARSE_MERGE_KERNEL = True
    logger.info("sparse_merge kernel loaded from flash_mla")
except (ImportError, AttributeError):
    try:
        from sgl_kernel.flash_mla import flash_mla_sparse_merge_fwd as _sparse_merge_fwd
        _HAS_SPARSE_MERGE_KERNEL = True
        logger.info("sparse_merge kernel loaded from sgl_kernel")
    except (ImportError, AttributeError):
        logger.info("sparse_merge kernel NOT available, using PyTorch fallback")


def _debug_sync(tag: str):
    """Synchronize CUDA and log if an error surfaces."""
    if not _USC_DEBUG_SYNC:
        return
    try:
        torch.cuda.synchronize()
    except Exception as e:
        logger.error(f"[USC_DEBUG_SYNC] CUDA error after '{tag}': {e}")
        raise

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class USCSparseAttnCache:    
    def __init__(
        self, 
        hidden_size: int = 0,
        n_heads: int = 0,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        topk: int = 2048,
        prefix: str = "",
        quant_config=None,
        enable_estimator: bool = False,
    ) -> None:
        # Hit rate statistics
        self._hit_rate_call_count = 0
        self._hit_rate_tp_total = 0
        self._hit_rate_fn_total = 0
        self._hit_rate_print_interval = 100
        
        # Async stream for hit attention
        self.alt_stream = torch.cuda.Stream()
        
        # CP info
        self.cp_size = 1
        self.cp_rank = 0
        self.use_cp = False
        
        # Index estimator
        self.topk = topk
        self.estimator = None
        self._estimate_event = None
        self._estimated_index = None

    def set_cp_info(self, cp_size: int, cp_rank: int, use_cp: bool = True):
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.use_cp = use_cp and cp_size > 1

    def index_estimate_b(self, estimate_event) -> Optional[torch.Tensor]:
        return estimate_event

    def _update_hit_rate_stats(self, tp: int, fn: int):
        self._hit_rate_tp_total += tp
        self._hit_rate_fn_total += fn
        self._hit_rate_call_count += 1

        if self._hit_rate_call_count % self._hit_rate_print_interval == 0:
            total = self._hit_rate_tp_total + self._hit_rate_fn_total
            if total > 0:
                hit_rate = self._hit_rate_tp_total / total
                logger.info(
                    f"[USC Attn HIT RATE] {self._hit_rate_call_count} calls: "
                    f"TP={self._hit_rate_tp_total}, FN={self._hit_rate_fn_total}, "
                    f"Hit Rate={hit_rate:.4f} ({hit_rate*100:.2f}%)"
                )

    def forward_absorb_prepare_qkv_only(
        self,
        attn_module: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
        zero_allocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[tuple, Optional[torch.Tensor]]:
        from sglang.srt.layers.communicator import get_attn_tp_context
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp

        ctx = getattr(attn_module, "_usc_attn_ctx", {})
        per_token_group_quant_mla_deep_gemm_masked_fp8 = ctx.get(
            "per_token_group_quant_mla_deep_gemm_masked_fp8"
        )
        per_tensor_quant_mla_fp8 = ctx.get("per_tensor_quant_mla_fp8")
        bmm_fp8 = ctx.get("bmm_fp8")
        deep_gemm_wrapper = ctx.get("deep_gemm_wrapper")
        fused_rms_mxfp4_quant = ctx.get("fused_rms_mxfp4_quant")
        fused_rms_fp8_group_quant = ctx.get("fused_rms_fp8_group_quant")
        batched_gemm_afp4wfp4_pre_quant = ctx.get("batched_gemm_afp4wfp4_pre_quant")
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant = ctx.get(
            "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
        )
        _is_cublas_ge_129 = ctx.get("_is_cublas_ge_129", False)
        _is_hip = ctx.get("_is_hip", False)
        _use_aiter_gfx95 = ctx.get("_use_aiter_gfx95", False)
        _use_aiter = ctx.get("_use_aiter", False)
        _is_gfx95_supported = ctx.get("_is_gfx95_supported", False)

        q_lora = None
        topk_indices = None
        if attn_module.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [
                        attn_module.q_lora_rank,
                        attn_module.kv_lora_rank + attn_module.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : attn_module.kv_lora_rank]

            # overlap qk norm
            if attn_module.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                attn_module.alt_stream.wait_stream(current_stream)
                q = attn_module.q_a_layernorm(q)
                with torch.cuda.stream(attn_module.alt_stream):
                    k_nope = attn_module.kv_a_layernorm(k_nope)
                current_stream.wait_stream(attn_module.alt_stream)
            else:
                if _use_aiter_gfx95 and attn_module.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        attn_module.q_a_layernorm.weight,
                        attn_module.q_a_layernorm.variance_epsilon,
                        k_nope,
                        attn_module.kv_a_layernorm.weight,
                        attn_module.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and attn_module.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if attn_module.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                attn_module.q_a_layernorm.weight,
                                attn_module.q_a_layernorm.variance_epsilon,
                                k_nope,
                                attn_module.kv_a_layernorm.weight,
                                attn_module.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                attn_module.q_a_layernorm.weight,
                                attn_module.q_a_layernorm.variance_epsilon,
                                k_nope,
                                attn_module.kv_a_layernorm.weight,
                                attn_module.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = attn_module.q_a_layernorm(q)
                        k_nope = attn_module.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if attn_module.use_nsa and q_lora is None:
                q_lora = q

            if (
                attn_module.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                attn_module.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(attn_module.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = attn_module.q_b_proj(q)[0].view(
                        -1, attn_module.num_local_heads, attn_module.qk_head_dim
                    )
                current_stream.wait_stream(attn_module.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = attn_module.q_b_proj(q)[0].view(
                    -1, attn_module.num_local_heads, attn_module.qk_head_dim
                )
        else:
            q = attn_module.q_proj(hidden_states)[0].view(
                -1, attn_module.num_local_heads, attn_module.qk_head_dim
            )
            latent_cache = attn_module.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : attn_module.kv_lora_rank]
            k_nope = attn_module.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split(
            [attn_module.qk_nope_head_dim, attn_module.qk_rope_head_dim], dim=-1
        )
        k_pe = latent_cache[..., attn_module.kv_lora_rank :].unsqueeze(1)

        if attn_module.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (attn_module.num_local_heads, aligned_m, attn_module.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (attn_module.w_kc, attn_module.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and attn_module.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    attn_module.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    attn_module.w_kc.transpose(-2, -1),
                    attn_module.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if _use_aiter_gfx95 and attn_module.w_kc.dtype == torch.float8_e4m3fn:
                    q_nope_out = (
                        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                            X=q_nope,
                            WQ=attn_module.w_kc.transpose(-1, -2),
                            w_scale=attn_module.w_scale,
                            group_size=128,
                            YQ=None,  # allocate (B, M, N)
                            transpose_bm=False,  # (B, M, N)
                            transpose_bm_in=True,  # (M, B, K)
                            dtype=torch.bfloat16,
                        )
                    )
                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        attn_module.w_kc.to(torch.bfloat16) * attn_module.w_scale,
                    )

        elif attn_module.w_kc.dtype == torch.float8_e4m3fn:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val,
                attn_module.w_kc,
                q_nope_scale,
                attn_module.w_scale,
                torch.bfloat16,
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), attn_module.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if (
            attn_module.rotary_emb is not None
            and (not attn_module._fuse_rope_for_trtllm_mla(forward_batch))
            and (not _use_aiter or not _is_gfx95_supported or attn_module.use_nsa)
        ):
            q_pe, k_pe = attn_module.rotary_emb(positions, q_pe, k_pe)

        # CP expansion: all-gather KV across context-parallel ranks.
        # After this, k_nope/k_pe grow from N_local to N_total tokens.
        if nsa_use_prefill_cp(forward_batch):
            k_nope, k_pe = attn_module.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        # Save KV to pool AFTER CP expansion.
        # out_cache_loc has entries for ALL tokens (N_total), so k_nope/k_pe
        # must also have N_total rows.  Saving before CP expansion would cause
        # an OOB read in the Triton kernel because k_nope only had N_local
        # rows while out_cache_loc had N_total entries.
        cache_loc = (
            forward_batch.out_cache_loc
            if not getattr(attn_module, "is_cross_attention", False)
            else forward_batch.encoder_out_cache_loc
        )

        # Defensive: ensure the Triton kernel won't read OOB on k_nope/k_pe.
        # The kernel iterates pid_loc in [0, n_loc) and reads
        # cache_k_nope[pid_loc, :], so n_loc must <= k_nope.shape[0].
        n_loc = cache_loc.numel()
        n_kv = k_nope.shape[0]
        if n_loc > n_kv:
            logger.warning(
                f"[USC] set_mla_kv_buffer: cache_loc({n_loc}) > k_nope rows({n_kv}), "
                f"truncating cache_loc to avoid OOB in Triton kernel"
            )
            cache_loc = cache_loc[:n_kv]

        # Defensive: clamp cache_loc to pool buffer bounds to prevent OOB write
        kv_pool = forward_batch.token_to_kv_pool.get_key_buffer(
            attn_module.attn_mqa.layer_id
        )
        pool_sz = kv_pool.shape[0]
        cache_loc = cache_loc.clamp(max=pool_sz - 1)

        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            attn_module.attn_mqa,
            cache_loc,
            k_nope,
            k_pe,
        )
        _debug_sync("set_mla_kv_buffer")

        inner_state = (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

        return inner_state, q_lora

    def usc_prepare_qkv(self, state, attn_module: nn.Module):
        input_hidden = state.pop("hidden_states_after_comm_pre_attn")
        forward_batch = state.forward_batch
        attn_forward_method = attn_module.dispatch_attn_forward_method(forward_batch)
        AttnForwardMethod = (
            attn_module._usc_attn_ctx.get("AttnForwardMethod")
            if hasattr(attn_module, "_usc_attn_ctx")
            else None
        )
        if AttnForwardMethod is None:
            state.usc_attn_intermediate = attn_module.forward_prepare(
                positions=state.positions,
                hidden_states=input_hidden,
                forward_batch=forward_batch,
                zero_allocator=state.zero_allocator,
            )
            state._data["_usc_q_lora"] = None
            state._data["_usc_indexer_hidden_states"] = None
            state._data["_usc_q_all"] = None
            state._data["_usc_kv_pool"] = None
            state._data["_usc_sm_scale"] = None
            return

        if attn_forward_method != AttnForwardMethod.MLA:
            state.usc_attn_intermediate = attn_module.forward_prepare(
                positions=state.positions,
                hidden_states=input_hidden,
                forward_batch=forward_batch,
                zero_allocator=state.zero_allocator,
            )
            state._data["_usc_q_lora"] = None
            state._data["_usc_indexer_hidden_states"] = None
            state._data["_usc_q_all"] = None
            state._data["_usc_kv_pool"] = None
            state._data["_usc_sm_scale"] = None
            return

        inner_state, q_lora = self.forward_absorb_prepare_qkv_only(
            attn_module,
            positions=state.positions,
            hidden_states=input_hidden,
            forward_batch=forward_batch,
            zero_allocator=state.zero_allocator,
        )
        state.usc_attn_intermediate = (
            None,
            attn_forward_method,
            forward_batch,
            inner_state,
        )
        state._data["_usc_q_lora"] = q_lora
        state._data["_usc_indexer_hidden_states"] = input_hidden

        # ----- New split-QK: pre-compute q_all for hit_a / miss_reduce -----
        q_pe, q_nope_out = inner_state[0], inner_state[2]
        llama_4_scaling = inner_state[8]

        q_all = torch.cat([q_nope_out, q_pe], dim=-1)   # [s_q, h_q, d_qk]
        if llama_4_scaling is not None:
            q_all = q_all * llama_4_scaling

        # KV pool buffer: all indices from NSA indexer are PAGED (pool slot indices).
        # set_mla_kv_buffer already wrote k_nope/k_pe into this pool above.
        kv_pool = forward_batch.token_to_kv_pool.get_key_buffer(
            attn_module.attn_mqa.layer_id
        )

        state._data["_usc_q_all"] = q_all
        state._data["_usc_kv_pool"] = kv_pool
        state._data["_usc_sm_scale"] = attn_module.attn_mqa.scaling

    def usc_prepare_topk(self, state, attn_module: nn.Module):
        q_lora = state._data.pop("_usc_q_lora", None)
        indexer_hidden = state._data.pop("_usc_indexer_hidden_states", None)

        intermediate_state = state.usc_attn_intermediate
        if intermediate_state is None:
            state.attn_actual_sparse_index = None
            return

        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None or len(inner_state) < 8:
            state.attn_actual_sparse_index = None
            return

        AttnForwardMethod = (
            attn_module._usc_attn_ctx.get("AttnForwardMethod")
            if hasattr(attn_module, "_usc_attn_ctx")
            else None
        )
        existing_topk = inner_state[7]
        if AttnForwardMethod is None or attn_forward_method != AttnForwardMethod.MLA:
            state.attn_actual_sparse_index = existing_topk
            return

        if existing_topk is not None:
            state.attn_actual_sparse_index = existing_topk
            return

        if attn_module.use_nsa and q_lora is not None:
            hidden_states_for_indexer = indexer_hidden if indexer_hidden is not None else hidden_states
            topk_indices = attn_module.indexer(
                x=hidden_states_for_indexer,
                q_lora=q_lora,
                positions=state.positions,
                forward_batch=forward_batch,
                layer_id=attn_module.layer_id,
            )
        else:
            topk_indices = None

        if topk_indices is None:
            state.attn_actual_sparse_index = existing_topk
            return

        inner_state_copy = list(inner_state)
        inner_state_copy[7] = topk_indices
        # Avoid overriding existing key via __setattr__ guard
        state._data.pop("usc_attn_intermediate", None)
        state.usc_attn_intermediate = (
            hidden_states,
            attn_forward_method,
            forward_batch,
            tuple(inner_state_copy),
        )
        state.attn_actual_sparse_index = topk_indices

    @staticmethod
    def _resolve_kv_and_trim(
        indices: torch.Tensor,          # [s_q, topk], may be padded with -1
        kv_pool: torch.Tensor,          # [pool_size, 1, d_qk]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Trim padded indices and clamp to pool bounds.

        NSA indexer pads topk indices to multiples of 2048 with -1.
        All valid indices are KV pool slot indices (PAGED).

        Returns:
            safe_idx:    [s_q, K]   clamped to [0, pool_sz-1], invalid slots → 0
            valid_mask:  [s_q, K]   True where original index >= 0
            K:           int        number of columns after trimming
        """
        # Trim trailing -1 padding
        valid_full = indices >= 0                                # [s_q, topk]
        K = max(int(valid_full.sum(dim=-1).max().item()), 1)
        trimmed = indices[:, :K]                                 # [s_q, K]
        valid_mask = trimmed >= 0                                # [s_q, K]

        # Clamp to pool bounds (invalid → 0, OOB → pool_sz-1)
        pool_sz = kv_pool.shape[0]
        safe_idx = trimmed.clamp(min=0, max=pool_sz - 1)        # [s_q, K]

        return safe_idx, valid_mask, K

    def compute_qk_scores_pytorch(
        self,
        q_all: torch.Tensor,      # [s_q, h_q, d_qk]
        indices: torch.Tensor,     # [s_q, topk]  (PAGED pool slot indices, padded with -1)
        sm_scale: float,
        kv_pool: torch.Tensor,    # [pool_size, 1, d_qk]
    ) -> torch.Tensor:
        """Compute Q@K^T scores for given indices.

        Called from op_usc_hit_a with predicted_indices.

        Returns:
            qk_scores: [s_q, h_q, topk], float32, invalid positions = -inf
        """
        s_q, h_q, _ = q_all.shape
        topk = indices.shape[-1]
        device = q_all.device

        safe_idx, valid_mask, K = self._resolve_kv_and_trim(indices, kv_pool)

        if K == 0 or not valid_mask.any():
            return torch.full((s_q, h_q, topk), float('-inf'),
                              dtype=torch.float32, device=device)

        # Gather K: [s_q, K, d_qk]
        selected_k = kv_pool[safe_idx, 0, :]
        _debug_sync("compute_qk_scores_pytorch:gather_k")

        # Q @ K^T: [s_q, h_q, K]
        qk = torch.bmm(
            q_all.float().contiguous(),
            selected_k.float().transpose(1, 2).contiguous(),
        ) * sm_scale
        _debug_sync("compute_qk_scores_pytorch:bmm_qk")

        # Mask invalid → -inf
        qk.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))

        if K == topk:
            return qk

        # Pad back to original topk dimension
        out = torch.full((s_q, h_q, topk), float('-inf'),
                         dtype=torch.float32, device=device)
        out[:, :, :K] = qk
        return out

    def verify_cache_v2(
        self,
        actual_indices: Optional[torch.Tensor],
        predicted_indices: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compare predicted vs actual sparse index, return hit_mask and miss_mask.

        Args:
            actual_indices: [num_tokens, topk] actual sparse index
            predicted_indices: [num_tokens, topk] predicted sparse index
        
        Returns:
            hit_mask:  [num_tokens, topk] on actual_indices, True where hit
            miss_mask: [num_tokens, topk] on actual_indices, True where miss
        """
        if actual_indices is None:
            return None, None

        valid_actual = actual_indices >= 0  # [num_tokens, topk]

        if predicted_indices is None:
            return torch.zeros_like(valid_actual), valid_actual

        # Sort predicted for searchsorted (replace invalid with -2 so they sort first)
        safe_predicted = predicted_indices.clone()
        safe_predicted[safe_predicted < 0] = -2
        sorted_predicted, _ = safe_predicted.sort(dim=-1)

        # For each actual index, binary-search in sorted predicted
        search_values = actual_indices.clamp(min=0)
        positions = torch.searchsorted(sorted_predicted, search_values)
        positions = positions.clamp(max=sorted_predicted.shape[-1] - 1)

        # A match iff the value at the found position equals the actual index
        is_match = (
            (torch.gather(sorted_predicted, 1, positions) == actual_indices)
            & valid_actual
        )

        hit_mask = is_match
        miss_mask = valid_actual & ~hit_mask

        # Update hit rate statistics
        tp = hit_mask.sum().item()
        fn = miss_mask.sum().item()
        self._update_hit_rate_stats(tp, fn)

        return hit_mask, miss_mask

    def _apply_wvc_absorption(
        self,
        attn_module: nn.Module,
        attn_output: torch.Tensor,   # [s_q, h_q, kv_lora_rank]
        zero_allocator,
    ) -> torch.Tensor:
        """Apply w_vc value-absorption BMM.

        Transforms [s_q, h_q, kv_lora_rank] → [s_q, h_q, v_head_dim]
        by multiplying each head's output with w_vc weight matrix.
        """
        ctx = attn_module._usc_attn_ctx
        per_token_group_quant = ctx.get("per_token_group_quant_mla_deep_gemm_masked_fp8")
        per_tensor_quant = ctx.get("per_tensor_quant_mla_fp8")
        bmm_fp8 = ctx.get("bmm_fp8")
        deep_gemm_wrapper = ctx.get("deep_gemm_wrapper")
        is_in_piecewise_cuda_graph = ctx.get("is_in_piecewise_cuda_graph")
        is_cublas_ge_129 = ctx.get("_is_cublas_ge_129", False)

        if attn_module.use_deep_gemm_bmm:
            if per_token_group_quant is None or deep_gemm_wrapper is None:
                raise RuntimeError("USC Attention: missing deep_gemm deps for w_vc")
            av, ascale, masked_m, expected_m, aligned_m = (
                per_token_group_quant(attn_output.transpose(0, 1))
            )
            buf = attn_output.new_empty(
                (attn_module.num_local_heads, aligned_m, attn_module.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (av, ascale),
                (attn_module.w_vc, attn_module.w_scale_v),
                buf,
                masked_m,
                expected_m,
            )
            attn_bmm_output = buf[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
        elif attn_module.w_vc.dtype == torch.float8_e4m3fn:
            if per_tensor_quant is None or bmm_fp8 is None:
                raise RuntimeError("USC Attention: missing fp8 bmm deps for w_vc")
            av, ascale = per_tensor_quant(
                attn_output.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
                    if is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            attn_bmm_output = bmm_fp8(
                av, attn_module.w_vc, ascale, attn_module.w_scale, torch.bfloat16,
            ).transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph is not None and is_in_piecewise_cuda_graph():
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), attn_module.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], attn_module.num_local_heads * attn_module.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    attn_module.w_vc,
                    out=attn_bmm_output.view(
                        -1, attn_module.num_local_heads, attn_module.v_head_dim
                    ).transpose(0, 1),
                )

        return attn_bmm_output.view(
            attn_output.shape[0], attn_module.num_local_heads, attn_module.v_head_dim
        )

    def compute_miss_reduce(
        self,
        attn_module: nn.Module,
        q_all: torch.Tensor,                        # [s_q, h_q, d_qk]
        actual_indices: torch.Tensor,                # [s_q, topk]  (PAGED, padded with -1)
        hit_mask: Optional[torch.Tensor],            # [s_q, topk]
        miss_mask: Optional[torch.Tensor],           # [s_q, topk]
        predicted_qk: Optional[torch.Tensor],        # [s_q, h_q, pred_topk]
        predicted_indices: Optional[torch.Tensor],   # [s_q, pred_topk]
        sm_scale: float,
        zero_allocator,
        kv_pool: torch.Tensor,                      # [pool_size, 1, d_qk]
    ) -> torch.Tensor:
        """Merged miss + hit_b + reduce.

        Tries the fused CUDA kernel first (miss-only QK + hit load + softmax + PV).
        Falls back to the PyTorch implementation if the kernel is unavailable.

        Returns:
            attn_output_3d: [s_q, h_q, v_head_dim] after w_vc absorption
        """
        if (
            _HAS_SPARSE_MERGE_KERNEL
            and q_all.dtype == torch.bfloat16
        ):
            try:
                # logger.info("Using cuda kernel sparse merge kernel")
                return self._compute_miss_reduce_kernel(
                    attn_module, q_all, actual_indices, hit_mask, miss_mask,
                    predicted_qk, predicted_indices, sm_scale, zero_allocator, kv_pool,
                )
            except Exception as e:
                logger.warning(
                    "sparse_merge kernel failed, falling back to PyTorch: %s", e
                )
        return self._compute_miss_reduce_pytorch(
            attn_module, q_all, actual_indices, hit_mask, miss_mask,
            predicted_qk, predicted_indices, sm_scale, zero_allocator, kv_pool,
        )

    def _compute_miss_reduce_kernel(
        self,
        attn_module: nn.Module,
        q_all: torch.Tensor,
        actual_indices: torch.Tensor,
        hit_mask: Optional[torch.Tensor],
        miss_mask: Optional[torch.Tensor],
        predicted_qk: Optional[torch.Tensor],
        predicted_indices: Optional[torch.Tensor],
        sm_scale: float,
        zero_allocator,
        kv_pool: torch.Tensor,
    ) -> torch.Tensor:
        s_q, h_q, d_qk = q_all.shape
        kv_lora_rank = attn_module.kv_lora_rank
        device = q_all.device

        # ---- Resolve & trim ----
        safe_idx, valid_mask, K = self._resolve_kv_and_trim(actual_indices, kv_pool)
        if hit_mask is not None and hit_mask.shape[-1] > K:
            hit_mask = hit_mask[:, :K]

        has_hit = (
            hit_mask is not None and hit_mask.any()
            and predicted_qk is not None
            and predicted_indices is not None
        )

        # ---- Sort indices: miss first, hit next, invalid last ----
        sort_key = torch.zeros(s_q, K, dtype=torch.long, device=device)
        if has_hit:
            sort_key[hit_mask & valid_mask] = 1
        sort_key[~valid_mask] = 2
        _, perm = sort_key.sort(dim=-1, stable=True)           # [s_q, K]

        sorted_idx   = torch.gather(safe_idx, 1, perm)        # [s_q, K]
        sorted_valid = torch.gather(valid_mask, 1, perm)       # [s_q, K]
        sorted_hit   = torch.gather(
            hit_mask if has_hit else torch.zeros_like(valid_mask),
            1, perm,
        )                                                       # [s_q, K]

        # ---- Remap predicted_qk to sorted order ----
        if has_hit:
            sorted_orig = torch.gather(actual_indices[:, :K], 1, perm)
            remap_pos = self._remap_predicted_to_actual(
                predicted_indices, sorted_orig, h_q,
            )                                                   # [s_q, h_q, K]
            remapped_qk = torch.gather(predicted_qk, 2, remap_pos)
            # For miss/invalid positions, set 0 (kernel ignores these)
            not_hit = ~(sorted_hit & sorted_valid).unsqueeze(1).expand(-1, h_q, -1)
            remapped_qk = remapped_qk.masked_fill(not_hit, 0.0)
        else:
            remapped_qk = torch.zeros(
                s_q, h_q, K, dtype=torch.float32, device=device,
            )

        # ---- Pad topk to multiple of 128 (= 2 * B_TOPK) ----
        B_TOPK_2 = 128
        pad_len = (B_TOPK_2 - (K % B_TOPK_2)) % B_TOPK_2
        if pad_len > 0:
            sorted_idx   = F.pad(sorted_idx,   (0, pad_len), value=0)   # clamped; is_kv_valid=False
            sorted_valid = F.pad(sorted_valid,  (0, pad_len), value=False)
            # Mark padded positions as "hit" so kernel skips QK GEMM for pad blocks
            sorted_hit   = F.pad(sorted_hit.long(), (0, pad_len), value=1).bool()
            remapped_qk  = F.pad(remapped_qk,  (0, pad_len), value=0.0)
        topk_padded = sorted_idx.shape[-1]

        # ---- Build kernel-format tensors ----
        # indices:      [s_q, h_kv=1, topk_padded], int32
        # For invalid positions, set index to -1 so the kernel marks is_kv_valid=False
        kernel_indices = sorted_idx.clone()
        kernel_indices[~sorted_valid] = -1
        indices_3d  = kernel_indices.unsqueeze(1).to(torch.int32).contiguous()

        # hit_mask:     [s_q, h_kv=1, topk_padded], int32  (1=hit, 0=miss)
        hit_mask_3d = sorted_hit.unsqueeze(1).to(torch.int32).contiguous()

        # predicted_qk: [s_q, h_q, topk_padded], float32
        remapped_qk = remapped_qk.contiguous()

        # ---- Pad h_q to multiple of B_H=64 (WGMMA atom M-dimension) ----
        B_H_KERNEL = 64
        pad_h = (B_H_KERNEL - (h_q % B_H_KERNEL)) % B_H_KERNEL
        if pad_h > 0:
            # Pad Q with zeros → Q@K^T = 0 for padded heads → harmless
            q_kernel = F.pad(q_all, (0, 0, 0, pad_h)).contiguous()         # [s_q, h_q+pad_h, d_qk]
            qk_kernel = F.pad(remapped_qk, (0, 0, 0, pad_h)).contiguous() # [s_q, h_q+pad_h, topk_padded]
        else:
            q_kernel = q_all.contiguous()
            qk_kernel = remapped_qk

        # ---- Launch kernel ----
        output, _max_logits, _lse = _sparse_merge_fwd(
            q=q_kernel,
            kv=kv_pool.contiguous(),
            actual_indices=indices_3d,
            predicted_qk=qk_kernel,
            hit_mask=hit_mask_3d,
            sm_scale=sm_scale,
            d_v=kv_lora_rank,
        )
        # logger.info(f"output shape: {output.shape}")
        # output: [s_q, h_q+pad_h, kv_lora_rank], bf16

        # Slice back to original h_q
        if pad_h > 0:
            output = output[:, :h_q, :].contiguous()

        # ---- w_vc absorption ----
        return self._apply_wvc_absorption(attn_module, output, zero_allocator)

    # ------------------------------------------------------------------ #
    #  PyTorch fallback path                                              #
    # ------------------------------------------------------------------ #
    def _compute_miss_reduce_pytorch(
        self,
        attn_module: nn.Module,
        q_all: torch.Tensor,
        actual_indices: torch.Tensor,
        hit_mask: Optional[torch.Tensor],
        miss_mask: Optional[torch.Tensor],
        predicted_qk: Optional[torch.Tensor],
        predicted_indices: Optional[torch.Tensor],
        sm_scale: float,
        zero_allocator,
        kv_pool: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback for compute_miss_reduce."""
        s_q, h_q, d_qk = q_all.shape
        kv_lora_rank = attn_module.kv_lora_rank
        device = q_all.device

        # ---- Trim + clamp ----
        safe_idx, valid_mask, K = self._resolve_kv_and_trim(actual_indices, kv_pool)

        if hit_mask is not None and hit_mask.shape[-1] > K:
            hit_mask = hit_mask[:, :K]
        if miss_mask is not None and miss_mask.shape[-1] > K:
            miss_mask = miss_mask[:, :K]

        has_miss = miss_mask is not None and miss_mask.any()
        has_hit = (
            hit_mask is not None and hit_mask.any()
            and predicted_qk is not None
            and predicted_indices is not None
        )

        # ---- Build full_qk [s_q, h_q, K] ----
        full_qk = torch.full(
            (s_q, h_q, K), float('-inf'), dtype=torch.float32, device=device,
        )

        # Step 1: Compute QK for miss positions only
        if has_miss or not has_hit:
            if has_miss and has_hit:
                miss_count = miss_mask.sum(dim=-1)
                max_miss = miss_count.max().item()
                if max_miss > 0:
                    sort_key = (~miss_mask).long()
                    _, sort_idx = sort_key.sort(dim=-1, stable=True)
                    compact_pos = sort_idx[:, :max_miss]
                    compact_kv_idx = torch.gather(safe_idx, 1, compact_pos)
                    compact_valid = (
                        torch.arange(max_miss, device=device).unsqueeze(0)
                        < miss_count.unsqueeze(1)
                    )
                    miss_k = kv_pool[compact_kv_idx, 0, :]
                    _debug_sync("miss_reduce:gather_miss_k")
                    miss_qk = torch.bmm(
                        q_all.float().contiguous(),
                        miss_k.float().transpose(1, 2).contiguous(),
                    ) * sm_scale
                    _debug_sync("miss_reduce:bmm_miss_qk")
                    miss_qk.masked_fill_(
                        ~compact_valid.unsqueeze(1).expand(-1, h_q, -1),
                        float('-inf'),
                    )
                    scatter_idx = compact_pos.unsqueeze(1).expand(-1, h_q, -1)
                    full_qk.scatter_(2, scatter_idx, miss_qk)
            else:
                all_k = kv_pool[safe_idx, 0, :]
                qk_all = torch.bmm(
                    q_all.float().contiguous(),
                    all_k.float().transpose(1, 2).contiguous(),
                ) * sm_scale
                qk_all.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))
                full_qk = qk_all

        # Step 2: Remap predicted QK for hit positions
        if has_hit:
            remap_pos = self._remap_predicted_to_actual(
                predicted_indices, actual_indices[:, :K], h_q,
            )
            remapped_qk = torch.gather(predicted_qk, 2, remap_pos)
            hit_exp = hit_mask.unsqueeze(1).expand(-1, h_q, -1)
            full_qk[hit_exp] = remapped_qk[hit_exp]

        full_qk.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))

        # Step 3: Softmax
        attn_weights = F.softmax(full_qk, dim=-1)

        # Step 4: Gather V, PV multiply
        v = kv_pool[safe_idx, 0, :kv_lora_rank]
        _debug_sync("miss_reduce:gather_v")
        v = v * valid_mask.unsqueeze(-1).to(v.dtype)

        attn_output = torch.bmm(
            attn_weights.to(v.dtype), v.contiguous(),
        )
        _debug_sync("miss_reduce:bmm_pv")

        result = self._apply_wvc_absorption(attn_module, attn_output, zero_allocator)
        _debug_sync("miss_reduce:wvc_absorption")
        return result

    @staticmethod
    def _remap_predicted_to_actual(
        predicted_indices: torch.Tensor,   # [s_q, pred_topk]
        actual_idx: torch.Tensor,          # [s_q, K]
        h_q: int,
    ) -> torch.Tensor:
        """For each actual position, find its position in predicted_indices.

        Returns index tensor [s_q, h_q, K] suitable for torch.gather(predicted_qk, 2, ...).
        """
        safe_pred = predicted_indices.clone()
        safe_pred[safe_pred < 0] = -2                        # sentinel for invalid
        sorted_pred, sort_perm = safe_pred.sort(dim=-1)

        search_vals = actual_idx.clamp(min=0)
        pos_in_sorted = torch.searchsorted(sorted_pred, search_vals)
        pos_in_sorted = pos_in_sorted.clamp(max=sorted_pred.shape[-1] - 1)
        original_pos = torch.gather(sort_perm, 1, pos_in_sorted)  # [s_q, K]

        return original_pos.unsqueeze(1).expand(-1, h_q, -1)      # [s_q, h_q, K]


    def apply_o_proj(self, attn_output: torch.Tensor, o_proj_layer: nn.Module) -> torch.Tensor:
        """Apply O projection to per-head attention output.
        
        Args:
            attn_output: [num_tokens, num_heads, v_head_dim]
            o_proj_layer: output projection layer
        
        Returns:
            output: [num_tokens, hidden_size]
        """
        if attn_output.dim() != 3:
            # If it's already 2D, just apply o_proj (though typically USC expects 3D here)
            if attn_output.dim() == 2:
                output, _ = o_proj_layer(attn_output)
                return output
            raise RuntimeError(
                f"USC Attention: attn_output must be 3D, got {attn_output.shape}"
            )
        attn_output_2d = attn_output.flatten(1, 2)
        output, _ = o_proj_layer(attn_output_2d)
        return output