from typing import Tuple, Optional, Callable, List, TYPE_CHECKING
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# try:
#     from sgl_kernel.flash_mla import flash_mla_sparse_merge_fwd as _sparse_merge_fwd
#     _HAS_SPARSE_MERGE_KERNEL = True
# except ImportError:
#     from flash_mla import flash_mla_sparse_merge_fwd as _sparse_merge_fwd
#     _HAS_SPARSE_MERGE_KERNEL = True
from flash_mla import flash_mla_sparse_merge_fwd as _sparse_merge_fwd
_HAS_SPARSE_MERGE_KERNEL = True

import deep_gemm

import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sgl_kernel.moe import fused_verify_remap as fused_verify_remap_cuda

@triton.jit
def _fused_verify_remap_kernel(
    actual_ptr,       # [N, topk] int64
    predicted_ptr,    # [N, pred_topk] int64
    hit_ptr,          # [N, topk] int8  output
    remap_ptr,        # [N, topk] int32 output (position in predicted)
    topk, pred_topk,
    stride_actual_row, stride_pred_row,
    BLOCK_A: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Fused verify (hit/miss) + remap (find position in predicted).

    Grid: (N, cdiv(topk, BLOCK_A))
    """
    pid_n = tl.program_id(0)   # token row
    pid_a = tl.program_id(1)   # actual-index chunk

    a_offs = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    a_mask = a_offs < topk

    actual = tl.load(
        actual_ptr + pid_n * stride_actual_row + a_offs,
        mask=a_mask, other=-1,
    ).to(tl.int32)
    valid = actual >= 0

    found_pos = tl.zeros([BLOCK_A], dtype=tl.int32) - 1

    for p_start in range(0, pred_topk, BLOCK_P):
        p_offs = p_start + tl.arange(0, BLOCK_P)
        p_mask = p_offs < pred_topk

        pred = tl.load(
            predicted_ptr + pid_n * stride_pred_row + p_offs,
            mask=p_mask, other=-2,
        ).to(tl.int32)

        eq = actual[:, None] == pred[None, :]

        not_yet = (found_pos < 0)                       # [BLOCK_A]
        eq_new = eq & not_yet[:, None] & p_mask[None, :]  # [BLOCK_A, BLOCK_P]

        pos_vals = p_offs[None, :].to(tl.int32)           # [1, BLOCK_P]
        masked_pos = tl.where(eq_new, pos_vals, pred_topk)
        min_pos = tl.min(masked_pos, axis=1)               # [BLOCK_A]

        newly_found = min_pos < pred_topk
        found_pos = tl.where(newly_found, min_pos, found_pos)

    hit  = (found_pos >= 0) & valid
    miss = (found_pos <  0) & valid
    remap_val = tl.where(found_pos >= 0, found_pos, 0)

    tl.store(hit_ptr   + pid_n * topk + a_offs, hit.to(tl.int8),  mask=a_mask)
    tl.store(remap_ptr + pid_n * topk + a_offs, remap_val,        mask=a_mask)


@triton.jit
def _fused_partition_and_prepare_kernel(
    # Inputs
    actual_ptr,           # [s_q, topk] int64
    hit_mask_ptr,         # [s_q, topk] int8
    remap_ptr,            # [s_q, topk] int32
    predicted_qk_ptr,     # [s_q, h_q, pred_topk] 
    q_all_ptr,            # [s_q, h_q, d_qk] bf16
    # Outputs — topk buffers
    idx_out_ptr,          # [s_q, 1, topk] int32
    hit_out_ptr,          # [s_q, 1, topk_padded] int32
    qk_out_ptr,           # [s_q, h_q, topk] 
    # Output — q buffer
    q_out_ptr,            # [s_q, h_q, d_qk] bf16
    # Dimensions
    topk, pool_sz, d_qk,
    # Input strides
    stride_actual_row,
    stride_hit_row,
    stride_remap_row,
    stride_pqk_s, stride_pqk_h,
    stride_qall_s, stride_qall_h,
    # Output strides
    stride_idx_row, stride_hout_row,
    stride_qk_s, stride_qk_h,
    stride_qout_s, stride_qout_h,
    # Constexpr
    HAS_HIT: tl.constexpr,
    H_Q: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,   # >= topk, power-of-2
    BLOCK_D: tl.constexpr,      # for q copy path
):
    """Fused binary-partition + sorted write + q_buf copy.

    Grid: ``(s_q,  1 + N_D_BLOCKS)``

    * Program  ``pid_b == 0``  : partition + scatter-write idx/hit/qk
    * Programs ``pid_b >= 1``  : copy q_all → q_buf (H_Q heads only)
    """
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    if pid_b == 0:
        k_offs    = tl.arange(0, BLOCK_TOPK)
        in_data   = k_offs < topk

        # ---- Load inputs ----
        actual = tl.load(
            actual_ptr + pid_s * stride_actual_row + k_offs,
            mask=in_data, other=-1,
        ).to(tl.int32)

        if HAS_HIT:
            hit = tl.load(
                hit_mask_ptr + pid_s * stride_hit_row + k_offs,
                mask=in_data, other=0,
            ).to(tl.int32)
            is_miss = (hit == 0) & in_data
            is_hit  = (hit != 0) & in_data
        else:
            hit = tl.zeros([BLOCK_TOPK], dtype=tl.int32)
            is_miss = in_data
            is_hit  = tl.zeros([BLOCK_TOPK], dtype=tl.int1)

        # ---- Stable binary partition: miss → front, hit → back ----
        is_miss_i32 = is_miss.to(tl.int32)
        is_hit_i32  = is_hit.to(tl.int32)

        miss_prefix = tl.cumsum(is_miss_i32)     # inclusive
        hit_prefix  = tl.cumsum(is_hit_i32)      # inclusive
        n_miss      = tl.sum(is_miss_i32)

        dest = tl.where(
            is_miss,
            miss_prefix - 1,
            n_miss + hit_prefix - 1,
        )

        # ---- Scatter-write idx_buf ----
        safe_idx = tl.where(actual < pool_sz, actual, pool_sz - 1)
        tl.store(
            idx_out_ptr + pid_s * stride_idx_row + dest,
            safe_idx, mask=in_data,
        )

        # ---- Scatter-write hit_buf ----
        tl.store(
            hit_out_ptr + pid_s * stride_hout_row + dest,
            hit.to(tl.int32), mask=in_data,
        )

        if HAS_HIT:
            remap = tl.load(
                remap_ptr + pid_s * stride_remap_row + k_offs,
                mask=is_hit, other=0,
            ).to(tl.int32)

            qk_base  = pid_s * stride_qk_s
            pqk_base = pid_s * stride_pqk_s

            for h in range(H_Q):
                qk_val = tl.load(
                    predicted_qk_ptr + pqk_base + h * stride_pqk_h + remap,
                    mask=is_hit, other=0.0,
                )
                tl.store(
                    qk_out_ptr + qk_base + h * stride_qk_h + dest,
                    qk_val, mask=is_hit,
                )
    else:
        d_idx  = pid_b - 1
        d_offs = d_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_qk

        qout_base = pid_s * stride_qout_s
        qall_base = pid_s * stride_qall_s

        for h in range(H_Q):
            val = tl.load(
                q_all_ptr + qall_base + h * stride_qall_h + d_offs,
                mask=d_mask, other=0.0,
            )
            tl.store(
                q_out_ptr + qout_base + h * stride_qout_h + d_offs,
                val, mask=d_mask,
            )


def _fused_partition_and_prepare(
    actual_indices: torch.Tensor,              # [s_q, topk]
    hit_mask: Optional[torch.Tensor],          # [s_q, topk] int8
    remap_pos: Optional[torch.Tensor],         # [s_q, topk] int32
    predicted_qk: Optional[torch.Tensor],      # [s_q, h_q, pred_topk] f32
    q_all: torch.Tensor,                       # [s_q, h_q, d_qk] bf16
    idx_buf: torch.Tensor,                     # [s_q, 1, topk] i32
    hit_buf: torch.Tensor,                     # [s_q, 1, topk] i32
    qk_buf: torch.Tensor,                      # [s_q, h_q, topk] f32
    q_buf: torch.Tensor,                       # [s_q, h_q, d_qk] bf16
    topk: int, 
    h_q: int, 
    pool_sz: int, d_qk: int,
    has_hit: bool,
):
    """Single kernel: binary partition (miss-first) + write + q_buf copy.
    """
    s_q = actual_indices.shape[0]
    BLOCK_TOPK = triton.next_power_of_2(topk)
    num_warps = max(4, min(BLOCK_TOPK // 64, 16))
    grid = (s_q, 1)
    BLOCK_D = 128 

    if has_hit:
        hit_ptr       = hit_mask
        remap_ptr     = remap_pos
        pqk_ptr       = predicted_qk
        stride_hit    = hit_mask.stride(0)
        stride_remap  = remap_pos.stride(0)
        stride_pqk_s  = predicted_qk.stride(0)
        stride_pqk_h  = predicted_qk.stride(1)
    else:
        hit_ptr       = actual_indices
        remap_ptr     = actual_indices
        pqk_ptr       = qk_buf
        stride_hit    = topk
        stride_remap  = topk
        stride_pqk_s  = 0
        stride_pqk_h  = 0

    _fused_partition_and_prepare_kernel[grid](
        actual_indices, hit_ptr, remap_ptr, pqk_ptr, q_all,
        idx_buf, hit_buf, qk_buf, q_buf,
        topk, pool_sz, d_qk,
        actual_indices.stride(0),
        stride_hit, stride_remap,
        stride_pqk_s, stride_pqk_h,
        q_all.stride(0), q_all.stride(1),
        idx_buf.stride(0), hit_buf.stride(0),
        qk_buf.stride(0), qk_buf.stride(1),
        q_buf.stride(0), q_buf.stride(1),
        HAS_HIT=has_hit,
        H_Q=h_q,
        BLOCK_TOPK=BLOCK_TOPK,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

class StreamTensorWrapper:
    def __init__(self, tensor: torch.Tensor, event: Optional[torch.cuda.Event] = None):
        self.tensor = tensor
        self.event = event

    def get_tensor(self):
        if self.event is not None:
            self.event.wait()
        return self.tensor

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
        
        # Async stream for hit attention (set to None to disable dual-stream)
        self.alt_stream = None # torch.cuda.Stream()
        self.alt_event0 = torch.cuda.Event()
        self.alt_event1 = torch.cuda.Event()
        
        # CP info
        self.cp_size = 1
        self.cp_rank = 0
        self.use_cp = False
        
        # Index estimator
        self.topk = topk
        self.estimator = None
        self._estimate_event = None
        self._estimated_index = None

        # Fused verify+remap output: consumed by _compute_miss_reduce_kernel
        self._last_remap_pos: Optional[torch.Tensor] = None

    _shared_qk_buf: Optional[torch.Tensor] = None
    _shared_q_all_buf: Optional[torch.Tensor] = None
    @classmethod
    def ensure_q_all_buf(
        cls,
        s_q: int,
        h_q: int,
        d_qk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a persistent buffer with shape >= [s_q, h_q, d_qk].

        Used to replace ``torch.cat([q_nope_out, q_pe], dim=-1)`` with
        two slice-copies into a pre-allocated buffer, avoiding a fresh
        allocation + memcpy every forward pass.
        """
        buf = cls._shared_q_all_buf
        if (
            buf is not None
            and buf.shape[0] >= s_q
            and buf.shape[1] >= h_q
            and buf.shape[2] >= d_qk
            and buf.dtype == dtype
        ):
            return buf

        alloc_sq = ((s_q + 255) // 256) * 256
        cls._shared_q_all_buf = torch.empty(
            alloc_sq, h_q, d_qk, dtype=dtype, device=device,
        )
        logger.info(
            f"[USC] allocated shared q_all buf: {list(cls._shared_q_all_buf.shape)} "
            f"dtype={dtype} "
            f"({cls._shared_q_all_buf.nelement() * cls._shared_q_all_buf.element_size() / 1024 / 1024:.0f} MB)"
        )
        return cls._shared_q_all_buf

    @classmethod
    def ensure_qk_buf(
        cls,
        s_q: int,
        h_q: int,
        topk: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return a persistent bf16 buffer with shape >= [s_q, h_q, topk].
        """
        buf = cls._shared_qk_buf
        if (
            buf is not None
            and buf.shape[0] >= s_q
            and buf.shape[1] >= h_q
            and buf.shape[2] >= topk
        ):  
            return buf

        # Round s_q up to next 256 to reduce future re-allocations
        alloc_sq = ((s_q + 255) // 256) * 256
        cls._shared_qk_buf = torch.empty(
            alloc_sq, h_q, topk, dtype=torch.bfloat16, device=device,
        )
        logger.info(
            f"[USC] allocated shared QK buf: {list(cls._shared_qk_buf.shape)} "
            f"({cls._shared_qk_buf.nelement() * 2 / 1024 / 1024:.0f} MB)"
        )
        return cls._shared_qk_buf

    # Pre-allocated buffers for miss-reduce kernel to avoid dynamic allocation
    _shared_mr_idx_buf: Optional[torch.Tensor] = None   # [alloc_sq, 1, topk_padded] int32
    _shared_mr_hit_buf: Optional[torch.Tensor] = None   # [alloc_sq, 1, topk_padded] int32
    _shared_mr_qk_buf: Optional[torch.Tensor] = None     # [alloc_sq, h_padded, topk_padded] f32
    _shared_mr_q_buf: Optional[torch.Tensor] = None     # [alloc_sq, h_padded, d_qk] bf16

    @classmethod
    def _ensure_mr_bufs(
        cls,
        s_q: int, h_q: int, topk: int, device: torch.device,
    ):
        """Lazy-allocate miss-reduce kernel buffers. Grows but never shrinks.

        Args:
            skip_q_buf: If True, skip allocating q_buf (used when h_q is already aligned)
        """
        alloc_sq = ((s_q + 255) // 256) * 256

        # idx / hit buffers — small (a few MB)
        if (
            cls._shared_mr_idx_buf is None
            or cls._shared_mr_idx_buf.shape[0] < s_q
            or cls._shared_mr_idx_buf.shape[2] < topk
        ):
            cls._shared_mr_idx_buf = torch.empty(
                alloc_sq, 1, topk, dtype=torch.int32, device=device,
            )
            cls._shared_mr_hit_buf = torch.empty(
                alloc_sq, 1, topk, dtype=torch.int32, device=device,
            )
            logger.info(
                f"[USC] allocated shared MR idx/hit buf: "
                f"sq={alloc_sq} topk={topk}"
            )

        if (
            cls._shared_mr_qk_buf is None
            or cls._shared_mr_qk_buf.shape[0] < s_q
            or cls._shared_mr_qk_buf.shape[1] < h_q
            or cls._shared_mr_qk_buf.shape[2] < topk
        ):
            cls._shared_mr_qk_buf = torch.zeros(
                alloc_sq, h_q, topk, dtype=torch.float32, device=device,
            )
            logger.info(
                f"[USC] allocated shared MR QK buf (zero-init): "
                f"{list(cls._shared_mr_qk_buf.shape)} "
                f"({cls._shared_mr_qk_buf.nelement() * 4 / 1024 / 1024:.0f} MB)"
            )

    def _async_execute(self, fn: Callable):
        if self.alt_stream is None:
            # Synchronous path — no dual stream.
            return StreamTensorWrapper(fn(), None)
        self.alt_event0.record()
        with torch.cuda.stream(self.alt_stream):
            self.alt_event0.wait()
            result = fn()
            self.alt_event1.record()
        return StreamTensorWrapper(
            result,
            self.alt_event1
        )

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
        # n_loc = cache_loc.numel()
        # n_kv = k_nope.shape[0]
        # if n_loc > n_kv:
        #     logger.warning(
        #         f"[USC] set_mla_kv_buffer: cache_loc({n_loc}) > k_nope rows({n_kv}), "
        #         f"truncating cache_loc to avoid OOB in Triton kernel"
        #     )
        #     cache_loc = cache_loc[:n_kv]

        # # Defensive: clamp cache_loc to pool buffer bounds to prevent OOB write
        # kv_pool = forward_batch.token_to_kv_pool.get_key_buffer(
        #     attn_module.attn_mqa.layer_id
        # )
        # pool_sz = kv_pool.shape[0]
        # cache_loc = cache_loc.clamp(max=pool_sz - 1)

        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            attn_module.attn_mqa,
            cache_loc,
            k_nope,
            k_pe,
        )
        # _debug_sync("set_mla_kv_buffer")

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

        # Concat q_nope_out and q_pe into pre-allocated buffer (1 kernel, 0 alloc).
        s_q = q_nope_out.shape[0]
        h_q = q_nope_out.shape[1]
        d_nope = q_nope_out.shape[2]
        d_pe = q_pe.shape[2]
        d_qk = d_nope + d_pe
        q_all_buf = self.ensure_q_all_buf(
            s_q, h_q, d_qk, q_nope_out.dtype, q_nope_out.device,
        )
        q_all = q_all_buf[:s_q, :h_q, :]
        torch.cat([q_nope_out, q_pe], dim=-1, out=q_all)
        if llama_4_scaling is not None:
            q_all.mul_(llama_4_scaling)

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
        #     # ===== QK Overlap 模式 =====
        #     # 检查是否启用 QK overlap：需要 CUDA 环境且 Indexer 有 forward_cuda_overlap
        #     enable_qk_overlap = (
        #         torch.cuda.is_available()
        #         and hasattr(attn_module.indexer, 'forward_cuda_overlap')
        #         and hasattr(attn_module.indexer, 'qk_compute_stream')
        #         and attn_module.indexer.qk_compute_stream is not None
        #     )
            
        #     if enable_qk_overlap:
        #         # 准备 QK overlap 参数
        #         # 1. 获取 q_all 作为 q_for_qk（从 state 中恢复）
        #         q_all = state.get("_usc_q_all")
                
        #         # 2. 获取用于 QK 计算的 indices（可以是现有的 topk，或者是历史 indices）
        #         predicted_indices = state.get("attn_estimated_sparse_index")

        #         if q_all is not None and predicted_indices is not None:
        #             # 3. 预分配 QK 输出 buffer
        #             s_q, h_q, d_qk = q_all.shape
        #             topk = predicted_indices.shape[-1]
        #             qk_output_buf = self.ensure_qk_buf(s_q, h_q, topk, q_all.device)
                    
        #             # 4. 调用 forward_cuda_overlap 进行并行计算
        #             # 注意：这里假设 attn_module.indexer 有 forward_cuda_overlap 方法
        #             # 该方法会在独立 stream 上计算 QK，与 set_index_k_scale_buffer 并行
        #             topk_indices = attn_module.indexer.forward_cuda_overlap(
        #                 x=hidden_states_for_indexer,
        #                 q_lora=q_lora,
        #                 positions=state.positions,
        #                 forward_batch=forward_batch,
        #                 layer_id=attn_module.layer_id,
        #                 attn_cache=self,
        #                 q_for_qk=q_all,
        #                 indices_for_qk=predicted_indices,
        #                 qk_output_buf=qk_output_buf,
        #                 return_indices=True,
        #             )
                    
        #             # QK 计算结果已经在 qk_output_buf 中，可以后续使用
        #             # 先删除已存在的（如果存在），避免断言错误
        #             if hasattr(state, '_data') and "attn_predicted_qk" in state._data:
        #                 state._data.pop("attn_predicted_qk", None)
        #             state.attn_predicted_qk = qk_output_buf[:s_q, :h_q, :topk]
        #         else:
        #             # 缺少参数，回退到普通 forward
        #             # 设置 QK 为 None，让后续代码可以 pop
        #             # 先删除已存在的（如果存在）
        #             if hasattr(state, '_data') and "attn_predicted_qk" in state._data:
        #                 state._data.pop("attn_predicted_qk", None)
        #             state.attn_predicted_qk = None
        #             enable_qk_overlap = False
            
        #     if not enable_qk_overlap:
        #         # 标准模式：调用普通 forward（通过 MultiPlatformOp 调度）
        #         topk_indices = attn_module.indexer(
        #             x=hidden_states_for_indexer,
        #             q_lora=q_lora,
        #             positions=state.positions,
        #             forward_batch=forward_batch,
        #             layer_id=attn_module.layer_id,
        #         )
        #         # 在非 overlap 模式下，QK 分数未计算，设置为 None
        #         # 先删除已存在的（如果存在），再设置
        #         if hasattr(state, '_data') and "attn_predicted_qk" in state._data:
        #             state._data.pop("attn_predicted_qk", None)
        #         state.attn_predicted_qk = None
        # else:
        #     topk_indices = None
        #     # 先删除已存在的（如果存在），再设置
        #     if hasattr(state, '_data') and "attn_predicted_qk" in state._data:
        #         state._data.pop("attn_predicted_qk", None)
        #     state.attn_predicted_qk = None

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


    def compute_qk_scores(
        self,
        q_all: torch.Tensor,      # [s_q, h_q, d_qk]
        indices: torch.Tensor,     # [s_q, topk]  (PAGED pool slot indices, padded with -1)
        kv_pool: torch.Tensor,    # [pool_size, 1, d_qk]
        qk_buf: torch.Tensor,     # [>=s_q, >=h_q, >=topk] bf16 
    ) -> torch.Tensor:
        def _fn():
            s_q, h_q, d_qk = q_all.shape
            topk = indices.shape[-1]
            out_view = qk_buf[:s_q, :h_q, :topk]

            pool_sz = kv_pool.shape[0]
            safe_idx = indices.clamp(min=0, max=pool_sz - 1)
            selected_k = kv_pool[safe_idx, 0, :]  # [s_q, topk, d_qk]

            out_view = torch.bmm(
                q_all.contiguous(),
                selected_k.transpose(1, 2).contiguous(),
            )
            return out_view

            # # Try DeepGEMM BF16 path for Q@K^T
            #     # Clamp indices to valid range
            #     pool_sz = kv_pool.shape[0]
            #     safe_idx = indices.clamp(min=0, max=pool_sz - 1)
                
            #     # Gather K: [s_q, topk, d_qk] - this is the expensive indirect access
            #     # DeepGEMM can't do gather, so we do it first
            #     selected_k = kv_pool[safe_idx, 0, :]  # [s_q, topk, d_qk]
                
            #     # Use DeepGEMM bf16_gemm_nt for Q@K^T
            #     # Input: q_all [s_q, h_q, d_qk], selected_k [s_q, topk, d_qk]
            #     # Output: qk [s_q, h_q, topk]
                
            #     # Prepare contiguous tensors
            #     # q_contig = q_all.contiguous()  # [s_q, h_q, d_qk]
            #     # k_contig = selected_k.contiguous()  # [s_q, topk, d_qk]
                
            #     # if s_q == 1:
            #     #     # Single sample case: use 2D gemm directly
            #     #     # q: [h_q, d_qk], k: [topk, d_qk], out: [h_q, topk]
            #     #     q_2d = q_contig[0]  # [h_q, d_qk]
            #     #     k_2d = k_contig[0]  # [topk, d_qk]
            #     #     # bf16_gemm_nt expects: lhs [M, K], rhs [N, K], out [M, N]
            #     #     # Here M=h_q, K=d_qk, N=topk
            #     #     # out_view[0] is already [h_q, topk] bf16, use it directly
            #     #     deep_gemm.bf16_gemm_nt(q_2d, k_2d, out_view[0])
            #     # else:
            #     #     # Batched case: process each sample, write directly to out_view
            #     #     for i in range(s_q):
            #     #         # q_i: [h_q, d_qk], k_i: [topk, d_qk]
            #     #         # out_view[i] is [h_q, topk] bf16, use directly as output buffer
            #     #         deep_gemm.bf16_gemm_nt(q_contig[i], k_contig[i], out_view[i])
            #     for i in range(s_q):
            #         deep_gemm.bf16_gemm_nt(q_all[i], selected_k[i], out_view[i]) # q_i: [h_q, d_qk], k_i: [topk, d_qk]
            #         # deep_gemm.bf16_gemm_nt(q_contig[i], k_contig[i], out_view[i]) # q_i: [h_q, d_qk], k_i: [topk, d_qk]
            #     return out_view
        return self._async_execute(_fn)


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

    def compute_miss_reduce(
        self,
        attn_module: nn.Module,
        q_all: torch.Tensor,                        # [s_q, h_q, d_qk]
        actual_indices: torch.Tensor,                # [s_q, topk]  (PAGED, padded with -1)
        predicted_qk: Optional[torch.Tensor],        # [s_q, h_q, pred_topk]
        predicted_indices: Optional[torch.Tensor],   # [s_q, pred_topk]
        sm_scale: float,
        zero_allocator,
        kv_pool: torch.Tensor,                      # [pool_size, 1, d_qk]
    ) -> torch.Tensor:
        return self._compute_miss_reduce_kernel(
            attn_module, q_all, actual_indices,
            predicted_qk, predicted_indices, sm_scale, zero_allocator, kv_pool,
        )

    def _compute_miss_reduce_kernel(
        self,
        attn_module: nn.Module,
        q_all: torch.Tensor,
        actual_indices: torch.Tensor,
        predicted_qk: Optional[torch.Tensor],
        predicted_indices: Optional[torch.Tensor],
        sm_scale: float,
        zero_allocator,
        kv_pool: torch.Tensor,
    ) -> torch.Tensor:
        s_q, h_q, d_qk = q_all.shape
        kv_lora_rank = attn_module.kv_lora_rank
        device = q_all.device
        topk = actual_indices.shape[-1]
        pool_sz = kv_pool.shape[0]

        # ---- Pre-allocate / reuse kernel buffers (zero-initialized) ----
        # Only allocate q_buf if we need padding, otherwise use q_all directly
        self._ensure_mr_bufs(
            s_q=s_q, h_q=h_q, topk=topk, device=device,
        )
        idx_buf = self._shared_mr_idx_buf[:s_q, :, :topk]
        hit_buf = self._shared_mr_hit_buf[:s_q, :, :topk]
        qk_buf = self._shared_mr_qk_buf[:s_q, :h_q, :topk]

        q_buf = q_all

        # ---- has_hit: structural check ----
        has_hit = (
            predicted_qk is not None
            and predicted_indices is not None
        )

        # ---- Compute hit_mask and remap_pos on-demand (eliminate intermediate buffers) ----
        if has_hit:
            # hit_raw = torch.empty(s_q, topk, dtype=torch.int8, device=device)
            # remap_pos = torch.empty(s_q, topk, dtype=torch.int32, device=device)

            # _fused_verify_remap_kernel[(s_q, triton.cdiv(topk, 64))]( # BLOCK_A=64, BLOCK_P=64
            #     actual_indices,
            #     predicted_indices,
            #     hit_raw,
            #     remap_pos,
            #     topk, predicted_indices.shape[-1],
            #     actual_indices.stride(0), predicted_indices.stride(0),
            #     BLOCK_A=64, BLOCK_P=64,
            # )
            hit_mask_for_partition, remap_pos = fused_verify_remap_cuda(actual_indices.to(torch.int64), predicted_indices.to(torch.int64))
        else:
            hit_mask_for_partition = None
            remap_pos = None

        # ---- Fused partition + write + (optional) q_buf copy ----
        _fused_partition_and_prepare(
            actual_indices=actual_indices,
            hit_mask=hit_mask_for_partition,
            remap_pos=remap_pos,
            predicted_qk=predicted_qk,
            q_all=q_all,
            idx_buf=idx_buf,
            hit_buf=hit_buf,
            qk_buf=qk_buf,
            q_buf=q_all,
            topk=topk,
            h_q=h_q,
            pool_sz=pool_sz,
            d_qk=d_qk,
            has_hit=has_hit,
        )

        # ---- Launch sparse merge kernel ----
        output, _max_logits, _lse = _sparse_merge_fwd(
            q=q_buf,
            kv=kv_pool.contiguous(),
            actual_indices=idx_buf,
            predicted_qk=qk_buf,
            hit_mask=hit_buf,
            sm_scale=sm_scale,
            d_v=kv_lora_rank,
        )

        return self._apply_wvc_absorption(attn_module, output, zero_allocator)

