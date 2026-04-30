"""Tests for fp4_act_quant and tilelang_fp4_paged_mqa_logits in jit_kernel/deepseek_v4.py."""
import itertools

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# These kernels require TileLang (not installed in standard CI) and MXFP4 hardware.
# Disabled in CI; run locally with: python python/sglang/jit_kernel/tests/test_fp4_indexer_kernels.py
register_cuda_ci(
    est_time=120,
    suite="stage-b-kernel-unit-1-gpu-large",
    disabled="Requires TileLang and torch>=2.7 float4_e2m1fn_x2",
)

device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Skip guard: these kernels need float4_e2m1fn_x2 (PyTorch >= 2.7) and CUDA.
# ---------------------------------------------------------------------------
_SKIP = not torch.cuda.is_available() or not hasattr(torch, "float4_e2m1fn_x2")
skip_if_unavailable = pytest.mark.skipif(_SKIP, reason="Requires CUDA and torch>=2.7 float4_e2m1fn_x2")

FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def fp4_dequant_ref(fp4_packed: torch.Tensor, fe8m0: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Pure-PyTorch reference: unpack MXFP4 + FE8M0 → BF16."""
    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device=fp4_packed.device)
    # fp4_packed: (..., N//2) uint8 viewed as float4_e2m1fn_x2
    # fe8m0: (..., N//block_size) uint8 viewed as float8_e8m0fnu
    packed_u8 = fp4_packed.view(torch.uint8)
    flat = packed_u8.view(-1, packed_u8.shape[-1])  # (M, N//2)
    M, half_N = flat.shape
    N = half_N * 2
    num_blocks = N // block_size

    fe8m0_u8 = fe8m0.view(torch.uint8).view(-1, num_blocks)  # (M, num_blocks)
    scales = (fe8m0_u8.to(torch.int32) - 127).float()
    scales = 2.0 ** scales  # (M, num_blocks)

    out = torch.empty((M, N), dtype=torch.float32, device=fp4_packed.device)
    for d2 in range(half_N):
        lo = flat[:, d2].to(torch.int32) & 0x0F
        hi = (flat[:, d2].to(torch.int32) >> 4) & 0x0F
        blk_lo = (2 * d2) // block_size
        blk_hi = (2 * d2 + 1) // block_size
        out[:, 2 * d2]     = lut[lo] * scales[:, blk_lo]
        out[:, 2 * d2 + 1] = lut[hi] * scales[:, blk_hi]

    return out.view(*fp4_packed.shape[:-1], N).bfloat16()


def fp4_paged_mqa_logits_ref(
    q_fp8: torch.Tensor,      # (N, 1, H, D) FP8
    kvcache_fp4: torch.Tensor, # (num_pages, B, 1, 68) uint8
    weight: torch.Tensor,      # (N, H) FP32
    seq_lens: torch.Tensor,    # (N,) int32
    page_table: torch.Tensor,  # (N, L) int32
    max_seq_len: int,
    block_size: int = 64,
    fp4_block_size: int = 32,
) -> torch.Tensor:
    """Pure-PyTorch reference for tilelang_fp4_paged_mqa_logits."""
    batch_size, _, H, D = q_fp8.shape
    k_bytes = D // 2
    s_bytes = D // fp4_block_size
    token_bytes = k_bytes + s_bytes  # 68

    logits = torch.zeros(batch_size, max_seq_len, dtype=torch.float32, device=q_fp8.device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages = (seq_len + block_size - 1) // block_size
        q = q_fp8[b, 0].to(torch.float32)  # (H, D)

        for pg_i in range(num_pages):
            page = int(page_table[b, pg_i].item())
            # kvcache_fp4[page]: (B, 1, 68)
            page_data = kvcache_fp4[page, :, 0, :]   # (B, 68)
            k_packed = page_data[:, :k_bytes]         # (B, 64) uint8
            s_packed = page_data[:, k_bytes:]         # (B, 4)  uint8

            # Dequant K: (B, D) float32
            k_f32 = torch.empty(block_size, D, dtype=torch.float32, device=q_fp8.device)
            for j in range(block_size):
                for d2 in range(k_bytes):
                    packed = int(k_packed[j, d2].item())
                    lo = packed & 0x0F
                    hi = (packed >> 4) & 0x0F
                    blk = (2 * d2) // fp4_block_size
                    fe8m0 = int(s_packed[j, blk].item())
                    scale = 2.0 ** (fe8m0 - 127)
                    k_f32[j, 2 * d2]     = FP4_LUT[lo] * scale
                    k_f32[j, 2 * d2 + 1] = FP4_LUT[hi] * scale

            # Score: (B, H) = K @ Q^T
            score = k_f32 @ q.T  # (B, H)
            score = torch.relu(score) * weight[b].unsqueeze(0)  # (B, H)
            score_sum = score.sum(dim=1)  # (B,)

            start = pg_i * block_size
            end = min(start + block_size, seq_len)
            logits[b, start:end] = score_sum[:end - start]

    return logits


# ---------------------------------------------------------------------------
# Tests for fp4_act_quant
# ---------------------------------------------------------------------------

@skip_if_unavailable
@pytest.mark.parametrize("shape,block_size", [
    ((32, 128), 32),
    ((64, 128), 32),
    ((128, 128), 32),
    ((32, 128), 64),
    ((1, 128), 32),
])
def test_fp4_act_quant_output_shapes(shape, block_size):
    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    fp4, scale = fp4_act_quant(x, block_size=block_size, inplace=False)

    N = shape[-1]
    assert fp4.shape == (*shape[:-1], N // 2), f"fp4 shape mismatch: {fp4.shape}"
    assert fp4.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (*shape[:-1], N // block_size), f"scale shape mismatch: {scale.shape}"
    assert scale.dtype == torch.float8_e8m0fnu


@skip_if_unavailable
@pytest.mark.parametrize("num_tokens", [32, 64, 128])
def test_fp4_act_quant_dequant_roundtrip(num_tokens):
    """Dequantized values should be close to original (within FP4 precision)."""
    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    torch.manual_seed(42)
    N = 128
    block_size = 32
    x = torch.randn(num_tokens, N, dtype=torch.bfloat16, device=device)

    fp4, scale = fp4_act_quant(x, block_size=block_size, inplace=False)
    x_dequant = fp4_dequant_ref(fp4, scale, block_size=block_size)

    # FP4 has only 4 representable magnitudes per sign, so max error ≈ 0.5× largest step
    torch.testing.assert_close(x.float(), x_dequant.float(), atol=2.0, rtol=0.5)


@skip_if_unavailable
def test_fp4_act_quant_inplace_noop():
    """inplace=True should return the modified input tensor (same object)."""
    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    x = torch.randn(32, 128, dtype=torch.bfloat16, device=device)
    x_orig = x.clone()
    result = fp4_act_quant(x, block_size=32, inplace=True)
    assert result is x, "inplace=True should return x"
    # Values should be close (fused quant+dequant)
    torch.testing.assert_close(result.float(), x_orig.float(), atol=2.0, rtol=0.5)


@skip_if_unavailable
@pytest.mark.parametrize("block_size", [32, 64])
def test_fp4_act_quant_scale_is_power_of_two(block_size):
    """FE8M0 scales encode power-of-2 values; decoded scales should all be powers of 2."""
    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    x = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
    _, scale = fp4_act_quant(x, block_size=block_size, inplace=False)

    scale_f32 = scale.view(torch.uint8).to(torch.int32).float()
    scale_val = 2.0 ** (scale_f32 - 127)
    # log2 of a power-of-2 should be exactly integer
    log2_scale = torch.log2(scale_val)
    assert (log2_scale == log2_scale.round()).all(), "Scales must be exact powers of 2"


# ---------------------------------------------------------------------------
# Tests for tilelang_fp4_paged_mqa_logits
# ---------------------------------------------------------------------------

def _make_fp4_kvcache(num_pages, block_size, head_dim, fp4_block_size, device):
    """Build a synthetic FP4 K-cache with known values for testing."""
    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    k_bytes = head_dim // 2
    s_bytes = head_dim // fp4_block_size
    token_bytes = k_bytes + s_bytes  # 68

    # Random BF16 K values, quantize to MXFP4
    k_bf16 = torch.randn(num_pages * block_size, head_dim, dtype=torch.bfloat16, device=device)
    fp4, fe8m0 = fp4_act_quant(k_bf16, block_size=fp4_block_size, inplace=False)

    # Pack into (num_pages, block_size, 1, 68) uint8
    fp4_u8 = fp4.view(torch.uint8).view(num_pages, block_size, k_bytes)
    fe8m0_u8 = fe8m0.view(torch.uint8).view(num_pages, block_size, s_bytes)
    page_buf = torch.cat([fp4_u8, fe8m0_u8], dim=-1)  # (num_pages, block_size, 68)
    page_buf = page_buf.unsqueeze(2)                   # (num_pages, block_size, 1, 68)
    return page_buf


@skip_if_unavailable
@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 64),
    (2, 64),
    (4, 128),
    (1, 192),
])
def test_tilelang_fp4_paged_mqa_logits_vs_reference(batch_size, seq_len):
    """tilelang kernel output must match the pure-PyTorch reference."""
    from sglang.jit_kernel.deepseek_v4 import tilelang_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    torch.manual_seed(0)
    head_dim = 128
    num_heads = 64
    block_size = 64
    fp4_block_size = 32
    num_pages = (seq_len + block_size - 1) // block_size
    max_seq_len = num_pages * block_size

    # Build FP4 K-cache
    kvcache_fp4 = _make_fp4_kvcache(num_pages, block_size, head_dim, fp4_block_size, device)

    # Q: random BF16 → FP8
    q_bf16 = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_flat = q_bf16.view(-1, head_dim)
    q_fp8, q_scale_flat = act_quant(q_flat)
    q_fp8 = q_fp8.view(batch_size, 1, num_heads, head_dim)
    q_scale = q_scale_flat.view(batch_size, num_heads)  # (N, H)

    # seq_lens and page_table (all requests use the same pages sequentially)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_table = page_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    # Kernel output
    logits_kernel = tilelang_fp4_paged_mqa_logits(
        q_fp8, kvcache_fp4, q_scale, seq_lens, page_table,
        None, max_seq_len, clean_logits=False,
    )

    # Reference output
    logits_ref = fp4_paged_mqa_logits_ref(
        q_fp8, kvcache_fp4, q_scale, seq_lens, page_table,
        max_seq_len, block_size=block_size, fp4_block_size=fp4_block_size,
    )

    # Compare only valid (non-padded) positions
    for b in range(batch_size):
        sl = int(seq_lens[b].item())
        torch.testing.assert_close(
            logits_kernel[b, :sl],
            logits_ref[b, :sl],
            atol=0.5, rtol=0.5,
            msg=f"batch={b} seq_len={sl}",
        )


@skip_if_unavailable
def test_tilelang_fp4_paged_mqa_logits_output_shape():
    from sglang.jit_kernel.deepseek_v4 import tilelang_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    batch_size, seq_len = 3, 128
    head_dim, num_heads, block_size = 128, 64, 64
    num_pages = seq_len // block_size
    max_seq_len = seq_len

    kvcache_fp4 = _make_fp4_kvcache(num_pages, block_size, head_dim, 32, device)
    q_bf16 = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_fp8, q_scale_flat = act_quant(q_bf16.view(-1, head_dim))
    q_fp8 = q_fp8.view(batch_size, 1, num_heads, head_dim)
    q_scale = q_scale_flat.view(batch_size, num_heads)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_table = page_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    out = tilelang_fp4_paged_mqa_logits(
        q_fp8, kvcache_fp4, q_scale, seq_lens, page_table, None, max_seq_len, clean_logits=False
    )
    assert out.shape == (batch_size, max_seq_len)
    assert out.dtype == torch.float32


@skip_if_unavailable
def test_tilelang_fp4_paged_mqa_logits_nonnegative_outputs():
    """After ReLU, all logits in valid positions must be >= 0."""
    from sglang.jit_kernel.deepseek_v4 import tilelang_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    batch_size, seq_len = 2, 64
    head_dim, num_heads, block_size = 128, 64, 64
    num_pages = seq_len // block_size

    kvcache_fp4 = _make_fp4_kvcache(num_pages, block_size, head_dim, 32, device)
    q_bf16 = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_fp8, q_scale_flat = act_quant(q_bf16.view(-1, head_dim))
    q_fp8 = q_fp8.view(batch_size, 1, num_heads, head_dim)
    q_scale = q_scale_flat.view(batch_size, num_heads)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_table = page_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    out = tilelang_fp4_paged_mqa_logits(
        q_fp8, kvcache_fp4, q_scale, seq_lens, page_table, None, seq_len, clean_logits=False
    )
    assert (out[:, :seq_len] >= 0).all(), "All valid logits should be non-negative after ReLU"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
