"""Benchmark for fp4_act_quant and tilelang_fp4_paged_mqa_logits.

Run locally:
    python python/sglang/jit_kernel/benchmark/bench_fp4_indexer.py

Run like CI:
    cd test && python3 run_suite.py --hw cuda --suite stage-b-kernel-benchmark-1-gpu-large
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-kernel-benchmark-1-gpu-large",
    disabled="Requires TileLang and torch>=2.7 float4_e2m1fn_x2",
)

IS_CI = is_in_ci()

# ---------------------------------------------------------------------------
# Check availability
# ---------------------------------------------------------------------------
_tilelang_available = False
try:
    import tilelang  # noqa: F401

    _tilelang_available = True
except ImportError:
    pass

_fp4_available = hasattr(torch, "float4_e2m1fn_x2") and torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Benchmark 1: fp4_act_quant throughput vs plain BF16 copy
# ---------------------------------------------------------------------------

if IS_CI:
    QUANT_SIZES = [(64, 128)]
else:
    QUANT_SIZES = list(itertools.product([32, 64, 128, 256], [128]))

QUANT_SIZE_LABELS = [f"{m}x{n}" for m, n in QUANT_SIZES]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=QUANT_SIZE_LABELS,
        line_arg="provider",
        line_vals=["fp4_quant", "bf16_copy"],
        line_names=["FP4 Act Quant (TileLang)", "BF16 Clone (baseline)"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="fp4-act-quant-performance",
        args={},
    )
)
def bench_fp4_act_quant(shape: str, provider: str):
    if not _fp4_available or not _tilelang_available:
        return float("nan"), float("nan"), float("nan")

    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant

    m, n = [int(x) for x in shape.split("x")]
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "fp4_quant":
        fn = lambda: fp4_act_quant(x, block_size=32, inplace=False)
    else:
        fn = lambda: x.clone()

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


# ---------------------------------------------------------------------------
# Benchmark 2: tilelang_fp4_paged_mqa_logits throughput
# ---------------------------------------------------------------------------

if IS_CI:
    MQA_CONFIGS = [(2, 64)]  # (batch_size, seq_len)
else:
    MQA_CONFIGS = list(itertools.product([1, 2, 4, 8], [64, 128, 256, 512]))

MQA_CONFIG_LABELS = [f"bs{bs}_sl{sl}" for bs, sl in MQA_CONFIGS]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["config"],
        x_vals=MQA_CONFIG_LABELS,
        line_arg="provider",
        line_vals=["tilelang_fp4"],
        line_names=["TileLang FP4 Paged MQA Logits"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="fp4-paged-mqa-logits-performance",
        args={},
    )
)
def bench_fp4_paged_mqa_logits(config: str, provider: str):
    if not _fp4_available or not _tilelang_available:
        return float("nan"), float("nan"), float("nan")
    if torch.cuda.get_device_capability()[0] < 9:
        return float("nan"), float("nan"), float("nan")

    from sglang.jit_kernel.deepseek_v4 import fp4_act_quant, tilelang_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    parts = config.split("_")
    batch_size = int(parts[0][2:])
    seq_len = int(parts[1][2:])

    head_dim, num_heads, block_size, fp4_block_size = 128, 64, 64, 32
    num_pages = (seq_len + block_size - 1) // block_size
    max_seq_len = num_pages * block_size
    k_bytes = head_dim // 2
    s_bytes = head_dim // fp4_block_size
    token_bytes = k_bytes + s_bytes

    # Build FP4 KV cache
    k_bf16 = torch.randn(num_pages * block_size, head_dim, dtype=torch.bfloat16, device="cuda")
    fp4, fe8m0 = fp4_act_quant(k_bf16, block_size=fp4_block_size, inplace=False)
    fp4_u8 = fp4.view(torch.uint8).view(num_pages, block_size, k_bytes)
    fe8m0_u8 = fe8m0.view(torch.uint8).view(num_pages, block_size, s_bytes)
    kvcache = torch.cat([fp4_u8, fe8m0_u8], dim=-1).unsqueeze(2)  # (P, B, 1, 68)

    # Build Q FP8
    q_bf16 = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_fp8, q_scale_flat = act_quant(q_bf16.view(-1, head_dim))
    q_fp8 = q_fp8.view(batch_size, 1, num_heads, head_dim)
    q_scale = q_scale_flat.view(batch_size, num_heads)

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    fn = lambda: tilelang_fp4_paged_mqa_logits(
        q_fp8, kvcache, q_scale, seq_lens, page_table, None, max_seq_len, clean_logits=False
    )
    # Warm-up compile
    fn()

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    if not _tilelang_available:
        print("TileLang not available, skipping benchmark.")
    elif not _fp4_available:
        print("torch.float4_e2m1fn_x2 not available (requires torch>=2.7), skipping.")
    else:
        print("=== fp4_act_quant benchmark ===")
        bench_fp4_act_quant.run(print_data=True)
        print()
        if torch.cuda.get_device_capability()[0] >= 9:
            print("=== tilelang_fp4_paged_mqa_logits benchmark ===")
            bench_fp4_paged_mqa_logits.run(print_data=True)
        else:
            print("Skipping paged MQA logits benchmark (requires SM90+).")
