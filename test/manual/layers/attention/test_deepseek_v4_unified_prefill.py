import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
    _dsv4_build_unified_prefill_inputs_from_real_decode,
    _dsv4_build_unified_prefill_inputs_from_real_decode_torch_ref,
    _dsv4_dequantize_model1_fp8_sparse_k_cache,
    _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref,
)
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    fp8_dtype,
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.utils import ceil_div


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_dsv4_cache(
    num_blocks: int,
    block_size: int,
    *,
    padded_page: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    num_tokens = num_blocks * block_size
    k = torch.randn((num_tokens, 512), dtype=torch.bfloat16, device=device)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k)

    page_bytes = block_size * 584
    if padded_page:
        page_bytes = ceil_div(page_bytes, 576) * 576
    page = torch.empty((num_blocks, page_bytes), dtype=torch.uint8, device=device)
    page.zero_()

    cache = page.view(fp8_dtype)[:, : block_size * 584].view(
        num_blocks, block_size, 1, 584
    )
    flat = cache.view(num_blocks, -1)

    values = flat[:, : block_size * 576].view(num_blocks, block_size, 576)
    values[:, :, :448].copy_(pack.k_nope_fp8.view(num_blocks, block_size, 448))
    values[:, :, 448:576].view(torch.bfloat16).copy_(
        pack.k_rope_bf16.view(num_blocks, block_size, 64)
    )

    scales = flat[:, block_size * 576 :].view(torch.uint8).view(
        num_blocks, block_size, 8
    )
    scales[:, :, :7].copy_(
        pack.scale_k_nope_ue8m0.view(torch.uint8).view(num_blocks, block_size, 7)
    )
    scales[:, :, 7].zero_()
    return cache


def _make_indices(num_q: int, topk: int, max_index: int) -> torch.Tensor:
    indices = torch.randint(
        -8, max_index, (num_q, 1, topk), dtype=torch.int32, device="cuda"
    )
    indices[indices < 0] = -1
    return indices


def _assert_prefill_inputs_close(actual, expected) -> None:
    assert actual is not None
    assert expected is not None
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _time_cuda_ms(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.parametrize("padded_page", [False, True])
def test_dsv4_dequantize_model1_fp8_sparse_k_cache_matches_torch_ref(padded_page):
    torch.manual_seed(0)

    cache = _make_dsv4_cache(3, 256, padded_page=padded_page)
    expected = _dsv4_dequantize_model1_fp8_sparse_k_cache_torch_ref(cache)
    actual = _dsv4_dequantize_model1_fp8_sparse_k_cache(cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("has_extra", [False, True])
def test_dsv4_unified_prefill_inputs_match_torch_ref(has_extra):
    torch.manual_seed(1)

    num_q = 17
    q = torch.randn((num_q, 1, 64, 512), dtype=torch.bfloat16, device="cuda")
    swa_cache = _make_dsv4_cache(4, 256)
    swa_indices = _make_indices(num_q, 128, swa_cache.shape[0] * swa_cache.shape[1])
    swa_lengths = torch.randint(0, 129, (num_q,), dtype=torch.int32, device="cuda")

    extra_cache = None
    extra_indices = None
    extra_lengths = None
    if has_extra:
        extra_cache = _make_dsv4_cache(5, 64)
        extra_indices = _make_indices(
            num_q, 512, extra_cache.shape[0] * extra_cache.shape[1]
        )
        extra_lengths = torch.randint(
            0, 513, (num_q,), dtype=torch.int32, device="cuda"
        )

    expected = _dsv4_build_unified_prefill_inputs_from_real_decode_torch_ref(
        q,
        swa_cache,
        extra_cache,
        swa_indices,
        extra_indices,
        swa_lengths,
        extra_lengths,
    )
    actual = _dsv4_build_unified_prefill_inputs_from_real_decode(
        q,
        swa_cache,
        extra_cache,
        swa_indices,
        extra_indices,
        swa_lengths,
        extra_lengths,
    )

    _assert_prefill_inputs_close(actual, expected)


def test_dsv4_unified_prefill_inputs_triton_is_faster_than_torch_ref():
    torch.manual_seed(2)

    num_q = 512
    q = torch.randn((num_q, 1, 64, 512), dtype=torch.bfloat16, device="cuda")
    swa_cache = _make_dsv4_cache(64, 256)
    extra_cache = _make_dsv4_cache(128, 64)
    swa_indices = _make_indices(num_q, 128, swa_cache.shape[0] * swa_cache.shape[1])
    extra_indices = _make_indices(
        num_q, 512, extra_cache.shape[0] * extra_cache.shape[1]
    )
    swa_lengths = torch.randint(0, 129, (num_q,), dtype=torch.int32, device="cuda")
    extra_lengths = torch.randint(0, 513, (num_q,), dtype=torch.int32, device="cuda")

    def torch_ref():
        return _dsv4_build_unified_prefill_inputs_from_real_decode_torch_ref(
            q,
            swa_cache,
            extra_cache,
            swa_indices,
            extra_indices,
            swa_lengths,
            extra_lengths,
        )

    def triton_impl():
        return _dsv4_build_unified_prefill_inputs_from_real_decode(
            q,
            swa_cache,
            extra_cache,
            swa_indices,
            extra_indices,
            swa_lengths,
            extra_lengths,
        )

    _assert_prefill_inputs_close(triton_impl(), torch_ref())

    torch_ms = _time_cuda_ms(torch_ref, warmup=3, iters=10)
    triton_ms = _time_cuda_ms(triton_impl, warmup=3, iters=10)
    speedup = torch_ms / triton_ms
    print(
        "DSV4 unified prefill staging: "
        f"torch_ref={torch_ms:.3f} ms triton={triton_ms:.3f} ms "
        f"speedup={speedup:.2f}x"
    )
    assert triton_ms < torch_ms * 0.75
