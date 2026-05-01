import json
from pathlib import Path

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


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSV4_PRO_CONFIG_PATH = Path(
    "/workdir/huggingface.co/deepseek-ai/DeepSeek-V4-Pro/config.json"
)
_DSV4_FLASH_CONFIG_PATH = Path(
    "/workdir/huggingface.co/deepseek-ai/DeepSeek-V4-Flash/config.json"
)
_DSV4_CONFIG_SOURCES = [
    ("deepseek-v4-pro-checkpoint", _DSV4_PRO_CONFIG_PATH),
    ("deepseek-v4-flash-checkpoint", _DSV4_FLASH_CONFIG_PATH),
    (
        "deepseek-v4-flash-packaged-small",
        _REPO_ROOT / "python/sglang/srt/configs/config_backup_small.json",
    ),
    (
        "deepseek-v4-pro-packaged-large-override",
        _REPO_ROOT / "python/sglang/srt/configs/config_backup_large.json",
    ),
]


def _load_dsv4_config_cases():
    cases = []
    for name, path in _DSV4_CONFIG_SOURCES:
        if not path.exists():
            continue
        with path.open() as f:
            config = json.load(f)
        cases.append({"name": name, "path": str(path), "config": config})
    return cases


DSV4_CONFIG_CASES = _load_dsv4_config_cases()
DSV4_CONFIG_IDS = [case["name"] for case in DSV4_CONFIG_CASES]


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
    if topk >= 4:
        indices[:, :, 0] = -1
        indices[:, :, 1] = 0
        indices[:, :, 2] = max_index - 1
    return indices


def _make_lengths(num_q: int, topk: int, shape: str) -> torch.Tensor:
    lengths = torch.randint(0, topk + 1, (num_q,), dtype=torch.int32, device="cuda")
    if num_q >= 4:
        edge_lengths = torch.tensor([0, 1, max(topk - 1, 0), topk], device="cuda")
        lengths[:4] = edge_lengths.to(torch.int32)

    if shape == "1d":
        return lengths
    if shape == "2d":
        return lengths.view(num_q, 1)
    if shape == "3d":
        return lengths.view(num_q, 1, 1)
    raise AssertionError(f"unsupported length shape: {shape}")


def _get_config_int(config_case, key: str) -> int:
    value = config_case["config"][key]
    assert isinstance(value, int), f"{config_case['name']} {key}={value}"
    return value


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


@pytest.mark.parametrize("config_case", DSV4_CONFIG_CASES, ids=DSV4_CONFIG_IDS)
def test_dsv4_unified_prefill_inputs_model_config_match_torch_ref(config_case):
    torch.manual_seed(3)

    config = config_case["config"]
    assert config["head_dim"] == 512
    assert config["num_key_value_heads"] == 1
    assert 4 in config["compress_ratios"]

    num_q = 129
    num_heads = _get_config_int(config_case, "index_n_heads")
    swa_topk = _get_config_int(config_case, "sliding_window")
    extra_topk = _get_config_int(config_case, "index_topk")

    q = torch.randn((num_q, 1, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    swa_cache = _make_dsv4_cache(8, 256)
    extra_cache = _make_dsv4_cache(max(16, extra_topk // 16), 64)

    swa_indices = _make_indices(num_q, swa_topk, swa_cache.shape[0] * swa_cache.shape[1])
    extra_indices = _make_indices(
        num_q, extra_topk, extra_cache.shape[0] * extra_cache.shape[1]
    )
    swa_lengths = _make_lengths(num_q, swa_topk, "2d")
    extra_lengths = _make_lengths(num_q, extra_topk, "2d")

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


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "config_case": config_case,
                "num_q": 257,
                "swa_blocks": 16,
                "swa_block_size": 256,
                "extra_blocks": max(
                    96, _get_config_int(config_case, "index_topk") // 16
                ),
                "extra_block_size": 64,
                "extra_topk": _get_config_int(config_case, "index_topk"),
                "swa_length_shape": "2d",
                "extra_length_shape": "2d",
            },
            id=f"{config_case['name']}-c4-topk",
        )
        for config_case in DSV4_CONFIG_CASES
    ]
    + [
        pytest.param(
            {
                "config_case": config_case,
                "num_q": 128,
                "swa_blocks": 32,
                "swa_block_size": 256,
                "extra_blocks": 64,
                "extra_block_size": 128,
                "extra_topk": min(
                    8192, _get_config_int(config_case, "max_position_embeddings") // 128
                ),
                "swa_length_shape": "3d",
                "extra_length_shape": "3d",
            },
            id=f"{config_case['name']}-c128-long-context",
        )
        for config_case in DSV4_CONFIG_CASES
        if 128 in config_case["config"]["compress_ratios"]
    ],
)
def test_dsv4_unified_prefill_inputs_long_context_match_torch_ref(case):
    torch.manual_seed(4)

    config_case = case["config_case"]
    num_q = case["num_q"]
    num_heads = _get_config_int(config_case, "index_n_heads")
    q = torch.randn((num_q, 1, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    swa_cache = _make_dsv4_cache(case["swa_blocks"], case["swa_block_size"])
    extra_cache = _make_dsv4_cache(case["extra_blocks"], case["extra_block_size"])

    swa_topk = _get_config_int(config_case, "sliding_window")
    swa_indices = _make_indices(
        num_q, swa_topk, swa_cache.shape[0] * swa_cache.shape[1]
    )
    extra_indices = _make_indices(
        num_q,
        case["extra_topk"],
        extra_cache.shape[0] * extra_cache.shape[1],
    )
    swa_lengths = _make_lengths(num_q, swa_topk, case["swa_length_shape"])
    extra_lengths = _make_lengths(
        num_q, case["extra_topk"], case["extra_length_shape"]
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


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "name": "medium",
                "config_case": DSV4_CONFIG_CASES[0],
                "num_q": 512,
                "extra_blocks": 128,
                "extra_block_size": 64,
                "iters": 10,
            },
            id="medium",
        ),
        pytest.param(
            {
                "name": "long-context",
                "config_case": DSV4_CONFIG_CASES[0],
                "num_q": 128,
                "extra_blocks": 64,
                "extra_block_size": 128,
                "iters": 5,
            },
            id="long-context",
        ),
    ],
)
def test_dsv4_unified_prefill_inputs_triton_is_faster_than_torch_ref(case):
    torch.manual_seed(2)

    config_case = case["config_case"]
    num_q = case["num_q"]
    num_heads = _get_config_int(config_case, "index_n_heads")
    q = torch.randn((num_q, 1, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    swa_cache = _make_dsv4_cache(64, 256)
    extra_cache = _make_dsv4_cache(case["extra_blocks"], case["extra_block_size"])
    swa_topk = _get_config_int(config_case, "sliding_window")
    extra_topk = (
        _get_config_int(config_case, "index_topk")
        if case["name"] == "medium"
        else min(8192, _get_config_int(config_case, "max_position_embeddings") // 128)
    )
    swa_indices = _make_indices(
        num_q, swa_topk, swa_cache.shape[0] * swa_cache.shape[1]
    )
    extra_indices = _make_indices(
        num_q, extra_topk, extra_cache.shape[0] * extra_cache.shape[1]
    )
    swa_lengths = _make_lengths(num_q, swa_topk, "1d")
    extra_lengths = _make_lengths(num_q, extra_topk, "1d")

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

    torch_ms = _time_cuda_ms(torch_ref, warmup=3, iters=case["iters"])
    triton_ms = _time_cuda_ms(triton_impl, warmup=3, iters=case["iters"])
    speedup = torch_ms / triton_ms
    print(
        f"DSV4 unified prefill staging ({case['name']}, {config_case['name']}): "
        f"torch_ref={torch_ms:.3f} ms triton={triton_ms:.3f} ms "
        f"speedup={speedup:.2f}x"
    )
    assert triton_ms < torch_ms * 0.75
