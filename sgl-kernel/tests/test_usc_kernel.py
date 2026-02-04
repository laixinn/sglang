#!/usr/bin/env python3
"""
Unit tests for USC Hit Replace CUDA kernel
"""

from math import e
import torch
import pytest
import triton
import numpy as np

from sglang.srt.layers.moe.usc.moe_cache import verify_cache_mask_triton
from sgl_kernel import moe_usc_hit_replace

def usc_hit_replace_native(
    grounded_weights,
    miss_mask,
    hit_weights,
    hit_mask,
):
    new_hit_weights = hit_weights.clone()
    new_hit_weights[hit_mask] = grounded_weights[~miss_mask]
    return new_hit_weights

def generate_test_data(num_tokens, topk, num_experts, dtype, device):
    grounded_weights = torch.randn(num_tokens, topk, device=device, dtype=dtype)
    estimated_weights = torch.randn(num_tokens, topk, device=device, dtype=dtype)
    grounded_idx = torch.randn((num_tokens, num_experts), device=device, dtype=dtype)
    grounded_idx = grounded_idx.topk(k=topk, dim=1)[1]
    estimated_idx = torch.randn((num_tokens, num_experts), device=device, dtype=dtype)
    estimated_idx = estimated_idx.topk(k=topk, dim=1)[1]
    hit_mask, miss_mask = verify_cache_mask_triton(estimated_idx, grounded_idx)
    assert (hit_mask.sum(1) + miss_mask.sum(1) == topk).all()
    assert ((~miss_mask).sum(1) == hit_mask.sum(1)).all()
    return grounded_weights, estimated_weights, grounded_idx, estimated_idx, hit_mask, miss_mask

def test_usc_hit_replace():
    # TODO: fix num_tokens > 8
    for num_tokens in [8, 1024, 2048, 4096]:
        for topk in [8, 16, 32]:
            for num_experts in [32, 64, 128]:
                for dtype in [torch.bfloat16, torch.float32]:
                    # NOTE: ~miss_mask will lead to bitwise inversion, which is not what we want
                    grounded_weights, estimated_weights, grounded_idx, estimated_idx, hit_mask, miss_mask = \
                        generate_test_data(num_tokens=num_tokens, topk=topk, num_experts=num_experts, dtype=dtype, device='cuda')

                    native_lambda = lambda: usc_hit_replace_native(grounded_weights, miss_mask, estimated_weights, hit_mask)

                    cuda_lambda = lambda: moe_usc_hit_replace(grounded_weights, miss_mask, estimated_weights, hit_mask)

                    # check accuracy
                    output_ref = native_lambda()
                    output_ref_clone = output_ref.clone()
                    torch.cuda.synchronize()
                    output_cuda = cuda_lambda()
                    torch.testing.assert_close(output_ref, output_ref_clone, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(output_cuda, output_ref, rtol=1e-5, atol=1e-5)
                    print(f"✓ Basic test passed for num_tokens={num_tokens}, topk={topk}, num_experts={num_experts}, dtype={dtype}")

                    # benchmark
                    torch.cuda.synchronize()
                    quantiles = [0.2, 0.5, 0.8]
                    ms_native, min_ms_native, max_ms_native = triton.testing.do_bench(native_lambda, quantiles=quantiles)
                    ms_cuda, min_ms_cuda, max_ms_cuda = triton.testing.do_bench(cuda_lambda, quantiles=quantiles)

                    print(f"Native: {ms_native:.4f} ms, {min_ms_native:.4f} ms, {max_ms_native:.4f} ms")
                    print(f"CUDA: {ms_cuda:.4f} ms, {min_ms_cuda:.4f} ms, {max_ms_cuda:.4f} ms")


def usc_verify_cache_mask_native(
    grounded_idx,
    estimated_idx,
):
    hit_token_mask = torch.zeros_like(grounded_idx, dtype=torch.bool)
    miss_token_mask = torch.zeros_like(grounded_idx, dtype=torch.bool)

    M = grounded_idx.shape[0]
    for i in range(M):
        # apply on cache topk_idx
        hit_token_mask[i, :] = torch.isin(estimated_idx[i, :], grounded_idx[i, :], assume_unique=True)
        # apply on new index
        miss_token_mask[i, :] = torch.isin(grounded_idx[i, :], estimated_idx[i, :], assume_unique=True, invert=True)
    return hit_token_mask, miss_token_mask

def usc_verify_cache_mask_triton(
    grounded_idx,
    estimated_idx,
):
    hit_token_mask = torch.zeros_like(grounded_idx, dtype=torch.bool)
    miss_token_mask = torch.zeros_like(grounded_idx, dtype=torch.bool)
    hit_token_mask, miss_token_mask = verify_cache_mask_triton(grounded_idx, estimated_idx)
    return hit_token_mask, miss_token_mask

def test_usc_verify_cache_mask():
    for num_tokens in [1024, 2048, 4096]:
        for topk in [8, 16, 32]:
            for num_experts in [32, 64, 128]:
                grounded_weights, estimated_weights, grounded_idx, estimated_idx, hit_mask, miss_mask = \
                    generate_test_data(num_tokens=num_tokens, topk=topk, num_experts=num_experts, dtype=torch.bfloat16, device='cuda')

                native_lambda = lambda: usc_verify_cache_mask_native(grounded_idx, estimated_idx)
                cuda_lambda = lambda: usc_verify_cache_mask_triton(grounded_idx, estimated_idx)

                # check accuracy
                hit_native, miss_native = native_lambda()
                hit_cuda, miss_cuda = cuda_lambda()
                torch.testing.assert_close(hit_cuda, hit_native, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(miss_cuda, miss_native, rtol=1e-5, atol=1e-5)
                print(f"✓ Basic test passed for num_tokens={num_tokens}, topk={topk}, num_experts={num_experts}")

                # benchmark
                torch.cuda.synchronize()
                quantiles = [0.2, 0.5, 0.8]
                ms_native, min_ms_native, max_ms_native = triton.testing.do_bench(native_lambda, quantiles=quantiles)
                ms_cuda, min_ms_cuda, max_ms_cuda = triton.testing.do_bench(cuda_lambda, quantiles=quantiles)

                print(f"Native: {ms_native:.4f} ms, {min_ms_native:.4f} ms, {max_ms_native:.4f} ms")
                print(f"CUDA: {ms_cuda:.4f} ms, {min_ms_cuda:.4f} ms, {max_ms_cuda:.4f} ms")


if __name__ == '__main__':
    # test_usc_verify_cache_mask()
    test_usc_hit_replace()

