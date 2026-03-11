/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

#define VEC_SIZE 4
using Vec = int4;

// Debug flag - set to 1 to enable debug prints
#define DEBUG_KERNEL 0

template <typename scalar_t, typename bool_t>
__global__ void moe_usc_hit_replace_kernel(
    const scalar_t* __restrict__ grounded_weights,
    const bool_t* __restrict__ miss_mask,
    scalar_t* __restrict__ hit_weights,
    const bool_t* __restrict__ hit_mask,
    const int32_t num_tokens,
    const int32_t topk) {
  int tid = threadIdx.x;
  int stride = blockDim.x / 2 * topk;
  int row_id = tid / 2;
  int role = tid % 2;

  int lane_id = tid % WARP_SIZE;
  int lane_offset = lane_id & ~1;
  unsigned mask_sync = 0x3 << lane_offset;

#if DEBUG_KERNEL
  if (tid > 1 && tid <= 3) {
    printf("DEBUGGING: tid=%d lane_id=%d lane_offset=%d mask_sync=%u\n", tid, lane_id, lane_offset, mask_sync);
  }
#endif

  // avoid accessing out of bounds
  if (row_id >= num_tokens) return;

  const int total_elems = num_tokens * topk;
#pragma unroll
  for (int base = 0; base < total_elems; base += stride){
    int offset = base + row_id * topk;
    // Bounds check: skip this chunk if the thread's row exceeds num_tokens
    if (offset + topk - 1 >= total_elems) break;
    int producer_idx = 0;
    int consumer_idx = 0;

    while (producer_idx < topk || consumer_idx < topk) {
      bool will_send = false;
      bool will_recv = false;
      scalar_t value_to_send = 0;

      if (role == 0) {
        // Producer
        while (producer_idx < topk && !will_send) {
          bool_t flag = miss_mask[offset + producer_idx];
#if DEBUG_KERNEL
          if (tid > 1 && tid <= 3) {
            printf("DEBUGGING: miss_mask[%d + %d] = %d\n", offset, producer_idx, flag);
          }
#endif
          if (!flag) {
            will_send = true;
            value_to_send = grounded_weights[offset + producer_idx];
          }
          
          producer_idx++;
        }
      }else{
        // Consumer
        while (consumer_idx < topk && !will_recv) {
          bool_t flag = hit_mask[offset + consumer_idx];
          if (flag) {
            will_recv = true;
          } else {
            consumer_idx++;
          }
        }
      }

      // Check if both sides of the pair are ready using single shuffle
      bool my_ready = (role == 0) ? will_send : will_recv;
      bool partner_ready = __shfl_xor_sync(mask_sync, my_ready, 1);
      bool can_exchange = my_ready && partner_ready;

#if DEBUG_KERNEL
      if (tid > 1 && tid <= 3){
        printf("DEBUGGING: tid=%d producer_idx=%d consumer_idx=%d will_send=%d will_recv=%d my_ready=%d partner_ready=%d can_exchange=%d\n", 
          tid, producer_idx, consumer_idx, will_send, will_recv, my_ready, partner_ready, can_exchange);
      }
#endif

      if (can_exchange) {
        scalar_t recevied_value = static_cast<scalar_t>(
          __shfl_xor_sync(mask_sync, static_cast<float>(value_to_send), 1)
        );

        if (role == 1 && will_recv) {
          hit_weights[offset + consumer_idx] = recevied_value;
          consumer_idx++;
        }

      } else {
        // If one side finished, break to avoid hanging
        break;
      }
    }

    __syncwarp(mask_sync);
  }
}


// ---------------------------------------------------------------------------
// fused_verify_remap: shared-memory hash-table based implementation
//
// For each token row n and each of its `topk` actual KV page indices, this
// kernel determines:
//   hit[n, i]   = 1  iff  actual[n, i] appears somewhere in predicted[n, :]
//                      AND  actual[n, i] >= 0  (not padding)
//   remap[n, i] = position j in predicted[n, :] where actual[n, i] == predicted[n, j]
//                 (0 when hit[n,i] == 0)
//
// Complexity vs. the tiled scan:
//   tiled scan      : O(topk × pred_topk) per token row
//   hash table      : O(topk + pred_topk) per token row
//
// Grid:  (N,)         — one CTA per token row
// Block: BLOCK threads — cooperate on insert + lookup
// Smem:  HT_SIZE × 2 × 4 bytes  (keys array + values array)
//
// HT_SIZE must be a power of 2 and >= 2 × pred_topk for a ≤50% load factor,
// which keeps expected probe length close to 1.
//
// Assumption: predicted indices are unique per row (no duplicate page IDs).
//             actual indices >= 0 are also unique per row.
//             Values in predicted[] are always >= 0 (valid page IDs).
// ---------------------------------------------------------------------------

// Knuth multiplicative hash — good distribution for dense integer keys.
__device__ __forceinline__ int ht_hash(int32_t key, int mask) {
    return (int)((uint32_t)(key * 2654435761u) & (uint32_t)mask);
}

// ---------------------------------------------------------------------------
// fused_verify_remap_kernel
//
// Template parameters (all compile-time):
//   K       = topk = pred_topk  (supported: 512, 1024, 2048)
//   BLOCK   = threadblock size  (K/2, capped at 1024)
//   HT_SIZE = hash-table slots  (2*K, load factor ≤ 50%)
//
// All three loop bounds (init: HT_SIZE/BLOCK, insert: K/BLOCK, lookup: K/BLOCK)
// are compile-time constants → nvcc fully unrolls them via #pragma unroll.
//
// Dispatch table (fused_verify_remap):
//   K= 512  →  BLOCK=256,  HT_SIZE=1024  ( 8 KB smem, 6 blocks/SM)
//   K=1024  →  BLOCK=512,  HT_SIZE=2048  (16 KB smem, 3 blocks/SM)
//   K=2048  →  BLOCK=1024, HT_SIZE=4096  (32 KB smem, 2 blocks/SM)
// ---------------------------------------------------------------------------
template <int BLOCK, int HT_SIZE, int K>
__global__ void fused_verify_remap_kernel(
    const int64_t* __restrict__ actual_ptr,
    const int64_t* __restrict__ predicted_ptr,
    int8_t*  __restrict__ hit_ptr,
    int32_t* __restrict__ remap_ptr,
    int stride_actual_row, int stride_pred_row)
{
    static_assert((HT_SIZE & (HT_SIZE - 1)) == 0, "HT_SIZE must be a power of 2");
    static_assert(K       % BLOCK == 0, "K must be divisible by BLOCK");
    static_assert(HT_SIZE % BLOCK == 0, "HT_SIZE must be divisible by BLOCK");
    constexpr int     HT_MASK = HT_SIZE - 1;
    constexpr int32_t EMPTY   = INT32_MIN;

    const int pid_n = blockIdx.x;
    const int tid   = threadIdx.x;

    __shared__ int32_t ht_keys[HT_SIZE];
    __shared__ int32_t ht_vals[HT_SIZE];

    // Init hash table (HT_SIZE/BLOCK iters — compile-time → fully unrolled).
#pragma unroll
    for (int i = tid; i < HT_SIZE; i += BLOCK)
        ht_keys[i] = EMPTY;
    __syncthreads();

    // Insert predicted indices (K/BLOCK iters — fully unrolled).
#pragma unroll
    for (int j = tid; j < K; j += BLOCK) {
        const int32_t key = (int32_t)predicted_ptr[(int64_t)pid_n * stride_pred_row + j];
        if (key < 0) continue;
        int slot = ht_hash(key, HT_MASK);
        while (atomicCAS(&ht_keys[slot], EMPTY, key) != EMPTY)
            slot = (slot + 1) & HT_MASK;
        ht_vals[slot] = j;
    }
    __syncthreads();

    // Lookup actual indices (K/BLOCK iters — fully unrolled).
#pragma unroll
    for (int i = tid; i < K; i += BLOCK) {
        const int32_t key   = (int32_t)actual_ptr[(int64_t)pid_n * stride_actual_row + i];
        const bool    valid = (key >= 0);
        int32_t found_pos = -1;
        if (valid) {
            int slot = ht_hash(key, HT_MASK);
            while (ht_keys[slot] != EMPTY) {
                if (ht_keys[slot] == key) { found_pos = ht_vals[slot]; break; }
                slot = (slot + 1) & HT_MASK;
            }
        }
        hit_ptr  [(int64_t)pid_n * K + i] = (int8_t)(found_pos >= 0 ? 1 : 0);
        remap_ptr[(int64_t)pid_n * K + i] = (found_pos >= 0) ? found_pos : 0;
    }
}

void fused_verify_remap(
    torch::Tensor actual_indices,
    torch::Tensor predicted_indices,
    torch::Tensor& hit_out,
    torch::Tensor& remap_out)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(actual_indices.scalar_type()    == at::kLong,
                "fused_verify_remap: actual_indices must be int64");
    TORCH_CHECK(predicted_indices.scalar_type() == at::kLong,
                "fused_verify_remap: predicted_indices must be int64");
    TORCH_CHECK(hit_out.scalar_type()           == at::kChar,
                "fused_verify_remap: hit_out must be int8");
    TORCH_CHECK(remap_out.scalar_type()         == at::kInt,
                "fused_verify_remap: remap_out must be int32");

    const int N             = actual_indices.size(0);
    const int topk          = actual_indices.size(1);
    const int stride_actual = actual_indices.stride(0);
    const int stride_pred   = predicted_indices.stride(0);

    TORCH_CHECK(N > 0, "fused_verify_remap: empty tensors not supported");
    TORCH_CHECK(topk == (int)predicted_indices.size(1),
                "fused_verify_remap: topk and pred_topk must be equal, got ",
                topk, " vs ", predicted_indices.size(1));

    const auto* act_ptr  = actual_indices.data_ptr<int64_t>();
    const auto* pred_ptr = predicted_indices.data_ptr<int64_t>();
    auto*       hit_p    = (int8_t*)hit_out.data_ptr();
    auto*       remap_p  = remap_out.data_ptr<int32_t>();

    switch (topk) {
        case 512:
            fused_verify_remap_kernel<256, 1024, 512>
                <<<dim3(N), dim3(256), 0, stream>>>(
                    act_ptr, pred_ptr, hit_p, remap_p, stride_actual, stride_pred);
            break;
        case 1024:
            fused_verify_remap_kernel<512, 2048, 1024>
                <<<dim3(N), dim3(512), 0, stream>>>(
                    act_ptr, pred_ptr, hit_p, remap_p, stride_actual, stride_pred);
            break;
        case 2048:
            fused_verify_remap_kernel<1024, 4096, 2048>
                <<<dim3(N), dim3(1024), 0, stream>>>(
                    act_ptr, pred_ptr, hit_p, remap_p, stride_actual, stride_pred);
            break;
        default:
            TORCH_CHECK(false,
                "fused_verify_remap: unsupported topk=", topk,
                "; supported values are 512, 1024, 2048");
    }
}


// ---------------------------------------------------------------------------
// moe_usc_hit_replace (existing)
// ---------------------------------------------------------------------------

void moe_usc_hit_replace(
  torch::Tensor grounded_weights,
  torch::Tensor miss_mask,
  torch::Tensor& hit_weights,
  torch::Tensor hit_mask) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t num_tokens = grounded_weights.size(0);
  int32_t topk = grounded_weights.size(1);

  TORCH_CHECK(hit_weights.dim() == 2, "hit_weights must be 2D [num_tokens, topk]");
  TORCH_CHECK(hit_weights.size(0) == num_tokens && hit_weights.size(1) == topk,
              "hit_weights must have shape [num_tokens, topk]");
  TORCH_CHECK(grounded_weights.scalar_type() == hit_weights.scalar_type(), "grounded_weights and hit_weights must have the same scalar type");

  int threads_per_block = 2 * 512;
  threads_per_block = ((threads_per_block + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  DISPATCH_FLOAT_TYPES(grounded_weights.scalar_type(), "moe_usc_hit_replace_kernel", [&] {
    using bool_t = bool;
    const int32_t threads = max((int32_t)threads_per_block, WARP_SIZE);
    const int32_t shared_mem_size = 0; //(threads * topk) * sizeof(int32_t);

    auto replace_kernel = moe_usc_hit_replace_kernel<scalar_t, bool_t>;
    replace_kernel<<<1, threads, shared_mem_size, stream>>>(
        grounded_weights.data_ptr<scalar_t>(),
        (bool_t*)miss_mask.data_ptr(),
        hit_weights.data_ptr<scalar_t>(),
        (bool_t*)hit_mask.data_ptr(),
        num_tokens,
        topk
    );
    
    return true;
  });
}