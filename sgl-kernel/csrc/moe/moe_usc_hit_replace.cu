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

#pragma unroll
  for (int base = 0; base < num_tokens * topk; base += stride){
    int offset = base + row_id * topk;
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