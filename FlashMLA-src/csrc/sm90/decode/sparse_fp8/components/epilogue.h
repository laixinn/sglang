#pragma once

#include "named_barriers.h"

// Store O / OAccum
template<
    bool IS_NO_SPLIT,
    typename TMAParams,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename Tensor3
>
__forceinline__ __device__ void store_o(
    Tensor0 &rO,	// ((2, 2, 32), 1, 1)
    Tensor1 &gOorAccum,	// (BLOCK_SIZE_M, HEAD_DIM_V)
    Tensor2 &sOutputBuf,
    Tensor3 &sOutputAccumBuf,
    float rL[2],
    TMAParams &tma_params,
    int batch_idx,
    int s_q_idx,
    int head_block_idx,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using cutlass::arch::NamedBarrier;
    if constexpr (IS_NO_SPLIT) {
        // Should convert the output to bfloat16 / float16, and save it to O
        Tensor rOb = make_tensor_like<bf16>(rO);
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < size(rO); ++idx) {
            rOb(idx) = (bf16)(rO(idx) / rL[idx%4 >= 2]);
        }

        Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));
        TiledCopy r2s_tiled_copy = make_tiled_copy_C(
            Copy_Atom<SM90_U32x4_STSM_N, bf16>{},
            TiledMMA_PV_LocalP{}
        );
        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
        cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);
        cutlass::arch::fence_view_async_shared();
        
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

        if (threadIdx.x == 0) {
            Tensor tma_gO = tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, batch_idx);
            auto thr_tma = tma_params.tma_O.get_slice(_0{});
            Tensor my_tma_gO = flat_divide(tma_gO, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{})(_, _, head_block_idx, _0{});
            cute::copy(
                tma_params.tma_O,
                thr_tma.partition_S(sOutputBuf),
                thr_tma.partition_D(my_tma_gO)
            );
            cute::tma_store_arrive();
        }
    } else {
        // Should save the result to OAccum
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < size(rO); idx += 2) {
            int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (idx%4 >= 2 ? 8 : 0);
            int col = warpgroup_idx*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
            *(float2*)(&(sOutputAccumBuf(row, col))) = float2 {
                rO(idx) / rL[idx%4 >= 2],
                rO(idx+1) / rL[idx%4 >= 2],
            };
        }
        cutlass::arch::fence_view_async_shared();
        
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);
        
        if (elect_one_sync()) {
            CUTLASS_PRAGMA_UNROLL
            for (int local_row = 0; local_row < BLOCK_M / (256/32); ++local_row) {
                int row = local_row * (256/32) + (threadIdx.x / 32);
                if (row < num_valid_seq_q) {
                    SM90_BULK_COPY_S2G::copy(&sOutputAccumBuf(row, _0{}), &gOorAccum(row, _0{}), HEAD_DIM_V*sizeof(float));
                }
            }
            cute::tma_store_arrive();
        }
    }
}
