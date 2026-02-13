#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

using namespace cute;

static constexpr int NUM_THREADS = 128*3;
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;
static constexpr int QUANT_TILE_SIZE = 128;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_V;
static constexpr int HEAD_DIM_ROPE = HEAD_DIM_K - HEAD_DIM_V;
static constexpr int NUM_SCALES = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
static constexpr int NUM_BYTES_PER_TOKEN = HEAD_DIM_NOPE + NUM_SCALES*sizeof(float) + HEAD_DIM_ROPE*sizeof(bf16);

static constexpr int NUM_K_BUFS = 2;

using SmemLayoutQTile = decltype(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<BLOCK_M>, Int<64>>{}
));

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(tile_to_shape(
    SmemLayoutQTile{},
    Shape<Int<BLOCK_M>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
));

using SmemLayoutQ = SmemLayoutQTiles<9>;

using SmemLayoutKTile = decltype(tile_to_shape(
    GMMA::Layout_INTER_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<TOPK_BLOCK_SIZE>, _64>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(tile_to_shape(
    SmemLayoutKTile{},
    Shape<Int<TOPK_BLOCK_SIZE>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
	SmemLayoutKTiles<NUM_TILES>{},
	Layout<Shape<Int<64*NUM_TILES>, Int<TOPK_BLOCK_SIZE>>, Stride<Int<TOPK_BLOCK_SIZE>, _1>>{}
));

using SmemLayoutOBuf = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}
));

using SmemLayoutOAccumBuf = Layout<
    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
>;

using SmemLayoutK = SmemLayoutKTiles<9>;
using SmemLayoutV = SmemLayoutKTilesTransposed<8>;
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<4>;

using SmemLayoutS = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}
));

struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    union {
        array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;
        array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;
    } u;
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M];
    transac_bar_t bar_q, bar_k_local_ready[NUM_K_BUFS], bar_k_remote_ready[NUM_K_BUFS], bar_k_avail[NUM_K_BUFS];
};

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_QK_rQ = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));
