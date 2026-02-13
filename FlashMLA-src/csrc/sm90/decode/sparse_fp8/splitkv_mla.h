#pragma once

#include "params.h"

namespace sm90 {

void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams &params, cudaStream_t stream);

}
