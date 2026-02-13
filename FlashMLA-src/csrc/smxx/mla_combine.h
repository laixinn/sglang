#pragma once

#include "params.h"

template<typename ElementT>
void run_flash_mla_combine_kernel(DecodingParams &params, cudaStream_t stream);
