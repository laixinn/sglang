#pragma once

#include "params.h"

namespace sm90 {

void run_fwd_kernel(const SparsePrefillParams& params);

void run_merge_fwd_kernel(const SparseMergePrefillParams& params);

}
