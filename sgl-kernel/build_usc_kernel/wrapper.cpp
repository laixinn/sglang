// Standalone wrapper to register moe_usc_hit_replace as a torch op
// under the sgl_kernel namespace, so that:
//   torch.ops.sgl_kernel.moe_usc_hit_replace.default(...)
// works from Python.

#include <torch/extension.h>

// Declaration of the CUDA function (defined in moe_usc_hit_replace.cu)
void moe_usc_hit_replace(
    torch::Tensor grounded_weights,
    torch::Tensor miss_mask,
    torch::Tensor& hit_weights,
    torch::Tensor hit_mask);

// Register the op under sgl_kernel namespace using TORCH_LIBRARY_FRAGMENT
// (extends the existing sgl_kernel library from the installed common_ops.so)
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("moe_usc_hit_replace(Tensor grounded_weights, Tensor miss_mask, Tensor(a!) hit_weights, Tensor hit_mask) -> ()");
    m.impl("moe_usc_hit_replace", torch::kCUDA, &moe_usc_hit_replace);
}
