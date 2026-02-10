#!/usr/bin/env python3
"""
Standalone build script for moe_usc_hit_replace kernel.
Compiles just this one CUDA kernel without needing the full sgl-kernel build.

Usage:
    python build.py

After building, add the following to your Python code (before importing sgl_kernel):
    import torch
    torch.ops.load_library("/home/hadoop-djst-algoplat/sglang/sgl-kernel/build_usc_kernel/moe_usc_ext.so")
"""

import os
import torch
from torch.utils.cpp_extension import load

sgl_kernel_dir = os.path.join(os.path.dirname(__file__), "..")

print("Building moe_usc_hit_replace kernel...")
print(f"  CUDA kernel: {sgl_kernel_dir}/csrc/moe/moe_usc_hit_replace.cu")
print(f"  Include dir: {sgl_kernel_dir}/include")

module = load(
    name="moe_usc_ext",
    sources=[
        os.path.join(os.path.dirname(__file__), "wrapper.cpp"),
        os.path.join(sgl_kernel_dir, "csrc/moe/moe_usc_hit_replace.cu"),
    ],
    extra_include_paths=[
        os.path.join(sgl_kernel_dir, "include"),
    ],
    extra_cuda_cflags=[
        "-gencode=arch=compute_90a,code=sm_90a",
        "-O3",
    ],
    extra_cflags=["-O3"],
    build_directory=os.path.dirname(__file__),
    verbose=True,
)

# Find the built .so file
so_file = None
build_dir = os.path.dirname(__file__)
for f in os.listdir(build_dir):
    if f.endswith(".so") and "moe_usc" in f:
        so_file = os.path.join(build_dir, f)
        break

if so_file:
    print(f"\n✅ Build successful! Library: {so_file}")
    print(f"\nTo verify:")
    print(f"  python -c \"import torch; torch.ops.load_library('{so_file}'); print(torch.ops.sgl_kernel.moe_usc_hit_replace)\"")
else:
    print("\n❌ Build failed - .so not found")
