#!/bin/bash
cd /home/hadoop-djst-algoplat/sglang-original

export PYTHONPATH=./python:$PYTHONPATH

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY

# 跳过 flashinfer 版本检查
export FLASHINFER_DISABLE_VERSION_CHECK=1
# DeepEP Auto requires DeepGEMM; keep JIT enabled but disable pre-compilation to avoid large one-time allocations
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_JIT_DEEPGEMM=1

# 内存优化：减少 CUDA 内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# USC Cache 配置:
# SGLANG_ENABLE_USC_CACHE=1 启用完整 hit/miss 分离流程
# SGLANG_ENABLE_USC_CACHE=0 启用 normal sbo 模式, 
# 默认为0, 如果启用usc cache hit/miss, 则需要设置 SGLANG_ENABLE_USC_CACHE=1
export SGLANG_ENABLE_USC_CACHE=1


python3 -m sglang.launch_server \
    --model-path /home/hadoop-djst-algoplat/models/deepseek-ai/DeepSeek-V3.2-Exp \
    --tp 8 \
    --disable-cuda-graph \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --chunked-prefill-size 1024 --mem-fraction-static 0.80 \
    2>&1 | tee server.log



# 测试命令
# unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY
# python3 test_requests.py    # test single request
# python3 benchmark/gsm8k/bench_sglang.py  --num-questions 20