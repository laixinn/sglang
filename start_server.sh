#!/bin/bash
cd /home/hadoop-djst-algoplat/sglang

unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY

# export FLASHINFER_DISABLE_VERSION_CHECK=1
# # DeepEP Auto requires DeepGEMM; keep JIT enabled but disable pre-compilation to avoid large one-time allocations
# export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
# export SGLANG_ENABLE_JIT_DEEPGEMM=1

MODEL_PATH=/home/hadoop-djst-algoplat/models/deepseek-ai/DeepSeek-V3.2-Exp/

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8418 \
        --tp 8 \
        --attention-backend nsa \
        --nsa-prefill-backend flashmla_sparse \
        --nsa-prefill-cp-mode round-robin-split \
        --enable-nsa-prefill-context-parallel \
        --mem-fraction-static 0.8 \
        --max-running-requests 128 \
        --chunked-prefill-size 8192 \
        --page-size 64 \
        --disable-radix-cache \
        --tool-call-parser deepseekv32 \
        --reasoning-parser deepseek-v3 \
        --enable-metrics --enable-cache-report --enable-usc \
        2>&1 | tee server.log &



# gsm8k
# nohup - 防止终端关闭后进程被杀
# > gsm8k_bench.log 2>&1 - 输出重定向到日志文件
# & - 后台运行
# nohup python3 benchmark/gsm8k/bench_sglang.py --host http://localhost --port 8418 --num-questions 100 > gsm8k_bench.log 2>&1 &

