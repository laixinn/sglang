#!/bin/bash
unset http_proxy && unset https_proxy
export SGLANG_OPT_USE_TILELANG_INDEXER_FP4=true
nohup python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /workdir/huggingface.co/deepseek-ai/DeepSeek-V4-Pro/ \
  --tp 8 \
  --moe-runner-backend marlin \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.88 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 30000 \
  --log-level debug \
  > dsv4.log 2>&1 &