#!/usr/bin/env bash
# Подбор числа CPU-потоков для Qwen: число слоёв с экспертами на CPU берётся из llama-fit-params.
set -euo pipefail
cd "$(dirname "$0")"
LLMS_DIR="${LLMS_DIR:-$HOME/llms}"
BIN_DIR="$LLMS_DIR/llama.cpp/llama-b11524"
MODEL="$LLMS_DIR/models/qwen38/Qwen3.8-Flash-Next-UD-Q3_K_XL-reap256-00001-of-00002.gguf"
mkdir -p results

FIT_ARGS=$("$BIN_DIR/llama-fit-params" -m "$MODEL" -c 16384 -fa on 2>/dev/null | tail -1)
NCMOE=$(grep -oE 'blk\\\.[0-9]+\\\.ffn_\(up' <<<"$FIT_ARGS" | wc -l)
echo "n_cpu_moe (from --fit): $NCMOE" | tee results/bench_threads.md
"$BIN_DIR/llama-bench" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -fa 1 -t 8,12,16 -p 256 -n 64 -r 1 -o md \
  | tee -a results/bench_threads.md
