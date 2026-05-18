#!/usr/bin/env bash
# Train DistilGPT2 (distilgpt2, ~82M) predictor on ShareGPT/Llama-3-8B labels.
# Decoder-only; pad_token được reuse từ eos_token trong PredModel.__init__.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python trainer.py \
  --config runs/distilgpt2/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id distilgpt2-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 32 \
  --epoch 100 \
  --label-group-size 10 \
  --loss listMLE
