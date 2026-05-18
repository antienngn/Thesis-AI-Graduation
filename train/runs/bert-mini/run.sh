#!/usr/bin/env bash
# Train BERT-mini (prajjwal1/bert-mini, ~11M) predictor on ShareGPT/Llama-3-8B labels.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python trainer.py \
  --config runs/bert-mini/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id bert-mini-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 32 \
  --epoch 1000 \
  --label-group-size 10 \
  --loss listMLE
