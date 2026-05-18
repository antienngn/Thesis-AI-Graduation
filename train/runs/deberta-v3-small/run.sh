#!/usr/bin/env bash
# Train DeBERTa-v3-small (microsoft/deberta-v3-small, ~44M) predictor.
# Yêu cầu `sentencepiece` package.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python trainer.py \
  --config runs/deberta-v3-small/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id deberta-v3-small-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 32 \
  --epoch 500 \
  --label-group-size 10 \
  --loss listMLE
