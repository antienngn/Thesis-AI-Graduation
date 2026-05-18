#!/usr/bin/env bash
# Train BERT-tiny (prajjwal1/bert-tiny, ~4M) predictor on ShareGPT/Llama-3-8B labels.
# Floor experiment cho capacity threshold của encoder-only models.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python trainer.py \
  --config runs/bert-tiny/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id bert-tiny-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 32 \
  --epoch 1000 \
  --label-group-size 10 \
  --loss listMLE
