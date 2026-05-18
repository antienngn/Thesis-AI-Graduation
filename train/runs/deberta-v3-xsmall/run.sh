#!/usr/bin/env bash
# Train DeBERTa-v3-xsmall (microsoft/deberta-v3-xsmall, ~22M) predictor.
# DeBERTa thường strong-per-param hơn BERT cùng size cho classification/ranking.
# Yêu cầu `sentencepiece` package.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" python trainer.py \
  --config runs/deberta-v3-xsmall/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id deberta-v3-xsmall-llama3-8b-sharegpt-score-trainbucket10-b16 \
  --batch-size 32 \
  --epoch 500 \
  --label-group-size 10 \
  --loss listMLE
