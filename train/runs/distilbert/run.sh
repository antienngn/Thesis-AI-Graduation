#!/usr/bin/env bash
# Train DistilBERT-base-uncased predictor trên ShareGPT/Llama-3-8B labels.
# Match baseline OPT-125M settings (batch=32, label-group-size=10, listMLE).
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

# Pin lên GPU 1. Trainer hardcode "cuda:0" → CUDA_VISIBLE_DEVICES remap GPU
# vật lý 1 thành cuda:0 visible. Override bằng cách set env var trước khi
# chạy: CUDA_VISIBLE_DEVICES=2 bash run.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python trainer.py \
  --config runs/distilbert/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id distilbert-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 16 \
  --epoch 25 \
  --label-group-size 10 \
  --loss listMLE
