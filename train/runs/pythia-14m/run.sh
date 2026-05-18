#!/usr/bin/env bash
# Train Pythia-14m-deduped predictor trên ShareGPT/Llama-3-8B labels.
# Match baseline OPT-125M settings (batch=32, label-group-size=10, listMLE).
# Smallest candidate — floor experiment cho capacity threshold.
set -euo pipefail

cd "$(dirname "$0")/../.."  # → train/

# Pin lên GPU 1. Trainer hardcode "cuda:0" → CUDA_VISIBLE_DEVICES remap GPU
# vật lý 1 thành cuda:0 visible. Override bằng cách set env var trước khi
# chạy: CUDA_VISIBLE_DEVICES=2 bash run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python trainer.py \
  --config runs/pythia-14m/config.txt \
  --file ../benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
  --job-dir MODEL \
  --run-id pythia-14m-llama3-8b-sharegpt-score-trainbucket10-b32 \
  --batch-size 32 \
  --epoch 500\
  --label-group-size 10 \
  --loss listMLE

#Best is epoch 50, Tau=0.44