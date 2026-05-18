#!/bin/bash
# bench_predictor.sh — đo OV CPU và GPU trong 2 PROCESS RIÊNG để cô lập hoàn toàn.
# Mỗi process load đúng 1 backend với config tương ứng, đo xong exit.
set -e

MODEL_DIR=MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32
PYTHIA_MODEL_DIR=MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32
DATASET=llama3-8b-sharegpt-test-t1-s0-8192.jsonl
LLAMA_TOK=meta-llama/Meta-Llama-3-8B-Instruct
OUT_DIR=BENCH_PRED_RES
GPU_ID=1
NUM_PROMPTS=500
BATCH_SIZE=1
N_ITERS=10
WARMUP=6

mkdir -p $OUT_DIR

# echo "=== [1/2] OpenVINO CPU predictor (process 1 - int 8) ==="
# python bench_predictor.py \
#   --backend openvino \
#   --config $MODEL_DIR/usage_config_ov.json \
#   --llama-tokenizer $LLAMA_TOK \
#   --dataset $DATASET \
#   --num-prompts $NUM_PROMPTS \
#   --batch-size $BATCH_SIZE \
#   --n-iters $N_ITERS \
#   --warmup $WARMUP \
#   --output $OUT_DIR/openvino.json

echo "=== [1/2] OpenVINO CPU predictor (process - f16) ==="
python bench_predictor.py \
  --backend openvino \
  --config $PYTHIA_MODEL_DIR/usage_config_ov.json \
  --llama-tokenizer $LLAMA_TOK \
  --dataset $DATASET \
  --num-prompts $NUM_PROMPTS \
  --batch-size $BATCH_SIZE \
  --n-iters $N_ITERS \
  --warmup $WARMUP \
  --ov-precision f16\
  --output $OUT_DIR/pythia-openvino.json

echo
echo "=== [2/2] PyTorch GPU predictor (process 2) ==="
python bench_predictor.py \
  --backend gpu \
  --config $MODEL_DIR/usage_config.json \
  --llama-tokenizer $LLAMA_TOK \
  --dataset $DATASET \
  --num-prompts $NUM_PROMPTS \
  --batch-size $BATCH_SIZE \
  --n-iters $N_ITERS \
  --warmup $WARMUP \
  --gpu-id $GPU_ID \
  --gpu-dtype half \
  --output $OUT_DIR/opt_gpu.json

echo
echo "=== Summary ==="
python -c "
import json
ov  = json.load(open('$OUT_DIR/pythia-openvino.json'))
gpu = json.load(open('$OUT_DIR/opt_gpu.json'))
print(f\"{'backend':<10} {'inference_ms':>15} {'kendall_tau':>15}\")
print('-' * 42)
for r in (ov, gpu):
    print(f\"{r['backend']:<10} {r['inference_time_ms']:>15.2f} {r['kendall_tau']:>15.4f}\")
"
