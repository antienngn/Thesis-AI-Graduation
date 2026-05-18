#!/bin/bash
# profile_trace_optxxx_r8.sh — Rerun opt-xxx scheduler @ r=8 với TRACE_EVENTS.
#
# Vì sao tách script riêng: bản gốc profile_trace_e2e_r8.sh chạy cả 2 scheduler,
# nhưng lần đầu opt-xxx tạo trace rỗng (gate warmup_seconds>0 trong
# Scheduler.add_seq_group bỏ qua opt-xxx). Sau khi fix gate, chỉ cần rerun
# opt-xxx, giữ nguyên trace_merged.csv + tick_profile_merged.csv đã có.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/TEMP_PROF_R8"
GPU=3
PORT=3444
RATE=8

mkdir -p "${OUT}"

echo "=============================================="
echo " opt-xxx trace rerun @ r=${RATE} (GPU ${GPU})"
echo " Output dir: ${OUT}"
echo "=============================================="

cd "${SCRIPT_DIR}"

TRACE_EVENTS=1 \
TRACE_EVENTS_PATH="${OUT}/trace_optxxx.csv" \
OPT_TIME=1 \
CUDA_VISIBLE_DEVICES=${GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-xxx \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
    > "${OUT}/server_optxxx.log" 2>&1 &

SERVER_PID=$!
echo "Server opt-xxx PID=${SERVER_PID}, waiting 120s (AUXLLM init slow)..."
sleep 120

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-xxx \
    --output-len -1 --request-rate ${RATE} \
    --port ${PORT} --result-dir "${OUT}"

echo "Killing server opt-xxx ${SERVER_PID}..."
sleep 5   # đợi trace flush
kill "${SERVER_PID}" 2>/dev/null || true
sleep 10
kill -9 "${SERVER_PID}" 2>/dev/null || true

echo ""
echo "=============================================="
echo " DONE. trace_optxxx.csv size:"
ls -la "${OUT}/trace_optxxx.csv"
echo ""
echo " Expected: hàng nghìn rows (scheduler.tick.* + OPT-TIME + model_executor.* events)"
echo "=============================================="
