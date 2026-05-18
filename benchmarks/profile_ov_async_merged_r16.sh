#!/bin/bash
# profile_ov_async_merged_r16.sh — Profile opt-cpu-async-merged1.0 ở r=16 QPS.
#
# Variant của profile_ov_async_merged.sh với rate=16. Mục đích kiểm chứng
# fix nới C2 ở rate cao — predictor có thể vẫn lag (chunk_size=4) nhưng
# Stage 2 dwell wall phải biến mất.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC_MERGE"
TS="$(date +%Y%m%d-%H%M%S)"
TICK_CSV="${RESULT_DIR}/tick_profile_merged_r16_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/server_log_merged_r16_${TS}.log"
PORT=3326
SCHED="opt-cpu-async-merged1.0"
RATE=16

mkdir -p "${RESULT_DIR}"

echo "============================================="
echo " Profile run: ${SCHED} @ r=${RATE} QPS"
echo " Tick CSV:   ${TICK_CSV}"
echo " Server log: ${SERVER_LOG}"
echo "============================================="

cd "${SCRIPT_DIR}"
OPT_PROFILE_TICK=1 \
OPT_PROFILE_TICK_PATH="${TICK_CSV}" \
OPT_TIME=1 \
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 20 --disable-log-requests \
    --schedule-type "${SCHED}" \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port "${PORT}" \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}, waiting 60s for warmup..."
sleep 60

echo "Starting benchmark..."
python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type "${SCHED}" \
    --output-len -1 --request-rate "${RATE}" \
    --port "${PORT}" --result-dir "${RESULT_DIR}"

echo "Killing server PID ${SERVER_PID}..."
kill "${SERVER_PID}" 2>/dev/null || true
sleep 5
if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -9 "${SERVER_PID}" 2>/dev/null || true
fi

echo ""
echo "============================================="
echo " Profile run done."
echo "============================================="
if [[ -f "${TICK_CSV}" ]]; then
    LINES=$(wc -l < "${TICK_CSV}")
    SIZE=$(du -h "${TICK_CSV}" | cut -f1)
    echo " Tick CSV: ${TICK_CSV}"
    echo "   ${LINES} rows, ${SIZE}"
    head -3 "${TICK_CSV}" | sed 's/^/     /'
    tail -3 "${TICK_CSV}" | sed 's/^/     /'
else
    echo " WARNING: Tick CSV not found at ${TICK_CSV}"
fi

echo ""
echo " Next: python plot_tick_profile.py ${TICK_CSV}"
