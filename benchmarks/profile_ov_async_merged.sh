#!/bin/bash
# profile_ov_async_merged.sh — Profile opt-cpu-async-merged1.0 ở r=8 QPS.
#
# Variant của profile_ov_async.sh dùng schedule_type 'opt-cpu-async-merged1.0'
# (bỏ Stage 2 quarantine) thay vì 'opt-cpu-async-warmup1.0'.
#
# Mục đích so sánh:
#   - async-warmup cũ ở r=8: Stage 2 dwell 15.7s, mean TTFT 31.3s, cliff
#     index 8.
#   - async-merged mới ở r=8 dự đoán: Stage 2 dwell ~0, mean TTFT 2-5s,
#     no cliff (TTFT smooth).
#
# Schema CSV identical → reuse plot_tick_profile.py không cần sửa.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC_MERGE"
TS="$(date +%Y%m%d-%H%M%S)"
TICK_CSV="${RESULT_DIR}/tick_profile_merged_r8_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/server_log_merged_r8_${TS}.log"
PORT=3325  # khác port với async-warmup r8 (3303) để chạy song song được nếu cần
SCHED="opt-cpu-async-merged1.0"
RATE=8

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
CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
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
    echo "Server still alive, sending SIGKILL..."
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
    echo "   First 3 rows:"
    head -3 "${TICK_CSV}" | sed 's/^/     /'
    echo "   Last 3 rows:"
    tail -3 "${TICK_CSV}" | sed 's/^/     /'
else
    echo " WARNING: Tick CSV not found at ${TICK_CSV}"
fi

echo ""
echo " Next: python plot_tick_profile.py ${TICK_CSV}"
