#!/bin/bash
# profile_ov_async_r16.sh — Profile opt-cpu-async-warmup1.0 ở r=16 QPS.
#
# Variant của profile_ov_async.sh với rate=16 thay vì 8. Mục đích:
# kiểm chứng predictor có còn keep up khi rate cao gấp đôi.
# Ở r=8 đã đo: stream_queue mean 9.0, max 22 (predictor không bottleneck).
# Ở r=16: dự đoán queue depth 2-4× cao hơn → có thể chạm ngưỡng 50+ → chunk_size=4
# trở thành bottleneck thật sự, hoặc vẫn keep up nếu Stage 2 dwell ngắn hơn.
#
# Output cùng schema CSV với r=8 → so sánh được trực tiếp.

set -e

# === Config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC"
TS="$(date +%Y%m%d-%H%M%S)"
TICK_CSV="${RESULT_DIR}/tick_profile_r16_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/server_log_r16_${TS}.log"
PORT=3324  # khác port với r=8 script để tránh xung đột nếu chạy song song
SCHED="opt-cpu-async-warmup1.0"
RATE=16

mkdir -p "${RESULT_DIR}"

echo "============================================="
echo " Profile run: ${SCHED} @ r=${RATE} QPS"
echo " Tick CSV:   ${TICK_CSV}"
echo " Server log: ${SERVER_LOG}"
echo "============================================="

# === Start server ===
# CUDA_VISIBLE_DEVICES=2: cùng GPU với r=8 script (đã verified avail).
# swap-space 20: nhỏ hơn 40 để giảm lo OOM khi rate cao (xem comment trong
#                bench_ser_ov_async.sh — tại r=16 user đã set 20 trước đó).
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

# === Run benchmark (60s of requests at 16 QPS = ~960 prompts) ===
echo "Starting benchmark..."
python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type "${SCHED}" \
    --output-len -1 --request-rate "${RATE}" \
    --port "${PORT}" --result-dir "${RESULT_DIR}"

# === Cleanup ===
echo "Killing server PID ${SERVER_PID}..."
kill "${SERVER_PID}" 2>/dev/null || true
sleep 5

if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server still alive, sending SIGKILL..."
    kill -9 "${SERVER_PID}" 2>/dev/null || true
fi

# === Sanity check ===
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
    echo "   Check OPT_PROFILE_TICK env was honored, server log:"
    echo "   ${SERVER_LOG}"
fi
