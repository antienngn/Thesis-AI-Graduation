#!/bin/bash
# profile_ov_async.sh — Profile GPU slot occupancy + predictor backlog
#                       cho opt-cpu-async-warmup1.0 ở r=8 QPS.
#
# Mục đích:
#   Trả lời câu hỏi: "Trong khoảng Stage 2 dwell, GPU có bao nhiêu KV slot
#   trống và bao nhiêu post-warmup đã scored sẵn để admit?"
#   Câu trả lời quyết định fix là 'nới C2' (đơn giản) hay 'redesign KV
#   admission policy' (phức tạp). Xem cây quyết định trong commit message
#   hoặc plan trong conversation history.
#
# Instrumentation gate:
#   OPT_PROFILE_TICK=1            bật tick profiler trong scheduler
#   OPT_PROFILE_TICK_PATH=<path>  output CSV path (default /tmp/...)
#   OPT_TIME=1                    bật predictor activity log (đã có sẵn)
#
# Output:
#   - TEMP_RES_ASYNC/tick_profile_r8_<timestamp>.csv  (per-tick CSV)
#   - TEMP_RES_ASYNC/server_log_r8_<timestamp>.log    (server stdout/stderr,
#                                                      bao gồm OPT_TIME logs)
#   - TEMP_RES_ASYNC/vllm-8.0qps-...-<timestamp>.json (benchmark output,
#                                                      do benchmark_serving_real
#                                                      ghi)

set -e

# === Config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC"
TS="$(date +%Y%m%d-%H%M%S)"
TICK_CSV="${RESULT_DIR}/tick_profile_r8_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/server_log_r8_${TS}.log"
PORT=3323
SCHED="opt-cpu-async-warmup1.0"
RATE=8

mkdir -p "${RESULT_DIR}"

echo "============================================="
echo " Profile run: ${SCHED} @ r=${RATE} QPS"
echo " Tick CSV:   ${TICK_CSV}"
echo " Server log: ${SERVER_LOG}"
echo "============================================="

# === Start server với profiling env vars ===
# OPT_PROFILE_TICK=1: bật tick logger trong _profile_tick_async_warmup
# OPT_PROFILE_TICK_PATH: chỉ định path CSV (absolute để tránh CWD issue)
# OPT_TIME=1: bật log "OPT-CPU-ASYNC-SUBMIT" và "OV-STREAM" — diagnostic
#             cho predictor activity, đối chiếu với CSV
cd "${SCRIPT_DIR}"
OPT_PROFILE_TICK=1 \
OPT_PROFILE_TICK_PATH="${TICK_CSV}" \
OPT_TIME=1 \
CUDA_VISIBLE_DEVICES=2 \
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

# === Run benchmark (60s of requests at 8 QPS) ===
# Cùng cấu hình với bench_ser_ov_async.sh để kết quả so sánh được với
# các JSON đã có trong TEMP_RES_ASYNC.
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

# Nếu server vẫn alive (vì child processes), force kill
if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server still alive, sending SIGKILL..."
    kill -9 "${SERVER_PID}" 2>/dev/null || true
fi

# === Sanity check output ===
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

echo ""
echo " Next steps:"
echo "   1. Plot KV occupancy timeline:"
echo "      python -c \"import pandas as pd; df=pd.read_csv('${TICK_CSV}'); print(df.describe())\""
echo "   2. Cây quyết định cải tiến — xem 5 metrics chính:"
echo "      a. Stage 2 dwell time:       max(t_rel where stage==2) - warmup_seconds"
echo "      b. Mean free KV % Stage 2:   mean(n_free / n_total where stage==2)"
echo "      c. Max scored backlog S2:    max(n_postwarmup_waiting_scored where stage==2)"
echo "      d. Predictor queue cuối S2:  n at last stage==2 row"
echo "      e. n_free vs n_postwarmup_scored cross-correlation tại Stage 2"
