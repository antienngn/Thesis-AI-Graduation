#!/bin/bash
# bench_ser_ov_async_burst.sh — Burst-pattern bench (Gamma cv=CV_VAL) cho
# opt-cpu-async-merged1.0 + opt-xxx, predictor = OPT-125m, đo predictor latency
# + trace event đầy đủ (TRACE_EVENTS + OPT_PROFILE_TICK) như TEMP_PROF_R8.
#
# Env vars profiling (giống profile_trace_e2e_r8.sh):
#   TRACE_EVENTS=1            → bật event tracer, ghi file CSV
#   TRACE_EVENTS_PATH=...     → đường dẫn trace CSV per rate
#   OPT_PROFILE_TICK=1        → bật per-tick CSV (chỉ merged dùng)
#   OPT_PROFILE_TICK_PATH=... → đường dẫn tick CSV per rate
#   STREAM_TIME=1             → log "OV-STREAM-TIME" cho async-merged
#   OPT_TIME=1                → log "OPT-TIME"        cho opt-xxx
#
# Override qua env: CV_VAL (default 16), OUT_DIR (default TEMP_RES_ASYNC_BURST_CV16),
#                   RATES (default "2 4 8 16 32 64"), GPU (default 0), PORT (default 3303)
#
# Output:
#   ${OUT_DIR}/server_<sched>_r<rate>.log
#   ${OUT_DIR}/trace_<sched>_r<rate>.csv      (TRACE_EVENTS)
#   ${OUT_DIR}/tick_profile_<sched>_r<rate>.csv (OPT_PROFILE_TICK, merged only)
#   ${OUT_DIR}/latency-*.pt , vllm-*.json     (benchmark_serving_real.py)

set -u
OV_CHUNK_SIZE=2
CV_VAL="${CV_VAL:-16}"
OUT_DIR="${OUT_DIR:-TEMP_RES_ASYNC_BURST_C2_CV${CV_VAL%.*}}"
RATES="${RATES:-2 4 8 16 32 64}"
GPU="${GPU:-0}"
PORT="${PORT:-3303}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/${OUT_DIR}"
mkdir -p "${OUT}"

echo "=============================================="
echo " Burst bench (cv=${CV_VAL}) — out: ${OUT}"
echo " GPU=${GPU}  PORT=${PORT}  rates=${RATES}"
echo "=============================================="
cd "${SCRIPT_DIR}"


run_one() {
    local SCHED="$1"     # opt-cpu-async-merged1.0 | opt-xxx
    local RATE="$2"
    local TAG="$3"       # async-merged | opt-xxx
    local CONFIG="$4"    # predictor config json path
    local WARMUP="$5"    # secs to wait after launch
    local EXTRA_ENV="$6" # additional env vars (STREAM_TIME | OPT_TIME)

    local LOG="${OUT}/server_${TAG}_r${RATE}.log"
    local TRACE="${OUT}/trace_${TAG}_r${RATE}.csv"
    local TICK="${OUT}/tick_profile_${TAG}_r${RATE}.csv"

    echo ""
    echo "--- [${TAG}] rate=${RATE} ---"

    # shellcheck disable=SC2086
    env \
        OV_CHUNK_SIZE=${OV_CHUNK_SIZE} \
        TRACE_EVENTS=1 \
        TRACE_EVENTS_PATH="${TRACE}" \
        OPT_PROFILE_TICK=1 \
        OPT_PROFILE_TICK_PATH="${TICK}" \
        ${EXTRA_ENV} \
        CUDA_VISIBLE_DEVICES=${GPU} \
        python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --swap-space 40 --disable-log-requests \
            --schedule-type "${SCHED}" \
            --enable-chunked-prefill --enforce-eager --dtype=half \
            --port "${PORT}" \
            --prefill-predictor-model-config "${CONFIG}" \
            > "${LOG}" 2>&1 &
    local SERVER_PID=$!
    echo "Server PID=${SERVER_PID}, waiting ${WARMUP}s..."
    sleep "${WARMUP}"

    python benchmark_serving_real.py --backend vllm \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
        --num-prompts -1 --request-time 60 \
        --schedule-type "${SCHED}" \
        --cv "${CV_VAL}" \
        --output-len -1 --request-rate "${RATE}" \
        --port "${PORT}" --result-dir "${OUT}"

    echo "Killing ${SERVER_PID}..."
    sleep 5
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 10
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    sleep 60
}


# =============================================================================
# OPT-125m + opt-cpu-async-merged1.0  (CPU OV streaming)
# =============================================================================
for RATE in ${RATES}; do
    run_one \
        "opt-cpu-async-merged1.0" \
        "${RATE}" \
        "async-merged" \
        "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json" \
        60 \
        "STREAM_TIME=1"
done


# =============================================================================
# OPT-125m + opt-xxx  (GPU AUXLLM sync)
# =============================================================================
for RATE in ${RATES}; do
    run_one \
        "opt-xxx" \
        "${RATE}" \
        "opt-xxx" \
        "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" \
        120 \
        "OPT_TIME=1"
done


# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo " DONE. Output in ${OUT}:"
ls -la "${OUT}/" | grep -E "\.(csv|log|json|pt)$" | awk '{printf "  %-60s %s bytes\n", $9, $5}'
echo ""
echo " Next:"
echo "   python parse_predictor_latency.py     # sửa LOG_DIR → ${OUT_DIR}"
echo "   python build_predictor_latency_xlsx.py"
echo "   python analyze_trace_e2e.py --dir ${OUT}"
echo "=============================================="
