#!/bin/bash
# Bench dual<T> rate=64 với DEBUG TIMING logs để tìm bug hang/crash.
# Output: SERVE_DUAL_R64_DEBUG/
#
# Bật DUAL_DEBUG_TIMING=1 → scheduler in per-phase ms mỗi 100 tick HOẶC
# khi tick > 100ms (chậm bất thường). Logs ra stdout server.log.

set -u

source /home/antn/miniconda3/etc/profile.d/conda.sh
conda activate vllm-ltr
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-1}"
PORT="${PORT:-3700}"
RATE="${RATE:-64}"
T_WARMUP="${T_WARMUP:-1.0}"
DURATION="${DURATION:-60}"
OUT="${SCRIPT_DIR}/SERVE_DUAL_R64_DEBUG"

# Kiểm tra LUT
LUT_CPU="${SCRIPT_DIR}/LUT_CREATE/data/cpu_predictor_lut.json"
LUT_MAIN="${SCRIPT_DIR}/LUT_CREATE/data/main_model_lut.json"
if [ ! -f "${LUT_CPU}" ] || [ ! -f "${LUT_MAIN}" ]; then
    echo "ERROR: LUT chưa build."
    exit 1
fi

mkdir -p "${OUT}"
cd "${SCRIPT_DIR}"

echo "=============================================="
echo " DEBUG bench dual${T_WARMUP} rate=${RATE}"
echo " GPU=${GPU}, duration=${DURATION}s"
echo " Output: ${OUT}/"
echo "=============================================="

# Tracing + debug timing
export TRACE_EVENTS=1
export TRACE_EVENTS_PATH="${OUT}/trace_merged.csv"
export OPT_PROFILE_TICK=1
export OPT_PROFILE_TICK_PATH="${OUT}/tick_profile_merged.csv"
export STREAM_TIME=1
export CUDA_VISIBLE_DEVICES=${GPU}

# === DEBUG: bật timing logs cho dual scheduler ===
export DUAL_DEBUG_TIMING=1
export DUAL_DEBUG_INTERVAL=50   # log mỗi 50 tick + mọi tick chậm

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type dual${T_WARMUP} \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_dual.json \
    > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID=${SERVER_PID}, warmup 90s..."
sleep 90

if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: server died"
    tail -40 "${OUT}/server.log"
    exit 1
fi

# Bench
echo "Running bench rate=${RATE}, t=${DURATION}s..."
python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time ${DURATION} \
    --schedule-type dual${T_WARMUP} \
    --output-len -1 --request-rate ${RATE} \
    --port ${PORT} --result-dir "${OUT}" \
    > "${OUT}/bench.log" 2>&1

echo "Bench done. Wait 30s before shutdown..."
sleep 30

# Shutdown
if kill -0 ${SERVER_PID} 2>/dev/null; then
    kill -INT ${SERVER_PID} 2>/dev/null || true
    sleep 15
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 5
    kill -KILL ${SERVER_PID} 2>/dev/null || true
fi

echo ""
echo "=============================================="
echo " Output files:"
ls -lh "${OUT}/"
echo "=============================================="

echo ""
echo "=== Bench result ==="
grep -E "Successful requests|Benchmark duration|Total generated|Total input|throughput|Mean TTFT|P99 TTFT" "${OUT}/bench.log" 2>/dev/null | head -10

echo ""
echo "=== Slow ticks (>100ms) ==="
grep "DUAL_DBG" "${OUT}/server.log" 2>/dev/null | awk -F'total=' '{print $2}' | awk -F'ms' '{if ($1+0 > 100) print "  ", $0}' | head -20

echo ""
echo "=== Top 10 slowest ticks ==="
grep "DUAL_DBG" "${OUT}/server.log" 2>/dev/null | sort -t= -k3 -rn | head -10

echo ""
echo "=== Trace event count ==="
if [ -f "${OUT}/trace_merged.csv" ]; then
    grep -c "scheduler.tick.end" "${OUT}/trace_merged.csv"
    echo "  (scheduler ticks)"
    grep -c "dual.router.end" "${OUT}/trace_merged.csv"
    echo "  (router decisions)"
fi
