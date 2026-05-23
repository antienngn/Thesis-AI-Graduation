#!/bin/bash
# Sweep bench scheduler opt-xxx (paper baseline) với rate=[2,4,8,16,32,64]
# Output: SERVE_OPTXXX/r{rate}/
#
# Mirror chính xác bench_all_dual.sh để fairness compare:
#   - Cùng dataset, seed, output_len=-1, ignore_eos, sampling seed=42
#   - Cùng server config (model, chunked_prefill, dtype, swap_space, ...)
#   - Khác biệt duy nhất: schedule_type=opt-xxx + GPU PyTorch predictor

set -u

source /home/antn/miniconda3/etc/profile.d/conda.sh
conda activate vllm-ltr
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-3600}"
DURATION="${DURATION:-60}"
RATES="${RATES:-2 4 8 16 32 64}"
OUT_ROOT="${SCRIPT_DIR}/SERVE_OPTXXX"

mkdir -p "${OUT_ROOT}"
cd "${SCRIPT_DIR}"

SUMMARY="${OUT_ROOT}/summary.log"
echo "" > "${SUMMARY}"

echo "=============================================="
echo " Bench sweep opt-xxx — rates=[${RATES}]"
echo " GPU=${GPU}, duration=${DURATION}s/rate"
echo " Output: ${OUT_ROOT}/"
echo "=============================================="

for RATE in ${RATES}; do
    OUT="${OUT_ROOT}/r${RATE}"
    PORT=$((PORT_BASE + RATE))
    mkdir -p "${OUT}"

    echo ""
    echo "──────────────────────────────────────────────"
    echo " rate=${RATE}  port=${PORT}  out=${OUT}"
    echo "──────────────────────────────────────────────"

    export TRACE_EVENTS=1
    export TRACE_EVENTS_PATH="${OUT}/trace_merged.csv"
    export OPT_PROFILE_TICK=1
    export OPT_PROFILE_TICK_PATH="${OUT}/tick_profile_merged.csv"
    export STREAM_TIME=1
    export CUDA_VISIBLE_DEVICES=${GPU}

    # Launch server — opt-xxx + GPU PyTorch predictor (usage_config.json)
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --swap-space 40 --disable-log-requests \
        --schedule-type opt-xxx \
        --enable-chunked-prefill --enforce-eager --dtype=half \
        --port ${PORT} \
        --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
        > "${OUT}/server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID=${SERVER_PID}, warmup 90s..."
    sleep 90

    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ERROR: server died for rate=${RATE}"
        tail -40 "${OUT}/server.log"
        echo "rate=${RATE}: SERVER DIED" >> "${SUMMARY}"
        continue
    fi

    # Bench — cùng tham số fairness với bench_all_dual.sh
    echo "  Running bench rate=${RATE}, t=${DURATION}s..."
    python benchmark_serving_real.py --backend vllm \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
        --num-prompts -1 --request-time ${DURATION} \
        --schedule-type opt-xxx \
        --output-len -1 --request-rate ${RATE} \
        --port ${PORT} --result-dir "${OUT}" \
        > "${OUT}/bench.log" 2>&1

    echo "  Bench done. Tail 90s..."
    sleep 90

    # Graceful shutdown
    if kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  Shutdown server..."
        kill -INT ${SERVER_PID} 2>/dev/null || true
        sleep 15
        kill -TERM ${SERVER_PID} 2>/dev/null || true
        sleep 5
        kill -KILL ${SERVER_PID} 2>/dev/null || true
    fi
    sleep 5

    # Per-rate summary
    if [ -f "${OUT}/bench.log" ]; then
        echo "  --- Bench result rate=${RATE} ---"
        grep -E "Successful requests|Benchmark duration|Total generated|Total input|throughput|TTFT|TPOT|Nlatency|Kendall" "${OUT}/bench.log" | head -25 | tee -a "${SUMMARY}"
        echo "rate=${RATE}: DONE" >> "${SUMMARY}"
    fi
    echo "" >> "${SUMMARY}"
done

echo ""
echo "=============================================="
echo " SWEEP DONE"
echo "=============================================="
echo "Per-rate output: ${OUT_ROOT}/r{${RATES// /,}}/"
echo "Summary log:     ${SUMMARY}"
echo ""
echo "--- Summary ---"
cat "${SUMMARY}"
