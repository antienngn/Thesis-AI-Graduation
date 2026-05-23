#!/bin/bash
# Bench sweep cho LUT main model.
# Chạy 6 rate [2,4,8,16,32,64] với scheduler opt-xxx + GPU AUX-LLM predictor,
# dataset ShareGPT, 60s/bench. Mỗi bench output trace_merged.csv (có đủ field
# n_running, n_decode, n_prefill, n_tokens, lat_ms) → input cho
# build_main_model_lut.py.
#
# Cho ai đọc: phần này nên chạy SEQUENTIALLY (cùng GPU, fairness measurement).
# Total time ~30 phút (6 × 5 phút).

set -u

source /home/antn/miniconda3/etc/profile.d/conda.sh
conda activate vllm-ltr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${SCRIPT_DIR}/data"
GPU="${GPU:-0}"
PORT_BASE="${PORT_BASE:-3500}"
DURATION="${DURATION:-60}"

RATES=(2 4 8 16 32 64)

mkdir -p "${DATA_DIR}"
cd "${BENCH_DIR}"

echo "=============================================="
echo " Bench sweep — LUT main model (opt-xxx)"
echo " GPU=${GPU}  rates=${RATES[*]}  duration=${DURATION}s/bench"
echo " Output: ${DATA_DIR}/bench_r{rate}/"
echo "=============================================="

for RATE in "${RATES[@]}"; do
    OUT="${DATA_DIR}/bench_r${RATE}"
    PORT=$((PORT_BASE + RATE))
    mkdir -p "${OUT}"

    echo ""
    echo "──────────────────────────────────────────────"
    echo " rate=${RATE}  port=${PORT}  out=${OUT}"
    echo "──────────────────────────────────────────────"

    export TRACE_EVENTS=1
    export TRACE_EVENTS_PATH="${OUT}/trace_merged.csv"
    export CUDA_VISIBLE_DEVICES=${GPU}

    # Launch server với opt-xxx + GPU PyTorch predictor
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --swap-space 40 --disable-log-requests \
        --schedule-type opt-xxx \
        --enable-chunked-prefill --enforce-eager --dtype=half \
        --port ${PORT} \
        --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
        > "${OUT}/server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID=${SERVER_PID}, warmup 60s..."
    sleep 60

    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ERROR: server died"
        tail -30 "${OUT}/server.log"
        continue
    fi

    # Run bench
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

    # Tail wait để các request cuối kịp finish
    echo "  Bench done. Tail wait 60s..."
    sleep 60

    # Graceful shutdown
    if kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  Shutting down server..."
        kill -INT ${SERVER_PID} 2>/dev/null || true
        sleep 15
        kill -TERM ${SERVER_PID} 2>/dev/null || true
        sleep 5
        kill -KILL ${SERVER_PID} 2>/dev/null || true
    fi
    sleep 5

    # Check output
    if [ -f "${OUT}/trace_merged.csv" ]; then
        N_LINES=$(wc -l < "${OUT}/trace_merged.csv")
        N_ME=$(grep -c "model_executor.start" "${OUT}/trace_merged.csv" || echo 0)
        echo "  ✓ trace: ${N_LINES} lines, ${N_ME} model_executor events"
    else
        echo "  ✗ no trace file"
    fi
done

echo ""
echo "=============================================="
echo " Sweep done. Per-rate summary:"
echo "=============================================="
for RATE in "${RATES[@]}"; do
    OUT="${DATA_DIR}/bench_r${RATE}"
    if [ -f "${OUT}/trace_merged.csv" ]; then
        N_ME=$(grep -c "model_executor.start" "${OUT}/trace_merged.csv" || echo 0)
        echo "  r=${RATE}: ${N_ME} model_executor events"
    else
        echo "  r=${RATE}: MISSING"
    fi
done
echo ""
echo "Now run: python build_main_model_lut.py"
