#!/bin/bash
# benchmark_dual.sh — Bench scheduler dual<T> với output FLAT (chỉ JSON + .pt)
# để so sánh fair với SERVE_RES baselines.
#
# Khác bench_all_dual.sh:
#   - Output flat (cùng dir cho mọi rate), KHÔNG subdir r{rate}/
#   - KHÔNG bật TRACE_EVENTS / OPT_PROFILE_TICK / STREAM_TIME (zero overhead)
#   - server.log + bench.log đi vào /tmp/ (ngoài result dir)
#   - Result dir chỉ chứa vllm-*.json + latency-*.pt (giống SERVE_RES)

set -u

source /home/antn/miniconda3/etc/profile.d/conda.sh
conda activate vllm-ltr
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
PORT_BASE="${PORT_BASE:-3700}"
T_WARMUP="${T_WARMUP:-1.0}"
DURATION="${DURATION:-60}"
RATES="${RATES:-2 4 8 16}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/SERVE_DUAL_FAIR}"
LOG_DIR="${LOG_DIR:-/tmp/benchmark_dual_logs}"

# Sanity check: LUT
LUT_CPU="${SCRIPT_DIR}/LUT_CREATE/data/cpu_predictor_lut.json"
LUT_MAIN="${SCRIPT_DIR}/LUT_CREATE/data/main_model_lut.json"
if [ ! -f "${LUT_CPU}" ] || [ ! -f "${LUT_MAIN}" ]; then
    echo "ERROR: LUT chưa build. Chạy: cd LUT_CREATE && bash run_all_tmux.sh"
    exit 1
fi
echo "LUT files OK: CPU=$(du -h ${LUT_CPU} | cut -f1) MAIN=$(du -h ${LUT_MAIN} | cut -f1)"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${SCRIPT_DIR}"

echo "=============================================="
echo " benchmark_dual.sh — scheduler dual${T_WARMUP}"
echo " rates=[${RATES}]  GPU=${GPU}  duration=${DURATION}s/rate"
echo " Result dir (flat):  ${OUT_DIR}"
echo " Logs (outside):     ${LOG_DIR}"
echo " Profiling env:      DISABLED (fair với SERVE_RES)"
echo "=============================================="

for RATE in ${RATES}; do
    PORT=$((PORT_BASE + RATE))
    SERVER_LOG="${LOG_DIR}/server_dual${T_WARMUP}_r${RATE}.log"
    BENCH_LOG="${LOG_DIR}/bench_dual${T_WARMUP}_r${RATE}.log"

    echo ""
    echo "──────────────────────────────────────────────"
    echo " rate=${RATE}  port=${PORT}"
    echo " server log: ${SERVER_LOG}"
    echo "──────────────────────────────────────────────"

    # Launch server — NO profiling env vars (chỉ CUDA_VISIBLE_DEVICES)
    env -i \
        HOME="${HOME}" PATH="${PATH}" CONDA_PREFIX="${CONDA_PREFIX:-}" \
        CUDA_VISIBLE_DEVICES=${GPU} \
        python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --swap-space 40 --disable-log-requests \
            --schedule-type dual${T_WARMUP} \
            --enable-chunked-prefill --enforce-eager --dtype=half \
            --port ${PORT} \
            --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_dual.json \
            > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID=${SERVER_PID}, warmup 90s..."
    sleep 90

    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ERROR: server died for rate=${RATE}"
        tail -40 "${SERVER_LOG}"
        continue
    fi

    # Bench — cùng tham số fairness với baseline SERVE_RES
    echo "  Running bench rate=${RATE}, t=${DURATION}s..."
    python benchmark_serving_real.py --backend vllm \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
        --num-prompts -1 --request-time ${DURATION} \
        --schedule-type dual${T_WARMUP} \
        --output-len -1 --request-rate ${RATE} \
        --port ${PORT} --result-dir "${OUT_DIR}" \
        > "${BENCH_LOG}" 2>&1

    echo "  Bench done. Tail 90s..."
    sleep 90

    # Graceful shutdown
    if kill -0 ${SERVER_PID} 2>/dev/null; then
        kill -INT ${SERVER_PID} 2>/dev/null || true
        sleep 15
        kill -TERM ${SERVER_PID} 2>/dev/null || true
        sleep 5
        kill -KILL ${SERVER_PID} 2>/dev/null || true
    fi
    sleep 5

    # Quick echo of result
    if [ -f "${BENCH_LOG}" ]; then
        echo "  --- result rate=${RATE} ---"
        grep -E "Successful requests|Benchmark duration|Total generated|Total input|throughput|Mean TTFT|Mean TPOT|Mean Nlatency" "${BENCH_LOG}" | head -15
    fi
done

echo ""
echo "=============================================="
echo " DONE."
echo " Result files (flat, giống SERVE_RES):"
ls -la "${OUT_DIR}/" | grep -E "\.(json|pt)$" | awk '{printf "  %s  %s\n", $5, $9}'
echo ""
echo " Logs (riêng):  ${LOG_DIR}/"
echo "=============================================="
