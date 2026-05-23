#!/bin/bash
# Bench scheduler dual<T> với nsys profile + NVTX markers.
# Yêu cầu: 2 LUT đã build (benchmarks/LUT_CREATE/data/*.json)

set -u

source /home/antn/miniconda3/etc/profile.d/conda.sh
conda activate vllm-ltr
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-1}"
PORT="${PORT:-3460}"
RATE="${RATE:-8}"
T_WARMUP="${T_WARMUP:-1.0}"
DURATION="${DURATION:-60}"
OUT="${SCRIPT_DIR}/TEMP_PROF_DUAL_R${RATE}_NSIGHT"

NSYS="/home/hieuvt/vllm-hpclab/tools/nsight-systems-cli/opt/nvidia/nsight-systems-cli/2025.6.1/bin/nsys"

# Kiểm tra LUT đã build chưa
LUT_CPU="${SCRIPT_DIR}/LUT_CREATE/data/cpu_predictor_lut.json"
LUT_MAIN="${SCRIPT_DIR}/LUT_CREATE/data/main_model_lut.json"
if [ ! -f "${LUT_CPU}" ] || [ ! -f "${LUT_MAIN}" ]; then
    echo "ERROR: LUT chưa build. Chạy:"
    echo "  cd LUT_CREATE && bash run_all_tmux.sh"
    exit 1
fi
echo "LUT files OK:"
echo "  CPU:  $(du -h ${LUT_CPU} | cut -f1)"
echo "  MAIN: $(du -h ${LUT_MAIN} | cut -f1)"

mkdir -p "${OUT}"
cd "${SCRIPT_DIR}"

echo "=============================================="
echo " nsys profile — scheduler dual${T_WARMUP}, rate=${RATE}"
echo " Output: ${OUT}  GPU=${GPU}  PORT=${PORT}  duration=${DURATION}s"
echo "=============================================="

# NVTX markers + event tracer
export PYTHONPATH="${SCRIPT_DIR}/SYSTEM_PROBE/sitecustomize_patch_nvtx_full:${PYTHONPATH:-}"
export TRACE_EVENTS=1
export TRACE_EVENTS_PATH="${OUT}/trace_merged.csv"
export OPT_PROFILE_TICK=1
export OPT_PROFILE_TICK_PATH="${OUT}/tick_profile_merged.csv"
export STREAM_TIME=1
export CUDA_VISIBLE_DEVICES=${GPU}

# Launch server với nsys wrapper + scheduler dual<T> + config dual
echo ""
echo "--- launch nsys + server (dual${T_WARMUP}) ---"
${NSYS} profile \
    --output="${OUT}/trace_merged" \
    --force-overwrite=true \
    --trace=cuda,cudnn,cublas,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --python-sampling=false \
    --stats=true \
    --export=sqlite \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --swap-space 40 --disable-log-requests \
        --schedule-type dual${T_WARMUP} \
        --enable-chunked-prefill --enforce-eager --dtype=half \
        --port ${PORT} \
        --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_dual.json \
        > "${OUT}/server.log" 2>&1 &
NSYS_PID=$!
echo "Nsys wrapper PID=${NSYS_PID}, warmup 90s..."
sleep 90

if ! kill -0 ${NSYS_PID} 2>/dev/null; then
    echo "ERROR: nsys died during warmup"
    tail -40 "${OUT}/server.log"
    exit 1
fi

# Find python server PID for graceful shutdown (nsys is grandparent)
SERVER_PID=$(ps -ef | awk -v port=${PORT} '
    /vllm.entrypoints.openai.api_server/ && /'${SCRIPT_DIR}'/ && !/nsys profile/ && !/awk/ {
        for (i=8; i<=NF; i++) if ($i ~ /--port/) { print $2; exit }
    }')
if [ -z "$SERVER_PID" ]; then
    SERVER_PID=$(pgrep -f "miniconda.*python.*vllm.entrypoints" | head -1)
fi
echo "Python server PID=${SERVER_PID}"

# Run bench
echo ""
echo "--- run bench rate=${RATE}, t=${DURATION}s ---"
python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time ${DURATION} \
    --schedule-type dual${T_WARMUP} \
    --output-len -1 --request-rate ${RATE} \
    --port ${PORT} --result-dir "${OUT}" \
    > "${OUT}/bench.log" 2>&1

echo "Bench done. Tail 180s for tail processing..."
sleep 30

if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
    echo "--- SIGINT server PID=$SERVER_PID for graceful shutdown ---"
    kill -INT $SERVER_PID
    sleep 20
    kill -TERM $SERVER_PID 2>/dev/null || true
    sleep 10
    kill -KILL $SERVER_PID 2>/dev/null || true
fi

echo "Waiting nsys to finalize..."
wait ${NSYS_PID} 2>/dev/null
echo "Nsys exited."

echo ""
echo "=============================================="
NSYS_REP="${OUT}/trace_merged.nsys-rep"
if [ -f "${NSYS_REP}" ]; then
    echo "✓ SUCCESS: ${NSYS_REP} ($(du -h $NSYS_REP | cut -f1))"
else
    echo "✗ FAIL: no .nsys-rep"
fi
ls -lh "${OUT}/" | grep -E "\.(nsys-rep|sqlite|csv|log|json|pt)$"
echo "=============================================="

# Verify router behavior từ trace
echo ""
echo "--- Router decision stats từ trace ---"
if [ -f "${OUT}/trace_merged.csv" ]; then
    N_ROUTER=$(grep -c "dual.router.end" "${OUT}/trace_merged.csv" || echo 0)
    N_CPU_SUBMIT=$(grep -c "predictor.submit.start.*dual_cpu" "${OUT}/trace_merged.csv" || echo 0)
    N_GPU_SYNC=$(grep -c "predictor.gpu_sync.start" "${OUT}/trace_merged.csv" || echo 0)
    N_OV_FWD=$(grep -c "predictor.worker.forward.end" "${OUT}/trace_merged.csv" || echo 0)
    N_ME=$(grep -c "model_executor.start" "${OUT}/trace_merged.csv" || echo 0)
    echo "  ${N_ROUTER} router decisions"
    echo "  ${N_CPU_SUBMIT} CPU submit calls"
    echo "  ${N_GPU_SYNC} GPU sync score calls"
    echo "  ${N_OV_FWD} OV inference (worker.forward) calls"
    echo "  ${N_ME} model_executor calls"
fi
echo "=============================================="
