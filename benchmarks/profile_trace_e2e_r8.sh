#!/bin/bash
# profile_trace_e2e_r8.sh — End-to-end event trace cho 2 scheduler @ r=8.
#
# Mục tiêu: tìm bottleneck của opt-cpu-async-merged bằng cách so sánh CPU/GPU
# overlap timeline với opt-xxx.
#
# Chạy TUẦN TỰ trên GPU 3 (theo yêu cầu user). Mỗi scheduler 1 file trace
# riêng. Toàn bộ bench được trace (~150-300s mỗi scheduler).
#
# Output: TEMP_PROF_R8/{trace,tick_profile,server,...}_{merged,optxxx}.{csv,log,json}
#
# Sau khi chạy:
#   python analyze_trace_e2e.py --dir TEMP_PROF_R8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/TEMP_PROF_R8"
GPU=3
PORT=3444
RATE=8

mkdir -p "${OUT}"

echo "=============================================="
echo " End-to-end trace profile @ r=${RATE} (GPU ${GPU})"
echo " Output dir: ${OUT}"
echo "=============================================="

cd "${SCRIPT_DIR}"

# =============================================================================
# RUN 1: opt-cpu-async-merged1.0 + OPT-125m OpenVINO CPU
# =============================================================================
echo ""
echo "=== [1/2] opt-cpu-async-merged1.0 ==="
TRACE_EVENTS=1 \
TRACE_EVENTS_PATH="${OUT}/trace_merged.csv" \
OPT_PROFILE_TICK=1 \
OPT_PROFILE_TICK_PATH="${OUT}/tick_profile_merged.csv" \
STREAM_TIME=1 \
OPT_TIME=1 \
CUDA_VISIBLE_DEVICES=${GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "${OUT}/server_merged.log" 2>&1 &

SERVER_PID=$!
echo "Server merged PID=${SERVER_PID}, waiting 60s..."
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate ${RATE} \
    --port ${PORT} --result-dir "${OUT}"

echo "Killing server merged ${SERVER_PID}..."
sleep 5  # đợi trace flush
kill "${SERVER_PID}" 2>/dev/null || true
sleep 10
kill -9 "${SERVER_PID}" 2>/dev/null || true
sleep 30  # cooldown


# =============================================================================
# RUN 2: opt-xxx + OPT-125m AUXLLM GPU
# =============================================================================
echo ""
echo "=== [2/2] opt-xxx (paper baseline) ==="
TRACE_EVENTS=1 \
TRACE_EVENTS_PATH="${OUT}/trace_optxxx.csv" \
OPT_TIME=1 \
CUDA_VISIBLE_DEVICES=${GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-xxx \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
    > "${OUT}/server_optxxx.log" 2>&1 &

SERVER_PID=$!
echo "Server opt-xxx PID=${SERVER_PID}, waiting 120s (AUXLLM init slow)..."
sleep 120

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-xxx \
    --output-len -1 --request-rate ${RATE} \
    --port ${PORT} --result-dir "${OUT}"

echo "Killing server opt-xxx ${SERVER_PID}..."
sleep 5
kill "${SERVER_PID}" 2>/dev/null || true
sleep 10
kill -9 "${SERVER_PID}" 2>/dev/null || true


# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo " DONE. Output:"
ls -la "${OUT}/" | grep -E "\.(csv|log|json)$" | awk '{print "  ", $9, "(" $5 " bytes)"}'
echo ""
echo " Next:"
echo "   python analyze_trace_e2e.py --dir ${OUT}"
echo "=============================================="
