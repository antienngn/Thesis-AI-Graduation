#!/bin/bash
# bench_ser_ov_async_merged_T_sweep.sh — Sweep warmup_seconds T cho
# opt-cpu-async-merged ở rate cố định.
#
# Mục đích:
#   Đo ảnh hưởng của warmup_seconds T tới performance của variant merged.
#   Ở opt-cpu-async-warmup cũ T cao = drain dwell dài = TTFT tệ. Variant
#   merged được thiết kế ĐỂ T không còn ảnh hưởng tail (vì không quarantine).
#   Sweep này confirm assumption đó.
#
# T=0:   không warmup — pure SJF từ đầu (cần predictor work ngay từ
#        request đầu tiên — có thể tail TTFT tăng vì cold-start predictor).
# T=0.5: warmup ngắn — vài request đầu FCFS, sau đó SJF.
# T=1.0: warmup mặc định (cùng với async-warmup baseline).
# T=2.0: warmup dài — kiểm chứng rằng T cao KHÔNG còn gây harm như async-warmup.
#
# Workload: r=8 QPS (sweet spot — đủ load để thấy hiệu ứng, đủ nhanh để
# chạy nhiều variant). Nếu muốn rate khác, sửa biến RATE bên dưới.
#
# Output:
#   TEMP_RES_ASYNC/vllm-8.0qps-...-opt-cpu-async-merged{0.0,0.5,1.0,2.0}-<TS>.json
#
# ETA: 4 variant × ~6 phút/variant ≈ 25-30 phút.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC"
PORT=3328
RATE=8

mkdir -p "${RESULT_DIR}"
cd "${SCRIPT_DIR}"

run_T() {
    local T=$1
    local SCHED="opt-cpu-async-merged${T}"
    echo ""
    echo "============================================="
    echo " T sweep: ${SCHED} @ r=${RATE} QPS"
    echo "============================================="

    OPT_TIME=1 \
    CUDA_VISIBLE_DEVICES=0 \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --swap-space 40 --disable-log-requests \
        --schedule-type "${SCHED}" \
        --enable-chunked-prefill --enforce-eager --dtype=half \
        --port "${PORT}" \
        --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
        > "/tmp/server_${SCHED}.log" 2>&1 &

    local PID=$!
    echo "Server PID: ${PID}, waiting 60s..."
    sleep 60

    python benchmark_serving_real.py --backend vllm \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
        --num-prompts -1 --request-time 60 \
        --schedule-type "${SCHED}" \
        --output-len -1 --request-rate "${RATE}" \
        --port "${PORT}" --result-dir "${RESULT_DIR}"

    kill "${PID}" 2>/dev/null || true
    sleep 5
    if kill -0 "${PID}" 2>/dev/null; then
        kill -9 "${PID}" 2>/dev/null || true
    fi
    sleep 30
}

# Sweep T values
run_T 0.0
run_T 0.5
run_T 1.0
run_T 2.0

echo ""
echo "============================================="
echo " T sweep done. Results in ${RESULT_DIR}"
echo "============================================="
ls -la "${RESULT_DIR}"/vllm-${RATE}.0qps-*-opt-cpu-async-merged*-*.json | tail -10

echo ""
echo " Compare metrics (paste vào python để A/B):"
echo " python3 -c \""
echo " import json, glob"
echo " for f in sorted(glob.glob('${RESULT_DIR}/vllm-${RATE}.0qps-*-opt-cpu-async-merged*.json')):"
echo "     d = json.load(open(f))"
echo "     print(f'{d[\\\"schedule_type\\\"]:35s}'"
echo "           f' mTTFT={d[\\\"mean_ttft_ms\\\"]:7.0f}ms'"
echo "           f' medTTFT={d[\\\"median_ttft_ms\\\"]:7.0f}ms'"
echo "           f' mTPOT={d[\\\"mean_tpot_ms\\\"]:6.1f}ms'"
echo "           f' tput={d[\\\"request_throughput\\\"]:.2f}')"
echo "\""
