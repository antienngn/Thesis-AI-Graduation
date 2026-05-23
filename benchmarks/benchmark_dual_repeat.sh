#!/bin/bash
# benchmark_dual_repeat.sh — Chạy benchmark_dual.sh N_RUNS lần để lấy trung bình.
#
# Mỗi lần chạy → 1 thư mục riêng SERVE_DUAL_FAIR_run<i>/.
# Sau khi xong, gọi aggregate_dual_repeat.py để tổng hợp mean + std per rate.
#
# Env override:
#   N_RUNS    default 5
#   GPU       default 0
#   PORT_BASE default 3700  (tăng theo run để tránh port clash nếu chạy song song)
#   T_WARMUP  default 1.0
#   DURATION  default 60
#   RATES     default "2 4 8 16 32 64"
#   OUT_ROOT  default SERVE_DUAL_FAIR_REPEAT

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_RUNS="${N_RUNS:-5}"
GPU="${GPU:-0}"
PORT_BASE="${PORT_BASE:-3700}"
T_WARMUP="${T_WARMUP:-1.0}"
DURATION="${DURATION:-60}"
RATES="${RATES:-2 4 8 16}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/SERVE_DUAL_FAIR_REPEAT}"

mkdir -p "${OUT_ROOT}"

START_TIME=$(date -Is)
echo "=============================================="
echo " benchmark_dual_repeat — ${N_RUNS} runs dual${T_WARMUP}"
echo " GPU=${GPU}  rates=[${RATES}]  duration=${DURATION}s/rate"
echo " Output root: ${OUT_ROOT}"
echo " Start: ${START_TIME}"
echo "=============================================="

for i in $(seq 1 ${N_RUNS}); do
    OUT_DIR="${OUT_ROOT}/run${i}"
    LOG_DIR="${OUT_ROOT}/logs_run${i}"
    # Offset port per run để tránh leftover TIME_WAIT socket
    THIS_PORT_BASE=$((PORT_BASE + (i - 1) * 100))

    echo ""
    echo "######################################################"
    echo "# Run ${i}/${N_RUNS}  port_base=${THIS_PORT_BASE}  $(date -Is)"
    echo "# Output: ${OUT_DIR}"
    echo "######################################################"

    GPU=${GPU} \
    PORT_BASE=${THIS_PORT_BASE} \
    T_WARMUP=${T_WARMUP} \
    DURATION=${DURATION} \
    RATES="${RATES}" \
    OUT_DIR="${OUT_DIR}" \
    LOG_DIR="${LOG_DIR}" \
        bash "${SCRIPT_DIR}/benchmark_dual.sh"

    # Pause giữa các run để GPU thoát hẳn
    if [ ${i} -lt ${N_RUNS} ]; then
        echo "Cooldown 30s trước run kế..."
        sleep 30
    fi
done

END_TIME=$(date -Is)
echo ""
echo "=============================================="
echo " ALL ${N_RUNS} RUNS DONE.  ${START_TIME} → ${END_TIME}"
echo "=============================================="

# Aggregate kết quả
echo ""
echo "--- Aggregating ${N_RUNS} runs → mean ± std ---"
python3 "${SCRIPT_DIR}/aggregate_dual_repeat.py" \
    --src "${OUT_ROOT}" \
    --out "${OUT_ROOT}/aggregate.xlsx"

echo ""
echo " Final output:"
echo "   per-run:   ${OUT_ROOT}/run{1..${N_RUNS}}/"
echo "   averaged:  ${OUT_ROOT}/aggregate.xlsx"
echo "=============================================="
