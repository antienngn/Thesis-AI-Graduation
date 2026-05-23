#!/bin/bash
# benchmark_dual_verify.sh — verify tính lặp lại bằng cách chạy
# benchmark_dual.sh 5 lần với rates=[4,8,16] (tập trung vùng elbow,
# bỏ qua low-rate noise + saturation regime), rồi gom mean ± std.
#
# Layout output: SERVE_DUAL_FAIR_VERIFY/run{1..5}/ flat JSONs.
# Cuối cùng aggregate.xlsx được sinh tự động.
#
# Env override (mặc định đủ dùng):
#   N_RUNS    default 5
#   RATES     default "4 8 16"
#   GPU       default 0
#   PORT_BASE default 3700
#   T_WARMUP  default 1.0
#   DURATION  default 60

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_RUNS="${N_RUNS:-5}"
GPU="${GPU:-0}"
PORT_BASE="${PORT_BASE:-3700}"
T_WARMUP="${T_WARMUP:-1.0}"
DURATION="${DURATION:-60}"
RATES="${RATES:-4 8 16}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/SERVE_DUAL_FAIR_VERIFY}"

mkdir -p "${OUT_ROOT}"

START_TIME=$(date -Is)
echo "=============================================="
echo " benchmark_dual_verify — ${N_RUNS} runs dual${T_WARMUP}"
echo " RATES=[${RATES}]  GPU=${GPU}  duration=${DURATION}s/rate"
echo " Output root: ${OUT_ROOT}"
echo " Start: ${START_TIME}"
echo "=============================================="

for i in $(seq 1 ${N_RUNS}); do
    OUT_DIR="${OUT_ROOT}/run${i}"
    LOG_DIR="${OUT_ROOT}/logs_run${i}"
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
