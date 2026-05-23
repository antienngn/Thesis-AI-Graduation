#!/bin/bash
# nsys profile dual1.0 vs dual2.0 ở rate=16, chạy tuần tự.
# Wrapper gọi bench_dual.sh với T_WARMUP={1.0, 2.0}.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
PORT="${PORT:-3470}"
RATE="${RATE:-16}"
DURATION="${DURATION:-60}"

MASTER_LOG_DIR="${SCRIPT_DIR}/TEMP_PROF_DUAL_R${RATE}_COMPARE"
mkdir -p "${MASTER_LOG_DIR}"
MASTER_LOG="${MASTER_LOG_DIR}/master.log"
SUMMARY_LOG="${MASTER_LOG_DIR}/summary.log"
: > "${SUMMARY_LOG}"

echo "==============================================" | tee "${MASTER_LOG}"
echo " profile_dual_r16_compare — dual1.0 vs dual2.0" | tee -a "${MASTER_LOG}"
echo " RATE=${RATE}  GPU=${GPU}  PORT=${PORT}  DURATION=${DURATION}" | tee -a "${MASTER_LOG}"
echo " Started: $(date -Is)" | tee -a "${MASTER_LOG}"
echo "==============================================" | tee -a "${MASTER_LOG}"

for T in 1.0 2.0; do
    echo "" | tee -a "${MASTER_LOG}"
    echo "######################################################" | tee -a "${MASTER_LOG}"
    echo "# Run dual${T}  rate=${RATE}  $(date -Is)" | tee -a "${MASTER_LOG}"
    echo "######################################################" | tee -a "${MASTER_LOG}"

    GPU=${GPU} PORT=${PORT} RATE=${RATE} T_WARMUP=${T} DURATION=${DURATION} \
        bash "${SCRIPT_DIR}/bench_dual.sh" 2>&1 | tee -a "${MASTER_LOG}"

    OUT_DIR="${SCRIPT_DIR}/TEMP_PROF_DUAL_R${RATE}_NSIGHT"
    DEST_DIR="${MASTER_LOG_DIR}/dual${T}"
    if [ -d "${OUT_DIR}" ]; then
        rm -rf "${DEST_DIR}"
        mv "${OUT_DIR}" "${DEST_DIR}"
        echo "Moved ${OUT_DIR} -> ${DEST_DIR}" | tee -a "${MASTER_LOG}"
    fi

    TRACE_CSV="${DEST_DIR}/trace_merged.csv"
    {
        echo "===== dual${T}  rate=${RATE}  $(date -Is) ====="
        if [ -f "${TRACE_CSV}" ]; then
            N_ROUTER=$(grep -c "dual.router.end" "${TRACE_CSV}" 2>/dev/null || echo 0)
            N_CPU_SUBMIT=$(grep -c "predictor.submit.start.*dual_cpu" "${TRACE_CSV}" 2>/dev/null || echo 0)
            N_GPU_SYNC=$(grep -c "predictor.gpu_sync.start" "${TRACE_CSV}" 2>/dev/null || echo 0)
            N_OV_FWD=$(grep -c "predictor.worker.forward.end" "${TRACE_CSV}" 2>/dev/null || echo 0)
            N_ME=$(grep -c "model_executor.start" "${TRACE_CSV}" 2>/dev/null || echo 0)
            printf "  %-30s %s\n" "router decisions"            "${N_ROUTER}"
            printf "  %-30s %s\n" "CPU submit calls"            "${N_CPU_SUBMIT}"
            printf "  %-30s %s\n" "GPU sync score calls"        "${N_GPU_SYNC}"
            printf "  %-30s %s\n" "OV inference (worker.fwd)"   "${N_OV_FWD}"
            printf "  %-30s %s\n" "model_executor calls"        "${N_ME}"
        else
            echo "  (no trace_merged.csv found at ${TRACE_CSV})"
        fi
        BENCH_LOG="${DEST_DIR}/bench.log"
        if [ -f "${BENCH_LOG}" ]; then
            BENCH_DUR=$(grep "Benchmark duration"     "${BENCH_LOG}" | awk '{print $NF}')
            TOT_IN=$(grep    "Total input tokens"      "${BENCH_LOG}" | awk '{print $NF}')
            TOT_OUT=$(grep   "Total generated tokens"  "${BENCH_LOG}" | awk '{print $NF}')
            SUCC=$(grep      "Successful requests"     "${BENCH_LOG}" | awk '{print $NF}')
            printf "  %-30s %s\n" "successful requests"        "${SUCC:-?}"
            printf "  %-30s %s\n" "benchmark duration (s)"     "${BENCH_DUR:-?}"
            printf "  %-30s %s\n" "total input tokens"         "${TOT_IN:-?}"
            printf "  %-30s %s\n" "total generated (re-tok)"   "${TOT_OUT:-?}"
        fi
        echo ""
    } | tee -a "${SUMMARY_LOG}" | tee -a "${MASTER_LOG}"

    echo "Cooldown 30s..." | tee -a "${MASTER_LOG}"
    sleep 30
done

echo "" | tee -a "${MASTER_LOG}"
echo "==============================================" | tee -a "${MASTER_LOG}"
echo " All done: $(date -Is)" | tee -a "${MASTER_LOG}"
echo " Results:" | tee -a "${MASTER_LOG}"
for T in 1.0 2.0; do
    D="${MASTER_LOG_DIR}/dual${T}"
    if [ -f "${D}/trace_merged.nsys-rep" ]; then
        SIZE=$(du -h "${D}/trace_merged.nsys-rep" | cut -f1)
        echo "  ✓ dual${T}: ${D}/trace_merged.nsys-rep (${SIZE})" | tee -a "${MASTER_LOG}"
    else
        echo "  ✗ dual${T}: MISSING ${D}/trace_merged.nsys-rep" | tee -a "${MASTER_LOG}"
    fi
done
echo "" | tee -a "${MASTER_LOG}"
echo "Summary:" | tee -a "${MASTER_LOG}"
cat "${SUMMARY_LOG}" | tee -a "${MASTER_LOG}"
echo "==============================================" | tee -a "${MASTER_LOG}"
