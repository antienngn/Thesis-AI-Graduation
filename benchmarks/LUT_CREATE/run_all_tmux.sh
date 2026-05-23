#!/bin/bash
# Tmux orchestrator: chạy song song 2 LUT jobs trong 2 panes.
#   pane 0 (top):    LUT CPU predictor (standalone, không cần GPU)
#   pane 1 (bottom): LUT main model bench sweep + build (cần GPU)
#
# Sau khi cả 2 xong → data/cpu_predictor_lut.json + data/main_model_lut.json
#
# Usage:
#   GPU=0 bash run_all_tmux.sh              # mặc định: GPU 0, session "lut"
#   GPU=3 SESSION=lut2 bash run_all_tmux.sh # custom GPU + session name
#
# Attach lại sau: tmux attach -t lut

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-lut}"
GPU="${GPU:-0}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' đã tồn tại. Kill trước rồi rerun:"
    echo "  tmux kill-session -t ${SESSION}"
    exit 1
fi

# Tạo session với pane đầu cho CPU LUT
tmux new-session -d -s "${SESSION}" -n main -c "${SCRIPT_DIR}"

# Pane 0: CPU LUT
tmux send-keys -t "${SESSION}:main.0" \
    "source /home/antn/miniconda3/etc/profile.d/conda.sh && conda activate vllm-ltr && python profile_cpu_predictor_lut.py 2>&1 | tee data/cpu_lut.log && echo '=== CPU LUT DONE ==='" \
    C-m

# Split horizontal → pane 1
tmux split-window -v -t "${SESSION}:main" -c "${SCRIPT_DIR}"

# Pane 1: Main model LUT (bench sweep + build)
tmux send-keys -t "${SESSION}:main.1" \
    "GPU=${GPU} bash run_bench_sweep_main_lut.sh 2>&1 | tee data/main_sweep.log && python build_main_model_lut.py 2>&1 | tee data/main_build.log && echo '=== MAIN LUT DONE ==='" \
    C-m

# Even split
tmux select-layout -t "${SESSION}:main" even-vertical

echo "=============================================="
echo " Tmux session '${SESSION}' đã khởi tạo."
echo " GPU=${GPU}"
echo ""
echo " Attach để theo dõi:"
echo "   tmux attach -t ${SESSION}"
echo ""
echo " Log files (theo dõi từ máy khác):"
echo "   tail -f ${SCRIPT_DIR}/data/cpu_lut.log"
echo "   tail -f ${SCRIPT_DIR}/data/main_sweep.log"
echo ""
echo " Tổng thời gian ước tính: ~30 phút"
echo "   - CPU LUT:  ~10 phút"
echo "   - Main LUT: ~30 phút (6 bench × 5 phút) + 1 phút build"
echo "=============================================="
