# bench_sweep.sh — Confirm batch contention hypothesis bằng cách cap max-num-seqs
# Hypothesis: TPOT cao của opt-cpu ở rate=16 là do running batch quá lớn.
# Test: cap max_num_seqs xuống các mức {32, 64, 128, 256}, đo TPOT.
#   - Nếu TPOT giảm khi cap nhỏ → batch contention CONFIRMED.
#   - Nếu TPOT không đổi → cause khác (CPU contention, predictor overhead, ...).
#
# Setup: rate=16 cố định (chỉ rate này có TPOT crossover). Mỗi cap = 1 server
# instance riêng → cần kill + restart giữa các cap để clean state.
#
# Output: SWEEP_RES/cap{N}/ chứa .pt và .json cho mỗi cap.
# Dùng analyze_sweep.py (sẽ tạo sau) để vẽ TPOT vs cap.

mkdir -p SWEEP_RES

# # ───── Cap = 512 ─────
# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-warmup2.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --max-num-seqs 512 \
#     --port 3302 \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
# sleep 60
# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-warmup2.0 \
#     --output-len -1 --request-rate 16 \
#     --port 3302 --result-dir SWEEP_RES/cap512
# kill $!
# sleep 60

# ───── Cap = 1024 ─────
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-warmup2.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --max-num-seqs 768 \
    --port 3302 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
sleep 60
python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-warmup2.0 \
    --output-len -1 --request-rate 16 \
    --port 3302 --result-dir SWEEP_RES/cap1024
kill $!
sleep 60

# # ───── Cap = 2048 ─────
# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-warmup2.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --max-num-seqs 2048 \
#     --port 3302 \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
# sleep 60
# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-warmup2.0 \
#     --output-len -1 --request-rate 16 \
#     --port 3302 --result-dir SWEEP_RES/cap2048
# kill $!
# sleep 60

# # ───── Cap = 256 (default-ish, baseline opt-cpu) ─────
# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-warmup2.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --max-num-seqs 256 \
#     --port 3302 \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
# sleep 60
# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-warmup2.0 \
#     --output-len -1 --request-rate 16 \
#     --port 3302 --result-dir SWEEP_RES/cap256
# kill $!
# sleep 60

# ───── Sau khi xong, view kết quả ─────
# ls SWEEP_RES/cap{32,64,128,256}/
# Mỗi folder có 1 file .pt và 1 file .json giống bench_ser_ov.sh.
# Compare TPOT mean của 4 cap → confirm batch contention nếu monotonic giảm khi cap nhỏ.
