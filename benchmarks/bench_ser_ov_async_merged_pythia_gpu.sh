#!/bin/bash
# bench_ser_ov_async_merged.sh — Full sweep r=2,4,8,16,32 cho
# opt-cpu-async-merged1.0. Tạo JSON output đầy đủ để A/B với baseline cũ.
#
# Khác biệt so với bench_ser_ov_async.sh:
#   - schedule_type 'opt-cpu-async-merged1.0' thay vì 'opt-cpu-async-warmup1.0'
#   - Port 3327 (riêng để chạy song song nếu cần)
#   - KHÔNG bật OPT_PROFILE_TICK (chỉ để bench production-like, không
#     overhead instrumentation; chạy profile riêng qua profile_ov_async_merged.sh)
#
# Output:
#   TEMP_RES_ASYNC/vllm-{2,4,8,16,32}.0qps-...-opt-cpu-async-merged1.0-<TS>.json
#
# Mỗi rate ~5-10 phút tùy tail drain. Total ~30-50 phút.

# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RESULT_DIR="${SCRIPT_DIR}/TEMP_RES_ASYNC_MERGE"
# PORT=3327
# SCHED="opt-cpu-async-merged1.0"

# bench_ser_ov_async.sh — Ranking + CPU OV predictor + 2s FCFS warmup + ASYNC streaming
#
# Schedule_type: opt-cpu-async-warmup2.0
#   - 2s warmup phase: FCFS by arrival_time (no predictor)
#   - Drain phase: pick CHỈ warmup-era requests, post-warmup BLOCKED, predictor
#     EAGER score post-warmup parallel với GPU drain
#   - Post-drain phase: sort by predictor score, gate unscored
#
# Predictor: streaming API (continuous queue, dedup nội bộ)
#   - submit_streaming always accepts
#   - Worker thread apply scores incremental cross-thread
#   - Request mới có thể submit anytime, không bị block
#
# Properties:
#   - 100% post-warmup requests scored TRƯỚC khi đưa vào model execution
#   - Warmup-era drain sequential (FCFS) trước predictor sort
#   - CPU/GPU true parallel (no GPU contention)
#
# Port khác bench_ser_ov.sh để có thể chạy song song nếu cần A/B test.

#2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 2 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU

kill $!
sleep 60


# #4
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 4 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU

kill $!
sleep 60

#8
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 8 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU

kill $!
sleep 60

# 16
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 20 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 16 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU

kill $!
sleep 60


#32
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 32 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU

kill $!
sleep 60

#64
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate 64 \
    --port 3303 --result-dir TEMP_RES_ASYNC_MERGE_PYTHIA_GPU
kill $!
sleep 60
# === Optional: enable OPT_TIME=1 để log predictor activity ===
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --schedule-type opt-cpu-async-warmup2.0 ... &
# Logs sẽ in:d
#   - "OPT-CPU-ASYNC-SUBMIT: n=<X> new=<Y> t=<latency>"
#     n: số prompts submit, new: số mới được add (sau dedup)
#   - "OV-STREAM: applied <X> scores"
#     mỗi mini-batch xong, log số scores apply

