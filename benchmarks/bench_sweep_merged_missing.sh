#!/bin/bash
# bench_sweep_merged_missing.sh — chạy nốt 3 cell thiếu trong sweep:
#   (rate=2,  chunk=4, T=2)
#   (rate=4,  chunk=4, T=3)
#   (rate=64, chunk=4, T=2)
# Cùng pattern với bench_sweep_merged.sh, chỉ khác GPU/PORT để chạy song song.

GPU=2
PORT=3502
RESULT_DIR=TEMP_RES_ASYNC_MERGE_SWEEP
CHUNK=4
mkdir -p ${RESULT_DIR}

# ---------- (rate=2, chunk=4, T=2) ----------
T=2
SCHED=opt-cpu-async-merged${T}
mkdir -p ${RESULT_DIR}/r2/chunk${CHUNK}/warmup${T}

OV_CHUNK_SIZE=${CHUNK} CUDA_VISIBLE_DEVICES=${GPU} python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type ${SCHED} \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type ${SCHED} \
    --output-len -1 --request-rate 2 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r2/chunk${CHUNK}/warmup${T}

kill $!
sleep 60


# ---------- (rate=4, chunk=4, T=3) ----------
T=3
SCHED=opt-cpu-async-merged${T}
mkdir -p ${RESULT_DIR}/r4/chunk${CHUNK}/warmup${T}

OV_CHUNK_SIZE=${CHUNK} CUDA_VISIBLE_DEVICES=${GPU} python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type ${SCHED} \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type ${SCHED} \
    --output-len -1 --request-rate 4 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r4/chunk${CHUNK}/warmup${T}

kill $!
sleep 60


# ---------- (rate=64, chunk=4, T=2) ----------
T=2
SCHED=opt-cpu-async-merged${T}
mkdir -p ${RESULT_DIR}/r64/chunk${CHUNK}/warmup${T}

OV_CHUNK_SIZE=${CHUNK} CUDA_VISIBLE_DEVICES=${GPU} python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type ${SCHED} \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port ${PORT} \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type ${SCHED} \
    --output-len -1 --request-rate 64 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r64/chunk${CHUNK}/warmup${T}

kill $!
sleep 60
