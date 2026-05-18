#!/bin/bash
# bench_sweep_merged.sh — Sweep T_warmup × chunk_size.
# Câu lệnh bench giống bench_ser_ov_async_merged.sh (6 block rate 2-64),
# wrap thêm 2 vòng for cho T và chunk_size qua env OV_CHUNK_SIZE.

GPU=0
PORT=3500
RESULT_DIR=TEMP_RES_ASYNC_MERGE_SWEEP
mkdir -p ${RESULT_DIR}

for CHUNK in 1 2 4 8 16 32; do
for T in 1 2 3 4 5; do
mkdir -p ${RESULT_DIR}/r{2,4,8,16,32,64}/chunk${CHUNK}/warmup${T}
SCHED=opt-cpu-async-merged${T}


# 2
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


# 4
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


# 8
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
    --output-len -1 --request-rate 8 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r8/chunk${CHUNK}/warmup${T}

kill $!
sleep 60


# 16
OV_CHUNK_SIZE=${CHUNK} CUDA_VISIBLE_DEVICES=${GPU} python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 20 --disable-log-requests \
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
    --output-len -1 --request-rate 16 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r16/chunk${CHUNK}/warmup${T}

kill $!
sleep 60


# 32
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
    --output-len -1 --request-rate 32 \
    --port ${PORT} --result-dir ${RESULT_DIR}/r32/chunk${CHUNK}/warmup${T}

kill $!
sleep 60


# 64
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


done   # T
done   # CHUNK
