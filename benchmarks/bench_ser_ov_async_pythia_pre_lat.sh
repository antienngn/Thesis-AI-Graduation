# bench_ser_ov_async_pythia_pre_lat.sh — Đo predictor latency end-to-end
#
# Mục tiêu: thu thập wall-clock của mỗi forward call predictor để compute
# mean/median/p99 latency.
#
# Env vars:
#   STREAM_TIME=1  → log "OV-STREAM-TIME: n=X t=Ys" cho mỗi chunk forward
#                    (dùng với opt-cpu-async-merged + Pythia)
#   OPT_TIME=1     → log "OV-PRED-TIME (sync): n=X t=Ys" cho mỗi sync call
#                    (dùng với opt-xxx + OPT-125m)
#
# Log capture: stdout/stderr của server redirect vào PRE_LAT_E2E/server_*.log
# (parser sẽ scan các file này).
#
# Pipeline sau khi chạy:
#   python parse_predictor_latency.py     → predictor_latency_raw.csv
#   python build_predictor_latency_xlsx.py → predictor_latency_e2e.xlsx

mkdir -p PRE_LAT_E2E

# =============================================================================
# PYTHIA-70m + opt-cpu-async-merged1.0  (streaming worker, STREAM_TIME=1)
# =============================================================================

# #2
# RATE=2
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E

# kill $!
# sleep 60


# #4
# RATE=4
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E

# kill $!
# sleep 60

# #8
# RATE=8
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E

# kill $!
# sleep 60

# # 16
# RATE=16
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 20 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E

# kill $!
# sleep 60


# #32
# RATE=32
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E

# kill $!
# sleep 60

# #64
# RATE=64
# LOG="PRE_LAT_E2E/server_async-merged_pythia_r${RATE}.log"
# STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 30 --disable-log-requests \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --enable-chunked-prefill --enforce-eager --dtype=half \
#     --port 3303 \
#     --prefill-predictor-model-config MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     > "$LOG" 2>&1 &
# sleep 60

# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-async-merged1.0 \
#     --output-len -1 --request-rate $RATE \
#     --port 3303 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60


# # =============================================================================
# # OPT-125m + opt-xxx  (sync predictor call, OPT_TIME=1)
# # =============================================================================

# # 2
# RATE=2
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60

# # 4
# RATE=4
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60

# # 8
# RATE=8
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60

# # 16
# RATE=16
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60

# # 32
# RATE=32
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60

# # 64
# RATE=64
# LOG="PRE_LAT_E2E/server_opt-xxx_opt_r${RATE}.log"
# OPT_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
#     --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --dtype=half \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     --port 3300 > "$LOG" 2>&1 &
# sleep 60
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 \
#     --schedule-type opt-xxx --output-len -1 --request-rate $RATE --port 3300 --result-dir PRE_LAT_E2E
# kill $!
# sleep 60


# =============================================================================
# OPT-125m + opt-cpu-async-merged1.0  (OV CPU streaming, STREAM_TIME=1)
# =============================================================================
# Cấu hình thứ 3 để isolate: cùng scheduler (async-merged) + cùng backend
# (OV streaming CPU) như Pythia blocks, nhưng đổi predictor model thành
# OPT-125m → so sánh phenomenon predictor model trên cùng infrastructure.
# Log filename: server_async-merged_opt_r<rate>.log (parser pick up qua
# predictor='opt' trong regex).
# =============================================================================

#2
RATE=2
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E

kill $!
sleep 60


#4
RATE=4
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E

kill $!
sleep 60


#8
RATE=8
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E

kill $!
sleep 60


# 16
RATE=16
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 20 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E

kill $!
sleep 60


#32
RATE=32
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 40 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E

kill $!
sleep 60


#64
RATE=64
LOG="PRE_LAT_E2E/server_async-merged_opt_r${RATE}.log"
STREAM_TIME=1 CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --swap-space 30 --disable-log-requests \
    --schedule-type opt-cpu-async-merged1.0 \
    --enable-chunked-prefill --enforce-eager --dtype=half \
    --port 3303 \
    --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
    > "$LOG" 2>&1 &
sleep 60

python benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
    --num-prompts -1 --request-time 60 \
    --schedule-type opt-cpu-async-merged1.0 \
    --output-len -1 --request-rate $RATE \
    --port 3303 --result-dir PRE_LAT_E2E
kill $!
sleep 60
