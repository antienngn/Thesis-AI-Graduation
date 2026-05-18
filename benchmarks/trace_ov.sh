# trace_ov.sh — Profile opt-cpu-warmup2.0 scheduler với torch.profiler.
# Output: ./server_tracee_ov/<host>_<pid>.<timestamp>.pt.trace.json.gz
#         → drag drop vào https://ui.perfetto.dev (Perfetto load .gz trực tiếp).
#
# Mục tiêu: verify CPU OpenVINO predictor và GPU forward (Llama-3-8B) có
# thực sự overlap. Trên Perfetto sẽ thấy:
#   - Row main thread: scheduler.schedule, model_executor.execute_model,
#                      ov_predictor.submit_async, ov_predictor.poll_results
#   - Row worker "ov-predictor-0": ov_predictor.batch_total chứa
#                                   ov_predictor.tokenize, ov_predictor.inference
#   - GPU stream: CUDA kernels của Llama forward
# → Nếu block "ov_predictor.inference" trên worker thread chồng lên block
#   "model_executor.execute_model" trên main thread → confirm CPU-GPU overlap.
#
# Schedule mặc định (async_llm_engine.py:211): wait=10, warmup=5, active=50,
# repeat=1 → capture 50 forward step (~2-3s steady state). Profiler tự stop
# sau 65 step total. Bench vẫn chạy bình thường, chỉ là không trace nữa.

# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --swap-space 40 \
#     --disable-log-requests \
#     --schedule-type opt-cpu-warmup2.0 \
#     --enable-chunked-prefill \
#     --enforce-eager \
#     --dtype=half \
#     --port 3302 \
#     --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json \
#     --profile-dir ./trace_exp &
# sleep 120

# # rate=16 để load mạnh, predictor được gọi liên tục → trace có nhiều block để check overlap
# python benchmark_serving_real.py --backend vllm \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \
#     --num-prompts -1 --request-time 60 \
#     --schedule-type opt-cpu-warmup2.0 \
#     --output-len -1 --request-rate 8 \
#     --port 3302 --result-dir trace_exp

# # Optional: thêm rate khác để compare
# # python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-cpu-warmup2.0 --output-len -1 --request-rate 8 --port 3302 --result-dir server_tracee_ov
# # python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-cpu-warmup2.0 --output-len -1 --request-rate 4 --port 3302 --result-dir server_tracee_ov

# kill $!
# sleep 60


CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type opt-xxx --swap-space 16 --enable-chunked-prefill --enforce-eager --profile-dir ./trace_exp --dtype=half --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json --port 3345 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --port 3345 --result-dir trace_exp

kill $!
sleep 60

# === Cách view trace ===
# 1. ls ./server_tracee_ov/
#    → thấy file *.pt.trace.json.gz (Chrome trace format, gzipped)
# 2. Mở https://ui.perfetto.dev trong browser
# 3. Drag-drop file .gz vào perfetto (Perfetto support gz native)
# 4. Search "ov_predictor" trên timeline để locate block predictor
# 5. Search "model_executor.execute_model" để locate GPU forward block
# 6. Zoom vào 1 vùng có cả 2 → check overlap visually
#
# === Alternative: TensorBoard ===
# pip install torch_tb_profiler
# tensorboard --logdir ./server_tracee_ov
# → mở browser tab "PYTORCH_PROFILER"
