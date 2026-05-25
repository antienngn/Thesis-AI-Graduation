# Dual Predict Scheduler: Hybrid CPU–GPU Predictor Routing cho Hệ thống Phục vụ LLM Hiệu quả

> **Khóa luận tốt nghiệp**:  Remaining-Token Prediction for Efficient Scheduling in Large Language Model Serving

> Sinh viên thực hiện: Nguyễn Tiến An — MSSV: 20021080

> Giảng viên hướng dẫn: TS. Đào Thanh Tuấn

> Năm học: 2025-2026

Khóa luận đề xuất **Dual Predict Scheduler** — một cơ chế scheduling cho LLM serving system, mở rộng kiến trúc Learning-to-Rank (LTR) bằng cách kết hợp **hai predictor chạy song song** trên hai loại phần cứng khác nhau (CPU và GPU), điều phối bởi một **LUT-based router** quyết định route từng batch unscored requests sao cho **giấu được latency của predictor sau model executor** mà không làm chậm hot path GPU.

Mã nguồn được phát triển trên nền **[vLLM-ltr](https://github.com/hao-ai-lab/vllm-ltr)** (Fu et al., 2024); phần đóng góp của khóa luận được ghi nhận chi tiết ở mục [Đóng góp](#đóng-góp-của-khóa-luận) và [Citation](#citation) ở cuối README.

---

## Motivation

Hầu hết LLM serving system (vLLM, TGI, …) dùng **First-Come-First-Serve (FCFS)** vì output length của request không thể biết trước. Điều này gây **Head-Of-Line (HOL) blocking** và làm giảm throughput.

vLLM-ltr giải quyết bằng cách dùng **predictor (OPT-125m fine-tune learning-to-rank)** để xếp hạng request theo độ dài output tương đối, từ đó xấp xỉ Shortest-Job-First (SJF) và đạt **2.8× giảm latency** cho chatbot, **6.5× tăng throughput** cho synthetic generation.

Tuy nhiên, phương pháp của **[Fu et al.](https://arxiv.org/abs/2408.15792)** là chạy predictor đồng bộ trên GPU, mỗi lần prediect chiếm cùng một stream với main model gây ra predictor latency cộng thẳng vào tick time của continuous batching loop. Khi:

- **Số unscored requests/tick lớn** (burst arrival), hoặc
- **Prompt dài** (predictor forward đắt),

thì phần thời gian scoring trở thành cổ chai và bù trừ một phần lợi ích của LTR.

Quan sát chính của khóa luận:

> Trong một tick continuous batching, **main model executor** đã chiếm full GPU compute, để lại CPU idle. Nếu chuyển predictor sang chạy song song trên CPU, nó có thể **overlap hoàn toàn** với model executor → predictor latency **bị giấu**, không cộng vào tick time, giảm bớt gánh nặng cho GPU.
>
> Tuy nhiên, khi batch unscored quá lớn hoặc prompt rất dài, **CPU predictor lại không kịp** xong trước khi main model executor end → giấu không hết, latency lớn lại cộng vào tick.

→ Cần một **router** quyết định **mỗi tick** đẩy bớt batch request sang CPU (async, không chặn) hay sang GPU (sync dựa trên ước lượng **`T_cpu(batch)` vs `T_main(state)`**, dùng 2 bảng tra cứu (Look-Up-Tables). Đó là ý tưởng chính của **Dual Predict Scheduler `dual<T>`**.

---

## Scheduler `dual<T>` — Kiến trúc và các thành phần

### Tổng quan flow trong một tick

```
                     ┌──────────────────────────────────────────────┐
   unscored          │              dual<T> scheduler                │
   request    ─────► │                                                │
                     │   if   t < T  (warmup):                        │
                     │       ───► GPU.obtain_aux_scores(all)   sync   │
                     │                                                │
                     │   else (post-warmup):                          │
                     │       cpu, gpu = Router.route_batch(unscored,  │
                     │                                       state)   │
                     │       if cpu:  CPU.submit_streaming(cpu)  async│
                     │       if gpu:  GPU.obtain_aux_scores(gpu) sync │
                     └──────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Main model executor │  (overlap với CPU)
                              │   Llama-3-8B forward │
                              └──────────────────────┘
```

Decision logic được implement ở [`vllm/model_executor/dual_predictor_router.py`](vllm/model_executor/dual_predictor_router.py) và compose tại [`vllm/model_executor/dual_aux_model.py`](vllm/model_executor/dual_aux_model.py); orchestrator ở [`vllm/core/scheduler.py`](vllm/core/scheduler.py).

### Các thành phần

#### 1. `DualAuxModel` — Composite predictor

[`vllm/model_executor/dual_aux_model.py`](vllm/model_executor/dual_aux_model.py)

Đóng gói **3 thành phần** sau một API duy nhất tương thích với cả `OpenVINOPredictor` và `AUXLLM` (để scheduler không cần biết bên trong là dual hay single):

| Field | Kiểu | Vai trò |
|---|---|---|
| `.cpu` | `OpenVINOPredictor` | Async streaming, chạy OPT-125m INT8 trên CPU AVX-512 |
| `.gpu` | `AUXLLM` | Sync, chạy OPT-125m PyTorch trên GPU (giống flow `opt-xxx`) |
| `.router` | `DualPredictorRouter` | Quyết định CPU vs GPU mỗi tick |

API chính:
- `submit_streaming(seq_groups)` → đẩy batch sang CPU (non-blocking).
- `obtain_aux_scores(seq_groups)` → score sync qua GPU AUXLLM.
- `route_batch(seq_groups, state)` → trả về `(cpu_list, gpu_list)`.
- `is_pending_cpu_score(seq_group)` → tránh race-condition khi cùng một request đang ở CPU queue mà scheduler vô tình dispatch lại sang GPU sync.

#### 2. `DualPredictorRouter` — LUT-based routing

[`vllm/model_executor/dual_predictor_router.py`](vllm/model_executor/dual_predictor_router.py)

Mỗi tick, router **quyết định whole-batch** (all-or-nothing): toàn bộ unscored route CPU hoặc toàn bộ route GPU, dựa trên so sánh hai estimate:

- **`T_cpu = LUT_CPU[ (n_req, longest_OPT_tokens) ]`** — ước lượng latency của CPU predictor khi score batch này.
- **`T_main = LUT_MAIN[ (n_running, n_decode, n_prefill, n_tokens_next) ]`** — ước lượng latency của main model executor cho tick tiếp theo.

Quyết định:
```
T_main >= T_cpu  →  route all to CPU   (giấu được predictor sau executor)
T_main <  T_cpu  →  route all to GPU   (CPU không kịp → chấp nhận sync)
```

**Lookup convention**:
- CPU LUT (2D): ceiling cả hai chiều — bucket `[lo, hi)` với key là `hi`. Out-of-LUT (vd prompt > 1024 OPT tokens) → trả `inf` → force GPU.
- Main LUT (4D): floor cả 4 chiều, exact lookup. Cell empty → trả `0` → force GPU.

**Tokenizer handling**: prompt được tokenize bằng **OPT tokenizer** (không phải Llama) vì LUT CPU build theo đơn vị OPT n_tokens; kết quả cache trên `seq_group._router_opt_n_tokens` để tránh tokenize lại mỗi tick.

#### 3. Warmup phase (`t < T`)

Trong `T` giây đầu kể từ lúc server start, **mọi request đều route GPU sync** bất kể router output. Lý do:
- Cold-start: CPU OpenVINO compile model (~vài giây), main model load + warmup pages.
- Tránh routing decision dựa trên state chưa ổn định (`n_running ≈ 0`, LUT cells chưa khớp).

`T` truyền qua tên scheduler — vd `--schedule-type dual1.0` nghĩa là warmup 1.0s.

#### 4. Hai Lookup Table (LUT) build offline

[`benchmarks/LUT_CREATE/`](benchmarks/LUT_CREATE/)

| LUT | Trục | Cách build |
|---|---|---|
| `cpu_predictor_lut.json` | `(n_requests, longest_OPT_tokens)` → `latency_ms` | [`profile_cpu_predictor_lut.py`](benchmarks/LUT_CREATE/profile_cpu_predictor_lut.py) — standalone, không cần GPU, ~10 phút |
| `main_model_lut.json` | `(n_running, n_decode, n_prefill, n_tokens)` → `latency_ms` | Sweep [`run_bench_sweep_main_lut.sh`](benchmarks/LUT_CREATE/run_bench_sweep_main_lut.sh) (6 bench rate, scheduler `opt-xxx`) + [`build_main_model_lut.py`](benchmarks/LUT_CREATE/build_main_model_lut.py) — ~30 phút |

Bench sweep dùng scheduler `opt-xxx` (paper baseline) để latency main model thuần, không lẫn delay của CPU predictor variant khác.

#### 5. Integration vào scheduler & engine

Các điểm sửa (so với upstream vLLM-ltr):

- [`vllm/core/scheduler.py`](vllm/core/scheduler.py) — thêm nhánh `schedule_type.startswith("dual")`, parse `T`, gọi `router.route_batch()`, dispatch CPU/GPU.
- [`vllm/engine/llm_engine.py`](vllm/engine/llm_engine.py), [`vllm/engine/async_llm_engine.py`](vllm/engine/async_llm_engine.py) — load `DualAuxModel` khi config `device="dual"`, thêm field `n_running` vào event trace để build main LUT.
- [`vllm/config_predictor.py`](vllm/config_predictor.py) — `PrefillModelConfig` thêm các field cho mode dual:
  - `device="dual"` để bật.
  - `gpu_predictor_config_path` — usage_config cho AUXLLM GPU.
  - `lut_cpu_path`, `lut_main_path` — đường dẫn 2 LUT.

---

## Installation

```bash
conda create -n vllm-ltr python=3.10
conda activate vllm-ltr
git clone https://github.com/antienngn/Thesis-AI-Graduation.git vllm-ltr
cd vllm-ltr
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 \
    -c pytorch -c nvidia
pip install -e .
pip install flash-attn torchaudio==2.2.1 torchvision==0.17.1 numpy==1.25.2 \
    fschat accelerate gcsfs scikit-learn scipy matplotlib evaluate
# Bổ sung cho mode dual:
pip install openvino openvino-genai
```

---

## Reproduce kết quả

### Bước 1 — Train predictor (hoặc dùng checkpoint có sẵn)

- Train từ đầu: xem [`./train`](train/).
- Hoặc tải fine-tuned checkpoint từ Hugging Face: [LLM-ltr/OPT-Predictors](https://huggingface.co/LLM-ltr/OPT-Predictors).

### Bước 2 — Build 2 LUT (1 lần / cấu hình phần cứng)

```bash
cd benchmarks/LUT_CREATE
GPU=0 bash run_all_tmux.sh         # CPU LUT + Main LUT song song trong tmux
tmux attach -t lut                   # theo dõi
```

Hoặc tuần tự:
```bash
python profile_cpu_predictor_lut.py
GPU=0 bash run_bench_sweep_main_lut.sh
python build_main_model_lut.py
```

Output ở `benchmarks/LUT_CREATE/data/{cpu_predictor_lut.json, main_model_lut.json}`. Xem [`benchmarks/LUT_CREATE/README.md`](benchmarks/LUT_CREATE/README.md) cho chi tiết schema và rebuild khi đổi hardware/model.

### Bước 3 — Chạy benchmark scheduler `dual<T>`

Bench single-rate có nsys profile + NVTX:
```bash
cd benchmarks
GPU=1 RATE=8 T_WARMUP=1.0 DURATION=60 bash bench_dual.sh
```

Sweep tất cả rate:
```bash
GPU=1 bash bench_all_dual.sh
```

Outputs trong `benchmarks/TEMP_PROF_DUAL_R*_NSIGHT/`: `trace_merged.nsys-rep`, `trace_merged.csv`, `tick_profile_merged.csv`, `bench.log`, `server.log`. Script tự verify số lượng router decision, CPU submit, GPU sync, OV forward call từ trace cuối.

### Bước 4 — Phân tích & vẽ plot

```bash
python benchmarks/aggregate_dual_repeat.py   # gộp nhiều repeat
python benchmarks/build_ser_res_dual_xlsx.py # xuất Excel so sánh
```

---

## Đóng góp của khóa luận

Repo này là **fork** của [hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr). Phần do khóa luận đóng góp:

- **Thiết kế scheduler `dual<T>`** — Warm up first T second và hoạt động với 2 predictor song song. [`vllm/core/scheduler.py`](vllm/core/scheduler.py)
- **`DualAuxModel` composite predictor** giữ API contract với scheduler không đổi. [`vllm/model_executor/dual_aux_model.py`](vllm/model_executor/dual_aux_model.py)
- **`DualPredictorRouter` LUT-based routing** với 2 LUT (CPU 2D + main 4D) và logic ceiling/floor lookup. [`vllm/model_executor/dual_predictor_router.py`](vllm/model_executor/dual_predictor_router.py)
- **OpenVINO INT8/AVX-512 backend** cho OPT-125m predictor (chạy non-blocking trên CPU).
- **Build LUT offline** ([`benchmarks/LUT_CREATE/`](benchmarks/LUT_CREATE/)) gồm: profile CPU latency, sweep main model latency, build LUT JSON, vẽ heatmap.
- **Bench `dual`** với nsys profile + NVTX markers + event tracer + tick profile CSV để phân tích overlap CPU/GPU. [`benchmarks/bench_dual.sh`](benchmarks/bench_dual.sh), [`benchmarks/benchmark_dual.sh`](benchmarks/benchmark_dual.sh), [`benchmarks/benchmark_dual_repeat.sh`](benchmarks/benchmark_dual_repeat.sh).
- **Phân tích chuyên sâu** dual vs `opt-xxx` baseline trên các metric TTFT, TPOT, latency sweep và batch composition (xem [`benchmarks/RESULT_DUAL_FINAL/`](benchmarks/RESULT_DUAL_FINAL/)).

---

## Acknowledgement & Citation

Khóa luận được xây dựng trực tiếp trên ý tưởng learning-to-rank scheduling và codebase của paper **"Efficient LLM Scheduling by Learning to Rank"** (Fu et al., NeurIPS 2024). Xin chân thành cảm ơn các tác giả gốc đã công khai mã nguồn để khóa luận có thể tiếp nối.

Nếu sử dụng repo hoặc kết quả này, vui lòng cite cả paper gốc:

```bibtex
@article{fu2024efficient,
  title   = {Efficient LLM Scheduling by Learning to Rank},
  author  = {Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal = {arXiv preprint arXiv:2408.15792},
  year    = {2024}
}
```

Paper gốc: [arXiv:2408.15792](https://arxiv.org/abs/2408.15792) — [hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr).
