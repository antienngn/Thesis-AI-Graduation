# LUT_CREATE — Build 2 lookup tables cho scheduler `dual<T>`

Folder này chứa toàn bộ script + dữ liệu để build 2 LUT mà router của
scheduler `dual<T>` cần:

1. **LUT CPU predictor** — `n_tokens → cpu_predictor_latency`
2. **LUT main model**   — `(n_running, n_decode, n_prefill, n_tokens) → main_model_latency`

## File layout

```
LUT_CREATE/
├── README.md                          # file này
├── profile_cpu_predictor_lut.py        # build LUT 1 (standalone)
├── run_bench_sweep_main_lut.sh         # sweep 6 bench rate (opt-xxx)
├── build_main_model_lut.py             # build LUT 2 từ trace
├── run_all_tmux.sh                     # orchestrator chạy song song
└── data/                                # output
    ├── cpu_predictor_lut.json
    ├── cpu_predictor_lut.png
    ├── main_model_lut.json
    ├── main_model_lut.png
    └── bench_r{2,4,8,16,32,64}/        # raw bench output
        ├── trace_merged.csv
        ├── server.log
        └── bench.log
```

## Cách chạy

### Cách 1 — Tmux parallel (recommended)

```bash
GPU=0 bash run_all_tmux.sh         # mặc định session "lut"
tmux attach -t lut                  # theo dõi tiến độ
```

CPU LUT chạy ở pane 0 (~10 phút, không cần GPU).
Main LUT chạy ở pane 1 (~30 phút, dùng GPU đã chỉ định).
Hai bên độc lập, không tranh nhau.

### Cách 2 — Sequential

```bash
# 1. CPU LUT
python profile_cpu_predictor_lut.py

# 2. Main LUT
GPU=0 bash run_bench_sweep_main_lut.sh
python build_main_model_lut.py
```

## Schema LUT

### `cpu_predictor_lut.json`

```json
{
  "model": "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32",
  "device": "cpu_ov_avx512",
  "tokenizer_for_bucketing": "meta-llama/Meta-Llama-3-8B-Instruct",
  "buckets": [
    {
      "n_tokens_lo": 0, "n_tokens_hi": 64,
      "lat_ms_p50": 12.3, "lat_ms_p95": 15.0,
      "lat_ms_mean": 12.8, "lat_ms_std": 1.2,
      "n_samples_kept": 90, ...
    }, ...
  ]
}
```

**Cách router lookup**: `n_tokens` (Llama tokenize) → tìm bucket `[lo, hi)` →
return `lat_ms_p50` hoặc `lat_ms_p95`.

### `main_model_lut.json`

```json
{
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "device": "v100_sxm2_fp16",
  "scheduler_used": "opt-xxx",
  "feature_keys": ["n_running", "n_decode", "n_prefill", "n_tokens"],
  "bucket_edges": {
    "n_running": [0, 4, 8, 16, 32, 64, 128, 256, 1000],
    "n_decode":  [0, 4, 8, 16, 32, 64, 128, 256, 1000],
    "n_prefill": [0, 1, 2, 3, 5, 10, 1000],
    "n_tokens":  [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000]
  },
  "cells": [
    {
      "key": {"n_running": [4,8], "n_decode": [4,8], "n_prefill": [0,1], "n_tokens": [32,64]},
      "n_samples": 1820, "lat_ms_p50": 22, "lat_ms_p95": 28, ...
    }, ...
  ]
}
```

**Cách router lookup**:
1. Quantize từng feature qua `bucket_edges` → tuple key
2. Tìm cell match exact, nếu không có → nearest-neighbor (Manhattan distance trên bucket indices)
3. Return `lat_ms_p50`

Cell với <5 sample bị drop (trống).

## Dependencies cho router

Khi viết `vllm/model_executor/dual_predictor_router.py`, load 2 JSON ở init,
implement 2 helper:

```python
def estimate_cpu_lat(self, n_tokens: int) -> float:
    """Lookup CPU predictor latency."""

def estimate_main_lat(self, state: Dict[str, int]) -> float:
    """Lookup main model latency, state = {n_running, n_decode, n_prefill, n_tokens}."""

def route(self, seq_group, state) -> str:  # 'cpu' | 'gpu'
    n_tokens = seq_group.prompt_token_count()
    t_cpu = self.estimate_cpu_lat(n_tokens)
    t_main = self.estimate_main_lat(state)
    return "cpu" if t_main >= t_cpu else "gpu"
```

## Rebuild khi đổi config

- Đổi model main (vd Llama 70B) → chạy lại `run_bench_sweep_main_lut.sh`
- Đổi predictor (vd OPT-350m) → chạy lại `profile_cpu_predictor_lut.py`
- Đổi GPU (vd A100 thay V100) → chạy lại main sweep
- Đổi CPU (vd Sapphire Rapids) → chạy lại CPU LUT

## Lưu ý

- Bench sweep dùng scheduler **opt-xxx** (paper baseline) để LUT main model
  phản ánh latency thuần của Llama-3-8B forward, không bị nhiễu bởi CPU
  predictor delay của các variant khác.
- Field `n_running = len(scheduler.running)` đã được thêm vào
  `event_tracer.log("model_executor.start", ...)` trong
  `vllm/engine/async_llm_engine.py`.
- Nếu chạy lại trên trace cũ thiếu `n_running`, script `build_main_model_lut.py`
  fallback dùng `n_seqs` (kèm warning).
