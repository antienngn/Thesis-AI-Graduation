# Predictor Replacement Experiments — ShareGPT

Train 4 candidate models thay thế OPT-125M baseline trên ShareGPT/Llama-3-8B labels.

## Candidates

| Folder | Model | Params (Seq classification) | Context | Architecture |
|---|---|---|---|---|
| `distilbert/` | `distilbert-base-uncased` | ~67M | 512 | Encoder |
| `pythia-14m/` | `EleutherAI/pythia-14m-deduped` | ~7.6M | 2048 | Decoder |
| `pythia-31m/` | `EleutherAI/pythia-31m-deduped` | ~17.6M | 2048 | Decoder |
| `pythia-70m/` | `EleutherAI/pythia-70m-deduped` | ~44.7M | 2048 | Decoder |

Baseline tham chiếu: **OPT-125M, Tau = 0.52** (ShareGPT/Llama-3-8B, từ `train.sh` line 55).

## Recommended training order

Train tuần tự (KHÔNG parallel) để có early signal:

1. **`pythia-70m/`** — most likely match baseline. Nếu không match, pivot sớm.
2. **`pythia-31m/`** — middle data point cho Pareto curve.
3. **`pythia-14m/`** — floor experiment. Biết capacity threshold.
4. **`distilbert/`** — encoder hypothesis test, độc lập với Pythia sweep.

```bash
bash runs/pythia-70m/run.sh
bash runs/pythia-31m/run.sh
bash runs/pythia-14m/run.sh
bash runs/distilbert/run.sh
```

## Pre-flight checks (đã verified ngày tạo)

- ✓ Train file tồn tại: `benchmarks/llama3-8b-sharegpt-train-t1-s0-8192.jsonl`
- ✓ Cả 4 models load + OV convert + compile + infer OK
- ✓ pad_token issue cho Pythia đã patch trong `vllm/model_executor/prefill_predictor.py`

## Patch áp dụng cho cả 4 runs

Modify `vllm/model_executor/prefill_predictor.py` — set `tokenizer.pad_token = tokenizer.eos_token` nếu None, sync `model.config.pad_token_id`. Cần thiết cho Pythia (GPT-NeoX không có pad mặc định). DistilBERT không bị ảnh hưởng (đã có `[PAD]` sẵn).

## Output structure

Tất cả checkpoints lưu vào `MODEL/results/<run-id>/finetuned/` (default từ trainer.py PathsContainer). Tên folder run-id mirror format của OPT baseline trong `train.sh`.

## Decision tree sau training

1. Nếu **MiniLM-class quality (DistilBERT/Pythia-14m) đạt Tau ≥ baseline−0.05** → ship cái nhỏ nhất đạt yêu cầu, biggest latency win
2. Nếu chỉ **Pythia-70m đạt baseline−0.03** → ship Pythia-70m, ~1.8× speedup
3. Nếu **không model nào đạt baseline−0.05** → pivot sang distillation từ OPT-125M teacher (modify `trainer.py` thêm distill loss)

## Khi train xong, plot Tau vs params

Pareto curve (params trục x, Tau trục y) sẽ identify knee point — model nhỏ nhất mà còn preserve quality. Manual eyeball từ 4 data points đủ tốt, không cần tooling.
