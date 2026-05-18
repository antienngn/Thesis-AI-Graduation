# Pythia-14m-deduped predictor run

## Model
- **HF ID**: `EleutherAI/pythia-14m-deduped`
- **Architecture**: Decoder-only (GPT-NeoX), 6 layers / 128 hidden
- **Params (SequenceClassification)**: ~7.6M (~16× nhỏ hơn OPT-125M)
- **Context length**: 2048 tokens (giữ như OPT)
- **Pretrain**: The Pile (deduped), 300B tokens
- Note: nominal "14m" tính cả LM head; SequenceClassification variant remove LM head → effective ~7.6M

## Hypothesis test
**Floor experiment** — model siêu nhỏ có còn học được task không?
Nếu Tau > 0.40 → biết task không quá khó, 31m chắc chắn đủ.
Nếu Tau ~0.30 (gần random) → capacity floor nằm giữa 14m và 31m.

## Watch-outs
- **pad_token issue**: Pythia tokenizer không có pad_token mặc định. Đã patch trong `vllm/model_executor/prefill_predictor.py` để set `pad_token = eos_token` tự động. Nếu chưa patch, training sẽ raise.
- **Underfit risk**: ~7.6M params có thể không đủ. Cân nhắc:
  - Tăng epoch (5 → 10) với early stop
  - Tăng learning rate (2e-5 → 5e-5) — ít risk overfit cho model nhỏ

## Run
```bash
bash run.sh
```

## Output location
`MODEL/results/pythia-14m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned/`

## Compare baseline
- OPT-125M ShareGPT/Llama-3-8B: **Tau = 0.52** (từ `train.sh` line 55)
- Expected Pythia-14m: Tau drop 0.08-0.15 (≈ 0.37-0.44)
- Latency speedup expected: ~5-8× vs OPT-125M trên CPU
