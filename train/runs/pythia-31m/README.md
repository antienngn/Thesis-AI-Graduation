# Pythia-31m-deduped predictor run

## Model
- **HF ID**: `EleutherAI/pythia-31m-deduped`
- **Architecture**: Decoder-only (GPT-NeoX), 6 layers / 256 hidden
- **Params (SequenceClassification)**: ~17.6M (~7× nhỏ hơn OPT-125M)
- **Context length**: 2048 tokens (giữ như OPT)
- **Pretrain**: The Pile (deduped), 300B tokens

## Hypothesis test
**Middle data point** quan trọng cho Pareto sweep.
Combined với 14m và 70m, plot được curve `Tau vs params` → identify knee point.

## Watch-outs
- **pad_token issue**: Đã patch trong `vllm/model_executor/prefill_predictor.py` (set `pad_token = eos_token` tự động).
- **Capacity**: 17M params nằm vùng "có thể đủ hoặc không tùy task" — chính là vùng cần đo

## Run
```bash
bash run.sh
```

## Output location
`MODEL/results/pythia-31m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned/`

## Compare baseline
- OPT-125M ShareGPT/Llama-3-8B: **Tau = 0.52**
- Expected Pythia-31m: Tau drop 0.04-0.10 (≈ 0.42-0.48)
- Latency speedup expected: ~3-4× vs OPT-125M trên CPU
