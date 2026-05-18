# Pythia-70m-deduped predictor run

## Model
- **HF ID**: `EleutherAI/pythia-70m-deduped`
- **Architecture**: Decoder-only (GPT-NeoX), 6 layers / 512 hidden
- **Params (SequenceClassification)**: ~44.7M (~3× nhỏ hơn OPT-125M)
- **Context length**: 2048 tokens (giữ như OPT)
- **Pretrain**: The Pile (deduped), 300B tokens

## Hypothesis test
**Most likely match baseline** trong 4 candidates.
Architecture giống OPT (decoder), context full 2048 (no truncation), chỉ shrink size → đây là isolated test of pure size reduction.

**Train đầu tiên trong sweep** — early signal cho cả flow:
- Nếu 70m KHÔNG match baseline → 14m và 31m chắc chắn không → cần pivot (vd distill loss)
- Nếu 70m match → tiếp tục train 31m, 14m để tìm minimum size

## Watch-outs
- **pad_token issue**: Đã patch trong `vllm/model_executor/prefill_predictor.py`.

## Run
```bash
bash run.sh
```

## Output location
`MODEL/results/pythia-70m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned/`

## Compare baseline
- OPT-125M ShareGPT/Llama-3-8B: **Tau = 0.52**
- Expected Pythia-70m: Tau drop 0.01-0.04 (≈ 0.48-0.51)
- Latency speedup expected: ~1.5-2× vs OPT-125M trên CPU
