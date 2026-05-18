# DistilBERT-base predictor run

## Model
- **HF ID**: `distilbert-base-uncased`
- **Architecture**: Encoder-only, 6 layers / 768 hidden
- **Params (SequenceClassification)**: ~67M (~2× nhỏ hơn OPT-125M)
- **Context length**: 512 tokens (architectural limit)
- **Pretrain**: Distilled from BERT-base (BookCorpus + Wiki, MLM)

## Hypothesis test
Switch decoder→encoder + giảm size ~2× có preserve được rank quality không?

## Watch-outs
- **Truncation**: ShareGPT có ~25% prompts > 512 tokens (p75=487, p95=1051) → bị cắt. Có thể hurt rank quality cho task dự đoán generation length.
- **fp16 save** ở cuối trainer (`predictor.model.half()`): DistilBERT support OK, không lo.

## Run
```bash
bash run.sh
```

## Output location
`MODEL/results/distilbert-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned/`

## Compare baseline
- OPT-125M ShareGPT/Llama-3-8B: **Tau = 0.52** (từ `train.sh` line 55)
- Expected DistilBERT: Tau drop 0.02-0.06 (≈ 0.46-0.50)
