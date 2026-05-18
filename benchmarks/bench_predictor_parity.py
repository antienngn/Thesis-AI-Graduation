"""bench_predictor_parity.py — verify GPU vs CPU predictor fairness.

Chạy CÙNG prompts trên cả 2 backend, so sánh score per-prompt và Kendall tau.

Khác với bench_predictor.py (so với ground truth riêng cho mỗi backend),
script này so 2 backend với NHAU để verify implementation parity.

Expected output:
  - Pearson correlation ≥ 0.99: implementation tương đương
  - Kendall tau ≥ 0.99: ranking identical
  - max abs diff: bound numerical drift do precision
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vllm.config_predictor import PrefillPredictorConfig


def load_prompts(path: str, n: int) -> List[str]:
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if 0 < n <= i:
                break
            d = json.loads(line)
            prompts.append(d["prompt"])
    return prompts


def score_with_openvino(cfg, prompts: List[str], precision: str) -> List[float]:
    from vllm.model_executor.openvino_predictor import OpenVINOPredictor
    print(f"[OV] Loading predictor (precision={precision})")
    pred = OpenVINOPredictor(
        model_path=cfg.path,
        tokenizer_name=cfg.pred_model,
        num_labels=cfg.num_labels,
        max_length=cfg.max_length,
        max_batch_size=cfg.max_batch_size,
        num_threads=8,
        inference_precision=precision,
        async_mode=False,
    )
    scores = []
    for i in range(0, len(prompts), cfg.max_batch_size):
        batch = prompts[i:i + cfg.max_batch_size]
        inps = pred.tokenizer(batch, max_length=cfg.max_length,
                              padding=True, truncation=True, return_tensors="pt")
        ov_out = pred.compiled_model([inps["input_ids"].numpy(),
                                       inps["attention_mask"].numpy()])
        logits = list(ov_out.values())[0]
        scores.extend(logits.squeeze(-1).tolist())
    return scores


def score_with_hf_gpu(cfg, prompts: List[str], dtype: str, gpu_id: int) -> List[float]:
    """Mô phỏng GPU AUXLLM bằng HF model trực tiếp (cùng head, cùng formula).

    Lưu ý: đây KHÔNG phải vLLM AUXLLM (cần engine + scheduler stack), nhưng
    về mặt mathematical thì tương đương vì cùng checkpoint, cùng head.
    Score = score.weight @ hidden_state[last_non_pad_token]
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    DTYPE_MAP = {"half": torch.float16, "float": torch.float32, "bfloat16": torch.bfloat16}
    print(f"[GPU] Loading predictor (gpu={gpu_id}, dtype={dtype})")
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    tok = AutoTokenizer.from_pretrained(cfg.pred_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.path, torch_dtype=DTYPE_MAP[dtype]
    ).to(device).eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), cfg.max_batch_size):
            batch = prompts[i:i + cfg.max_batch_size]
            inps = tok(batch, max_length=cfg.max_length,
                       padding=True, truncation=True, return_tensors="pt")
            ids = inps["input_ids"].to(device)
            mask = inps["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            scores.extend(logits.squeeze(-1).float().cpu().tolist())
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-ov", required=True, help="usage_config_ov.json")
    p.add_argument("--config-gpu", required=True, help="usage_config.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--gpu-dtype", choices=["half", "float", "bfloat16"], default="half")
    p.add_argument("--ov-precision", choices=["f32", "f16", "bf16", "int8"], default="f16")
    p.add_argument("--output", default="parity_report.json")
    args = p.parse_args()

    cfg_ov = PrefillPredictorConfig.from_json(args.config_ov).model
    cfg_gpu = PrefillPredictorConfig.from_json(args.config_gpu).model

    print(f"[load] {args.dataset} num_prompts={args.num_prompts}")
    prompts = load_prompts(args.dataset, args.num_prompts)
    print(f"[load] {len(prompts)} prompts")

    print("\n=== Scoring with OpenVINO CPU ===")
    ov_scores = score_with_openvino(cfg_ov, prompts, args.ov_precision)
    print(f"[OV] {len(ov_scores)} scores, "
          f"range=[{min(ov_scores):.3f}, {max(ov_scores):.3f}], "
          f"mean={np.mean(ov_scores):.3f}")

    print("\n=== Scoring with PyTorch GPU ===")
    gpu_scores = score_with_hf_gpu(cfg_gpu, prompts, args.gpu_dtype, args.gpu_id)
    print(f"[GPU] {len(gpu_scores)} scores, "
          f"range=[{min(gpu_scores):.3f}, {max(gpu_scores):.3f}], "
          f"mean={np.mean(gpu_scores):.3f}")

    print("\n=== Parity Analysis ===")
    ov = np.array(ov_scores)
    gpu = np.array(gpu_scores)
    abs_diff = np.abs(ov - gpu)
    rel_diff = abs_diff / (np.abs(gpu) + 1e-6)

    pearson_r, _ = pearsonr(ov_scores, gpu_scores)
    kendall_t, _ = kendalltau(ov_scores, gpu_scores)

    print(f"  Pearson correlation: {pearson_r:.6f}")
    print(f"  Kendall tau:         {kendall_t:.6f}")
    print(f"  Max abs diff:        {abs_diff.max():.6f}")
    print(f"  Mean abs diff:       {abs_diff.mean():.6f}")
    print(f"  Median abs diff:     {np.median(abs_diff):.6f}")
    print(f"  Max rel diff:        {rel_diff.max():.4%}")
    print(f"  Mean rel diff:       {rel_diff.mean():.4%}")

    if pearson_r >= 0.99 and kendall_t >= 0.99:
        print("\n  ✓ FAIR: implementations tương đương "
              "(numerical drift ở mức expected)")
    elif pearson_r >= 0.95:
        print("\n  ~ ACCEPTABLE: drift hơi cao, có thể do precision khác biệt")
    else:
        print("\n  ✗ DIVERGENT: implementations KHÔNG khớp — kiểm tra lại")

    report = {
        "config_ov": args.config_ov,
        "config_gpu": args.config_gpu,
        "num_prompts": len(prompts),
        "ov_precision": args.ov_precision,
        "gpu_dtype": args.gpu_dtype,
        "pearson_r": float(pearson_r),
        "kendall_tau": float(kendall_t),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "max_rel_diff": float(rel_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "scores_sample": {
            "ov_first10": ov_scores[:10],
            "gpu_first10": gpu_scores[:10],
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report → {args.output}")


if __name__ == "__main__":
    main()
