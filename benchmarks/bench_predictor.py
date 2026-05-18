"""bench_predictor.py — đo inference time + Kendall tau cho 1 backend.

Mỗi lần chạy chỉ load 1 backend (openvino HOẶC gpu) để cô lập hoàn toàn:
RAM/threads/CUDA context của backend khác không tồn tại trong process này.
Bash wrapper gọi 2 lần — mỗi backend 1 process.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vllm.config_predictor import PrefillPredictorConfig


def load_dataset(path: str, n: int, llama_tokenizer_name: str) -> Tuple[List[str], List[int]]:
    from transformers import AutoTokenizer
    llama_tok = AutoTokenizer.from_pretrained(llama_tokenizer_name)

    prompts, gens = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if 0 < n <= i:
                break
            d = json.loads(line)
            prompts.append(d["prompt"])
            gens.append(d["generated"])

    enc = llama_tok(gens, add_special_tokens=False)
    true_lens = [len(ids) for ids in enc["input_ids"]]
    return prompts, true_lens


class OVBackend:
    """OpenVINO CPU — num_threads=4, precision điều khiển qua --ov-precision."""
    name = "openvino"

    def __init__(self, cfg, precision: str):
        from vllm.model_executor.openvino_predictor import OpenVINOPredictor
        self.precision = precision
        self.predictor = OpenVINOPredictor(
            model_path=cfg.path,
            tokenizer_name=cfg.pred_model,
            num_labels=cfg.num_labels,
            max_length=cfg.max_length,
            max_batch_size=cfg.max_batch_size,
            num_threads=32,
            inference_precision=precision,
            async_mode=False,
        )
        # Pythia/GPT-NeoX tokenizer không có pad_token mặc định —
        # padding=True trong score() sẽ raise. OPT đã có sẵn nên no-op.
        if self.predictor.tokenizer.pad_token is None:
            self.predictor.tokenizer.pad_token = self.predictor.tokenizer.eos_token
        self.max_length = cfg.max_length

    def score(self, prompts: List[str]) -> List[float]:
        tok = self.predictor.tokenizer
        inps = tok(prompts, max_length=self.max_length,
                   padding=True, truncation=True, return_tensors="pt")
        ov_out = self.predictor.compiled_model(
            [inps["input_ids"].numpy(), inps["attention_mask"].numpy()]
        )
        logits = list(ov_out.values())[0]
        return logits.squeeze(-1).tolist()

    def sync(self):
        pass


class GPUBackend:
    """PyTorch GPU."""
    name = "gpu"

    DTYPE_MAP = {"half": torch.float16, "float": torch.float32, "bfloat16": torch.bfloat16}

    def __init__(self, cfg, gpu_id: int, dtype: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f"cuda:{gpu_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pred_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.path, torch_dtype=self.DTYPE_MAP[dtype]
        ).to(self.device).eval()
        self.max_length = cfg.max_length

    @torch.no_grad()
    def score(self, prompts: List[str]) -> List[float]:
        inps = self.tokenizer(prompts, max_length=self.max_length,
                              padding=True, truncation=True, return_tensors="pt")
        ids = inps["input_ids"].to(self.device)
        mask = inps["attention_mask"].to(self.device)
        logits = self.model(input_ids=ids, attention_mask=mask).logits
        return logits.squeeze(-1).float().cpu().tolist()

    def sync(self):
        torch.cuda.synchronize(self.device)


def measure_inference_time(backend, prompts: List[str], batch_size: int,
                           n_iters: int, warmup: int, seed: int) -> float:
    rng = np.random.default_rng(seed)

    def sample():
        idx = rng.choice(len(prompts), size=batch_size, replace=False)
        return [prompts[i] for i in idx]

    for _ in range(warmup):
        backend.score(sample())
        backend.sync()

    times = []
    for _ in range(n_iters):
        batch = sample()
        backend.sync()
        t0 = time.perf_counter()
        backend.score(batch)
        backend.sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def score_all(backend, prompts: List[str], batch_size: int) -> List[float]:
    out = []
    for i in range(0, len(prompts), batch_size):
        out.extend(backend.score(prompts[i:i + batch_size]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["openvino", "gpu"], required=True)
    p.add_argument("--config", required=True,
                   help="usage_config_ov.json cho OV, usage_config.json cho GPU")
    p.add_argument("--llama-tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--gpu-dtype", choices=["half", "float", "bfloat16"], default="half")
    p.add_argument("--ov-precision", choices=["f32", "f16", "bf16", "int8"], default="int8",
                   help="OpenVINO inference precision (int8 = NNCF weight-only quant)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = PrefillPredictorConfig.from_json(args.config).model
    print(f"[cfg] {args.config} -> path={cfg.path}")
    print(f"[cfg] pred_model={cfg.pred_model} max_length={cfg.max_length}")

    print(f"[load] {args.dataset} num_prompts={args.num_prompts}")
    prompts, true_lens = load_dataset(args.dataset, args.num_prompts, args.llama_tokenizer)
    print(f"[load] {len(prompts)} prompts loaded")

    if args.backend == "openvino":
        print(f"[init] OpenVINO CPU (threads=8, precision={args.ov_precision})")
        backend = OVBackend(cfg, args.ov_precision)
    else:
        print(f"[init] PyTorch GPU (gpu_id={args.gpu_id}, dtype={args.gpu_dtype})")
        backend = GPUBackend(cfg, args.gpu_id, args.gpu_dtype)

    print(f"[lat] {backend.name} bs={args.batch_size} iters={args.n_iters}")
    t_ms = measure_inference_time(backend, prompts, args.batch_size,
                                  args.n_iters, args.warmup, args.seed)
    print(f"      inference_time = {t_ms:.2f} ms")

    print(f"[acc] scoring {len(prompts)} prompts with {backend.name}")
    scores = score_all(backend, prompts, args.batch_size)
    tau, _ = kendalltau(scores, true_lens)
    print(f"      kendall_tau = {tau:.4f}")

    results = {
        "config": vars(args),
        "backend": backend.name,
        "inference_time_ms": t_ms,
        "kendall_tau": float(tau),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
