"""Benchmark OPT-125M predictor latency on CPU using ShareGPT dataset."""  
import argparse  
import json  
import time  
  
import numpy as np  
import torch  
import torch.nn as nn  
from transformers import AutoModelForSequenceClassification, AutoTokenizer  
from tqdm import tqdm  
  
  
# ---------------------------------------------------------------------------  
# Minimal CPU predictor (no .to("cuda:0"))  
# ---------------------------------------------------------------------------  
  
class PredModelCPU(nn.Module):  
    def __init__(self, pred_model: str, num_labels: int, mtype: str,  
                 max_length: int = 1024, max_batch_size: int = 32,  
                 tokenizer_name: str = None):  
        super().__init__()  
        print(f"[CPU] Loading model: {pred_model}")  
        self.model = AutoModelForSequenceClassification.from_pretrained(  
            pred_model, num_labels=num_labels  
        )  
        self.model.eval()  
        self.tokenizer = AutoTokenizer.from_pretrained(  
            pred_model if tokenizer_name is None else tokenizer_name  
        )  
        self.mtype = mtype  
        self.max_length = max_length  
        self.max_batch_size = max_batch_size  
  
    @torch.inference_mode()  
    def score(self, prompts):  
        ret = []  
        for i in range(0, len(prompts), self.max_batch_size):  
            batch = prompts[i: i + self.max_batch_size]  
            inps = self.tokenizer(  
                batch,  
                max_length=self.max_length,  
                padding=True,  
                truncation=True,  
                return_tensors="pt",  
            )  
            # CPU: no .to("cuda:0")  
            input_ids = inps["input_ids"]  
            attention_mask = inps["attention_mask"]  
  
            if self.mtype == "class":  
                out = self.model(input_ids, attention_mask).argmax(dim=-1)  
            elif self.mtype == "rank":  
                out = self.model(input_ids, attention_mask).logits  
            else:  
                raise ValueError(f"Unknown mtype: {self.mtype}")  
            ret.append(out)  
        return ret  
  
  
# ---------------------------------------------------------------------------  
# Dataset loading (same format as benchmark_serving_real.py)  
# ---------------------------------------------------------------------------  
  
def load_sharegpt_prompts(dataset_path: str, num_prompts: int):  
    """Load prompts from a ShareGPT-format .jsonl file."""  
    prompts = []  
    with open(dataset_path, "r") as f:  
        for line in f:  
            data = json.loads(line)  
            prompts.append(data["prompt"])  
            if len(prompts) >= num_prompts:  
                break  
    if len(prompts) < num_prompts:  
        raise ValueError(  
            f"Dataset has only {len(prompts)} entries, "  
            f"but {num_prompts} were requested."  
        )  
    return prompts  
  
  
# ---------------------------------------------------------------------------  
# Main benchmark  
# ---------------------------------------------------------------------------  
  
def main(args):  
    print(args)  
  
    # Load dataset  
    print(f"Loading {args.num_prompts} prompts from {args.dataset}...")  
    prompts = load_sharegpt_prompts(args.dataset, args.num_prompts)  
    print(f"Loaded {len(prompts)} prompts.")  
  
    # Load predictor on CPU  
    model = PredModelCPU(  
        pred_model=args.pred_model,  
        num_labels=args.num_labels,  
        mtype=args.mtype,  
        max_length=args.max_length,  
        max_batch_size=args.batch_size,  
    )  
  
    # Warmup  
    print("Warming up (3 iterations)...")  
    warmup_prompts = prompts[:args.batch_size]  
    for _ in range(3):  
        model.score(warmup_prompts)  
  
    # Benchmark: measure latency per batch  
    print(f"Benchmarking {args.num_iters} iterations, batch_size={args.batch_size}...")  
    batch_latencies_ms = []  
  
    for _ in tqdm(range(args.num_iters), desc="Iterations"):  
        # Use a fresh batch each iteration (cycle through dataset)  
        start_idx = (_ * args.batch_size) % (len(prompts) - args.batch_size)  
        batch = prompts[start_idx: start_idx + args.batch_size]  
  
        t0 = time.perf_counter()  
        model.score(batch)  
        t1 = time.perf_counter()  
  
        batch_latencies_ms.append((t1 - t0) * 1000.0)  
  
    # Per-token latency: total tokens in batch / batch latency  
    # (approximate: uses max_length as proxy; real token count varies)  
    # Better: measure actual tokenized lengths  
    print("\nTokenizing to get actual token counts...")  
    sample_batch = prompts[:args.batch_size]  
    inps = model.tokenizer(  
        sample_batch,  
        max_length=args.max_length,  
        padding=True,  
        truncation=True,  
        return_tensors="pt",  
    )  
    total_tokens = inps["attention_mask"].sum().item()  
    avg_tokens_per_batch = total_tokens  # total tokens in one batch  
  
    # Results  
    arr = np.array(batch_latencies_ms)  
    print("\n========== CPU Predictor Latency (OPT-125M) ==========")  
    print(f"  Model         : {args.pred_model}")  
    print(f"  Dataset       : {args.dataset}")  
    print(f"  Batch size    : {args.batch_size}")  
    print(f"  Num iters     : {args.num_iters}")  
    print(f"  Avg tokens/batch: {avg_tokens_per_batch:.0f}")  
    print()  
    print(f"  Batch latency (ms):")  
    print(f"    Mean  : {np.mean(arr):.2f}")  
    print(f"    Median: {np.median(arr):.2f}")  
    print(f"    P50   : {np.percentile(arr, 50):.2f}")  
    print(f"    P95   : {np.percentile(arr, 95):.2f}")  
    print(f"    P99   : {np.percentile(arr, 99):.2f}")  
    print(f"    Min   : {np.min(arr):.2f}")  
    print(f"    Max   : {np.max(arr):.2f}")  
    print()  
    per_token = arr / avg_tokens_per_batch  
    print(f"  Per-token latency (ms/token):")  
    print(f"    Mean  : {np.mean(per_token):.4f}")  
    print(f"    P95   : {np.percentile(per_token, 95):.4f}")  
    print(f"    P99   : {np.percentile(per_token, 99):.4f}")  
    print("======================================================")  
  
    if args.output:  
        torch.save({  
            "batch_latencies_ms": batch_latencies_ms,  
            "batch_size": args.batch_size,  
            "num_iters": args.num_iters,  
            "pred_model": args.pred_model,  
            "avg_tokens_per_batch": avg_tokens_per_batch,  
        }, args.output)  
        print(f"Saved raw results to: {args.output}")  
  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(  
        description="Benchmark OPT-125M predictor latency on CPU."  
    )  
    parser.add_argument(  
        "--pred-model", type=str, default="facebook/opt-125m",  
        help="HuggingFace model name or local path for the predictor."  
    )  
    parser.add_argument(  
        "--dataset", type=str, required=True,  
        help="Path to ShareGPT .jsonl dataset file."  
    )  
    parser.add_argument(  
        "--num-prompts", type=int, default=1000,  
        help="Number of prompts to load from the dataset."  
    )  
    parser.add_argument(  
        "--batch-size", type=int, default=32,  
        help="Number of prompts per scoring batch."  
    )  
    parser.add_argument(  
        "--max-length", type=int, default=1024,  
        help="Max token length for tokenizer truncation."  
    )  
    parser.add_argument(  
        "--num-labels", type=int, default=1,  
        help="Number of output labels (1 for rank mode)."  
    )  
    parser.add_argument(  
        "--mtype", type=str, default="rank", choices=["rank", "class"],  
        help="Predictor mode: rank or class."  
    )  
    parser.add_argument(  
        "--num-iters", type=int, default=100,  
        help="Number of benchmark iterations."  
    )  
    parser.add_argument(  
        "--output", type=str, default=None,  
        help="Path to save raw latency results as a .pt file."  
    )  
    args = parser.parse_args()  
    main(args)