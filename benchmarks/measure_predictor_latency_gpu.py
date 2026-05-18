"""Benchmark the inference latency of the OPT-125M predictor (PredModel.score())  
on GPU using real prompts from a ShareGPT-format dataset.  
  
Usage:  
    python benchmark_predictor_latency.py \  
        --usage-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \  
        --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl \  
        --num-prompts 500 \  
        --batch-size 32 \  
        --num-iters 10  
"""  
import argparse  
import json  
import sys  
import time  
from pathlib import Path  
  
import numpy as np  
import torch  
from tqdm import tqdm  
  
# Allow imports from the repo root  
sys.path.insert(0, str(Path(__file__).parent.parent))  
  
from vllm.config_predictor import PrefillPredictorConfig  
from vllm.model_executor.prefill_predictor import prefill_predictor_model  
  
  
# ---------------------------------------------------------------------------  
# Dataset loading  
# ---------------------------------------------------------------------------  
  
def load_prompts(dataset_path: str, num_prompts: int) -> list[str]:  
    """Load prompts from a ShareGPT-format .jsonl file."""  
    prompts = []  
    with open(dataset_path) as f:  
        for line in f:  
            data = json.loads(line)  
            prompts.append(data["prompt"])  
            if len(prompts) >= num_prompts:  
                break  
    if len(prompts) < num_prompts:  
        raise ValueError(  
            f"Dataset only has {len(prompts)} entries, "  
            f"but --num-prompts={num_prompts} was requested."  
        )  
    return prompts  
  
  
# ---------------------------------------------------------------------------  
# Main benchmark  
# ---------------------------------------------------------------------------  
  
def main(args: argparse.Namespace):  
    print(args)  
  
    # --- Load predictor config and model ---  
    config = PrefillPredictorConfig.from_json(args.usage_config)  
    mc = config.model  # PrefillModelConfig  
  
    print(f"Loading predictor: {mc.pred_model}  "  
          f"(mtype={mc.mtype}, num_labels={mc.num_labels}, "  
          f"max_length={mc.max_length}, max_batch_size={mc.max_batch_size})")  
  
    # model = prefill_predictor_model(  
    #     pred_model=mc.path if mc.path else mc.pred_model,  
    #     num_labels=mc.num_labels,  
    #     mtype=mc.mtype,  
    #     activation=mc.activation,  
    #     max_length=mc.max_length,  
    #     max_batch_size=args.batch_size,   # override with CLI batch size  
    # )  

    model = prefill_predictor_model(  
        pred_model=config.model.path,         # finetuned path → loads FINETUNED weights  
        num_labels=config.model.num_labels,  
        mtype=config.model.mtype,  
        activation=config.model.activation,  
        max_length=config.model.max_length,  
        max_batch_size=args.batch_size or config.model.max_batch_size,  
        tokenizer_name=config.model.pred_model,  # "facebook/opt-125m" → tokenizer from base model  
    )
    # Move model weights to GPU (score() hardcodes inputs to cuda:0)  
    model.model = model.model.to("cuda:1")  
    model.model.eval()  
    print("Model loaded and moved to cuda:1")  
  
    # --- Load dataset ---  
    all_prompts = load_prompts(args.dataset, args.num_prompts)  
    print(f"Loaded {len(all_prompts)} prompts from {args.dataset}")  
  
    # Build fixed batches of size --batch-size  
    batches = [  
        all_prompts[i: i + args.batch_size]  
        for i in range(0, len(all_prompts), args.batch_size)  
    ]  
    print(f"Total batches: {len(batches)}  (batch_size={args.batch_size})")  
  
    # --- Warmup ---  
    print("Warming up (3 passes)...")  
    for _ in range(3):  
        model.score(batches[0])  
    torch.cuda.synchronize()  
  
    # --- Benchmark ---  
    batch_latencies_ms = []   # wall time per score() call (ms)  
  
    for _ in tqdm(range(args.num_iters), desc="Benchmark iterations"):  
        for batch in batches:  
            torch.cuda.synchronize()  
            t0 = time.perf_counter()  
            model.score(batch)  
            torch.cuda.synchronize()  
            t1 = time.perf_counter()  
            batch_latencies_ms.append((t1 - t0) * 1000.0)  
  
    # Per-prompt latency  
    per_prompt_latencies_ms = [  
        lat / args.batch_size for lat in batch_latencies_ms  
    ]  
  
    # --- Report ---  
    print("\n========== Results ==========")  
    print(f"Predictor model : {mc.pred_model}")  
    print(f"Dataset         : {args.dataset}")  
    print(f"Num prompts     : {args.num_prompts}")  
    print(f"Batch size      : {args.batch_size}")  
    print(f"Num iters       : {args.num_iters}")  
    print(f"Total batches   : {len(batch_latencies_ms)}")  
    print()  
    print("--- Per-batch latency (ms) ---")  
    print(f"  Mean   : {np.mean(batch_latencies_ms):.2f}")  
    print(f"  Median : {np.median(batch_latencies_ms):.2f}")  
    print(f"  P95    : {np.percentile(batch_latencies_ms, 95):.2f}")  
    print(f"  P99    : {np.percentile(batch_latencies_ms, 99):.2f}")  
    print(f"  Min    : {np.min(batch_latencies_ms):.2f}")  
    print(f"  Max    : {np.max(batch_latencies_ms):.2f}")  
    print()  
    print("--- Per-prompt latency (ms) ---")  
    print(f"  Mean   : {np.mean(per_prompt_latencies_ms):.3f}")  
    print(f"  Median : {np.median(per_prompt_latencies_ms):.3f}")  
    print(f"  P95    : {np.percentile(per_prompt_latencies_ms, 95):.3f}")  
    print(f"  P99    : {np.percentile(per_prompt_latencies_ms, 99):.3f}")  
    print()  
    throughput = args.batch_size / (np.mean(batch_latencies_ms) / 1000.0)  
    print(f"Throughput      : {throughput:.1f} prompts/sec")  
  
    # --- Save raw results ---  
    if args.output:  
        torch.save({  
            "batch_latencies_ms": batch_latencies_ms,  
            "per_prompt_latencies_ms": per_prompt_latencies_ms,  
            "args": vars(args),  
        }, args.output)  
        print(f"\nRaw results saved to: {args.output}")  
  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(  
        description="Benchmark OPT-125M predictor latency on GPU."  
    )  
    parser.add_argument(  
        "--usage-config", type=str, required=True,  
        help="Path to usage_config.json produced by train/trainer.py"  
    )  
    parser.add_argument(  
        "--dataset", type=str, required=True,  
        help="Path to ShareGPT-format .jsonl dataset file"  
    )  
    parser.add_argument(  
        "--num-prompts", type=int, default=500,  
        help="Number of prompts to load from the dataset"  
    )  
    parser.add_argument(  
        "--batch-size", type=int, default=32,  
        help="Number of prompts per score() call"  
    )  
    parser.add_argument(  
        "--num-iters", type=int, default=10,  
        help="Number of full passes over all batches"  
    )  
    parser.add_argument(  
        "--output", type=str, default=None,  
        help="Path to save raw latency tensors (.pt file)"  
    )  
    args = parser.parse_args()  
    main(args)