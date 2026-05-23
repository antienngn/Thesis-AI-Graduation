"""Profile CPU predictor (OV OPT-125m) latency theo (n_requests, longest_tokens).

QUAN TRỌNG: bucket dựa trên **OPT tokenizer** (cùng tokenizer mà predictor
internally dùng để tokenize batch + padding). Số OPT tokens quyết định:
  - Tensor shape (n_req, longest_opt_tokens) sau padding
  - OV inference cost (forward time tỉ lệ tuyến tính theo seq_len)
KHÔNG dùng Llama tokenizer ở đây — Llama là cho main model, không liên quan
cost của CPU OV predictor.

Phương pháp:
  1. Load OV predictor → có OPT tokenizer
  2. Tokenize ShareGPT prompts bằng OPT tokenizer (truncate đến max_length=2048)
  3. Mỗi cell (n_req, longest_bucket):
       a. Anchor pool = prompts có OPT n_tokens ∈ longest_bucket
       b. Skip cell nếu anchor pool < 5
       c. 50 trials:
           - Pick 1 anchor (longest)
           - Pick (n_req - 1) filler với OPT len ≤ anchor.len
           - Random shuffle vị trí
           - Tokenize batch (padding=True), 1 OV call, đo latency
       d. Bỏ 5 warmup → median latency
  4. Output JSON + heatmap

Output:
  data/cpu_predictor_lut.json    — 2D LUT
  data/cpu_predictor_lut.png     — heatmap + line plot

Router lookup: n_tokens phải tính bằng OPT tokenizer (router cần access
predictor.tokenizer hoặc load OPT tokenizer riêng).
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
BENCH_DIR = HERE.parent
VLLM_ROOT = BENCH_DIR.parent

DATASET = BENCH_DIR / "llama3-8b-sharegpt-test-t1-s0-8192.jsonl"
PRED_CONFIG = (BENCH_DIR /
    "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/"
    "usage_config_ov.json")

OUTPUT_DIR = HERE / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUTPUT_DIR / "cpu_predictor_lut.json"
OUT_PNG = OUTPUT_DIR / "cpu_predictor_lut.png"

# Bucket edges.
# n_requests: số request trong 1 batch route CPU.
# longest_tokens: OPT tokens của prompt dài nhất trong batch (sau truncation).
#   Cap = max_length (2048) → bucket cuối 1536-2048.
N_REQUESTS_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
LONGEST_EDGES = [0, 64, 128, 256, 512, 768, 1024, 1536, 2048]

N_TRIALS = 50
N_WARMUP_TRIALS = 5
MIN_ANCHOR_POOL = 5
RANDOM_SEED = 42


def load_dataset():
    print(f"Loading dataset: {DATASET}")
    prompts = []
    with open(DATASET) as f:
        for line in f:
            try:
                prompts.append(json.loads(line)["prompt"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Loaded {len(prompts)} prompts")
    return prompts


def load_predictor():
    """Load OV predictor — predictor.tokenizer là OPT tokenizer."""
    print(f"Loading OV predictor from {PRED_CONFIG}")
    os.chdir(BENCH_DIR)
    cfg = json.loads(PRED_CONFIG.read_text())["model"]

    import sys
    sys.path.insert(0, str(VLLM_ROOT))
    from vllm.model_executor.openvino_predictor import OpenVINOPredictor

    predictor = OpenVINOPredictor(
        model_path=cfg["path"],
        tokenizer_name=cfg["pred_model"],   # facebook/opt-125m
        num_labels=cfg["num_labels"],
        max_length=cfg["max_length"],        # 2048
        max_batch_size=cfg["max_batch_size"],
        num_threads=cfg.get("num_threads", 32),
        inference_precision=cfg.get("inference_precision", "f16"),
        async_mode=False,
    )
    print(f"  Predictor ready (OPT tokenizer, max_length={predictor.max_length})")
    return predictor


def tokenize_with_opt(prompts, predictor):
    """Tokenize tất cả prompts bằng OPT tokenizer (TRUNCATE đến max_length).

    Returns list of (prompt, opt_n_tokens_after_truncation).
    """
    print(f"Tokenizing {len(prompts)} prompts với OPT tokenizer "
          f"(max_length={predictor.max_length})...")
    out = []
    raw_lens = []
    truncated_count = 0
    for p in prompts:
        # raw = không truncate, để stats
        raw_ids = predictor.tokenizer(p, add_special_tokens=True)["input_ids"]
        raw_n = len(raw_ids)
        raw_lens.append(raw_n)
        # actual = sau truncation tới max_length (đây là cái OV thực sự thấy)
        n = min(raw_n, predictor.max_length)
        if raw_n > predictor.max_length:
            truncated_count += 1
        out.append((p, n))

    lens = [n for _, n in out]
    print(f"  OPT raw n_tokens: min={min(raw_lens)}, "
          f"p50={int(np.percentile(raw_lens, 50))}, "
          f"p95={int(np.percentile(raw_lens, 95))}, "
          f"max={max(raw_lens)}")
    print(f"  OPT after truncate (used by predictor): "
          f"min={min(lens)}, p50={int(np.percentile(lens, 50))}, "
          f"p95={int(np.percentile(lens, 95))}, max={max(lens)}")
    print(f"  Truncated to max_length: {truncated_count}/{len(prompts)} prompts")
    return out


def measure_one_call(predictor, batch_prompts):
    """1 OV inference call cho batch → wall-clock ms (KHÔNG tính tokenize)."""
    inps = predictor.tokenizer(
        batch_prompts, max_length=predictor.max_length,
        padding=True, truncation=True, return_tensors="pt",
    )
    ids = inps["input_ids"].numpy()
    mask = inps["attention_mask"].numpy()
    t0 = time.perf_counter()
    _ = predictor.compiled_model([ids, mask])
    return (time.perf_counter() - t0) * 1000.0


def profile_cell(predictor, prompts_with_lens, n_req, longest_lo, longest_hi,
                  rng):
    """Profile 1 cell (n_req, [longest_lo, longest_hi))."""
    anchor_pool = [(p, n) for p, n in prompts_with_lens
                    if longest_lo <= n < longest_hi]
    if len(anchor_pool) < MIN_ANCHOR_POOL:
        return None
    filler_pool_global = [(p, n) for p, n in prompts_with_lens
                           if n < longest_hi]
    if len(filler_pool_global) < 1:
        return None

    lats = []
    actual_longest = []
    total_tokens = []
    for trial in range(N_TRIALS):
        anchor_p, anchor_n = rng.choice(anchor_pool)
        # Filler: chỉ chọn prompts có OPT len <= anchor_n
        filler_candidates = [(p, n) for p, n in filler_pool_global
                              if n <= anchor_n]
        if len(filler_candidates) < 1:
            filler_candidates = [(anchor_p, anchor_n)]
        if len(filler_candidates) >= n_req - 1:
            fillers = rng.sample(filler_candidates, n_req - 1)
        else:
            # Sample with replacement nếu thiếu
            fillers = [rng.choice(filler_candidates) for _ in range(n_req - 1)]
        batch = [anchor_p] + [p for p, _ in fillers]
        batch_lens = [anchor_n] + [n for _, n in fillers]
        rng.shuffle(batch)

        lat = measure_one_call(predictor, batch)
        lats.append(lat)
        actual_longest.append(max(batch_lens))
        total_tokens.append(sum(batch_lens))

    if len(lats) > N_WARMUP_TRIALS:
        lats_k = lats[N_WARMUP_TRIALS:]
        long_k = actual_longest[N_WARMUP_TRIALS:]
        tot_k = total_tokens[N_WARMUP_TRIALS:]
    else:
        lats_k, long_k, tot_k = lats, actual_longest, total_tokens

    return {
        "n_requests": n_req,
        "longest_tokens_lo": longest_lo,
        "longest_tokens_hi": longest_hi,
        "n_trials_kept": len(lats_k),
        "longest_opt_tokens_mean": float(np.mean(long_k)),
        "total_opt_tokens_mean": float(np.mean(tot_k)),
        "latency_ms": float(np.median(lats_k)),
    }


def plot_lut(cells, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              gridspec_kw={"wspace": 0.3})

    n_list = N_REQUESTS_LIST
    L_buckets = list(zip(LONGEST_EDGES[:-1], LONGEST_EDGES[1:]))
    grid = np.full((len(n_list), len(L_buckets)), np.nan)
    for c in cells:
        try:
            i = n_list.index(c["n_requests"])
            j = L_buckets.index((c["longest_tokens_lo"], c["longest_tokens_hi"]))
            grid[i, j] = c["latency_ms"]
        except (ValueError, IndexError):
            continue

    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(L_buckets)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in L_buckets],
                        rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(n_list)))
    ax.set_yticklabels(n_list, fontsize=9)
    ax.set_xlabel("longest_tokens (OPT) bucket")
    ax.set_ylabel("n_requests")
    ax.set_title("LUT CPU — latency (ms, median)", fontweight="bold")
    for i in range(len(n_list)):
        for j in range(len(L_buckets)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > np.nanmax(grid) * 0.5 else "black",
                        fontsize=7)
    plt.colorbar(im, ax=ax, label="latency (ms)")

    ax = axes[1]
    cmap = plt.cm.plasma(np.linspace(0, 1, len(n_list)))
    for i, n in enumerate(n_list):
        xs, ys = [], []
        for j, (lo, hi) in enumerate(L_buckets):
            v = grid[i, j]
            if not np.isnan(v):
                xs.append((lo + hi) / 2)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, "o-", color=cmap[i], label=f"n={n}",
                    linewidth=1.5, markersize=4)
    ax.set_xlabel("longest_tokens (OPT, mid of bucket)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("LUT CPU — slice per n_requests", fontweight="bold")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.savefig(png_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {png_path}")


def main():
    print("=" * 60)
    print(" LUT CPU predictor — 2D (n_requests, OPT longest_tokens)")
    print("=" * 60)
    print(f" N trials/cell: {N_TRIALS} (warmup {N_WARMUP_TRIALS})")
    print(f" n_requests sweep: {N_REQUESTS_LIST}")
    print(f" longest buckets (OPT): {LONGEST_EDGES}")
    print()

    prompts = load_dataset()
    print()

    # Load predictor TRƯỚC để dùng OPT tokenizer của nó
    predictor = load_predictor()
    print()

    prompts_with_lens = tokenize_with_opt(prompts, predictor)
    print()

    rng = random.Random(RANDOM_SEED)
    cells = []
    L_buckets = list(zip(LONGEST_EDGES[:-1], LONGEST_EDGES[1:]))
    n_cells_total = len(N_REQUESTS_LIST) * len(L_buckets)
    n_done = 0

    print(f"Profiling {n_cells_total} cells...")
    for n_req in N_REQUESTS_LIST:
        for longest_lo, longest_hi in L_buckets:
            n_done += 1
            r = profile_cell(predictor, prompts_with_lens, n_req,
                              longest_lo, longest_hi, rng)
            if r is None:
                print(f"  [{n_done:>3}/{n_cells_total}] "
                      f"n={n_req:>3} L=[{longest_lo:>4}-{longest_hi:>4}) "
                      f"SKIP (sparse anchor pool)")
                continue
            cells.append(r)
            print(f"  [{n_done:>3}/{n_cells_total}] "
                  f"n={n_req:>3} L=[{longest_lo:>4}-{longest_hi:>4}) "
                  f"lat={r['latency_ms']:>7.1f}ms "
                  f"longest_actual_mean={r['longest_opt_tokens_mean']:>5.0f}")

    out = {
        "schema_version": "v4_simple",
        "model": "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32",
        "device": "cpu_ov_avx512",
        "dataset": str(DATASET.name),
        "tokenizer_used_for_bucketing": "facebook/opt-125m",
        "max_length": predictor.max_length,
        "n_trials_per_cell": N_TRIALS,
        "n_warmup_trials": N_WARMUP_TRIALS,
        "bucket_edges": {
            "n_requests": N_REQUESTS_LIST,
            "longest_tokens": LONGEST_EDGES,
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved LUT: {OUT_JSON}  ({len(cells)}/{n_cells_total} cells)")

    plot_lut(cells, OUT_PNG)
    print("\nDone.")


if __name__ == "__main__":
    main()
