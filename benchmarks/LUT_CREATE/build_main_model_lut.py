"""Build LUT main model latency từ trace events của bench sweep.

Đọc data/bench_r{2,4,8,16,32,64}/trace_merged.csv (output từ
run_bench_sweep_main_lut.sh), pair start/end của `model_executor.*` events,
extract feature (n_running, n_decode, n_prefill, n_tokens) + latency, rồi
bucket + aggregate qua p50/p95 mỗi cell.

Output:
  data/main_model_lut.json    — LUT chính (cells dict)
  data/main_model_lut.png     — summary plot (marginal latency vs từng feature)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
RATES = [2, 4, 8, 16, 32, 64]

OUT_JSON = DATA_DIR / "main_model_lut.json"
OUT_PNG = DATA_DIR / "main_model_lut.png"

# Bucket edges. Mỗi feature đặt edges để cover state space realistic.
BUCKET_EDGES = {
    "n_running": [0, 4, 8, 16, 32, 64, 128, 256, 1000],
    "n_decode":  [0, 4, 8, 16, 32, 64, 128, 256, 1000],
    "n_prefill": [0, 1, 2, 3, 5, 10, 1000],
    "n_tokens":  [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000],
}

MIN_SAMPLES_PER_CELL = 5  # Cell <5 sample drop


def bucket_of(value, edges):
    """Tìm bucket index (lo, hi) cho value."""
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return (edges[i], edges[i + 1])
    return (edges[-2], edges[-1])  # fallback last bucket


def parse_trace(csv_path):
    """Parse trace_merged.csv → list of (features_dict, lat_ms)."""
    events = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ev = r["event"]
            if not (ev == "model_executor.start" or ev == "model_executor.end"):
                continue
            try:
                t = float(r["t_rel"])
                extra = json.loads(r["extra_json"])
            except (ValueError, json.JSONDecodeError):
                continue
            events.append({"t": t, "ev": ev, "thr": r["thread"], "extra": extra})

    # Pair start/end FIFO per thread
    by_thr = defaultdict(list)
    for e in events:
        by_thr[e["thr"]].append(e)
    pairs = []
    for thr_evs in by_thr.values():
        stack = []
        for e in thr_evs:
            if e["ev"] == "model_executor.start":
                stack.append(e)
            elif e["ev"] == "model_executor.end" and stack:
                s = stack.pop(0)
                lat = e["extra"].get("lat_ms")
                if lat is None:
                    lat = (e["t"] - s["t"]) * 1000.0
                pairs.append({
                    "features": s["extra"],
                    "lat_ms": float(lat),
                })
    return pairs


def aggregate(all_pairs):
    """Group pairs theo bucket key tuple → p50/p95/mean/std."""
    cells = defaultdict(list)
    n_running_missing = 0

    for p in all_pairs:
        f = p["features"]
        n_running = f.get("n_running")
        if n_running is None:
            # Trace cũ chưa có n_running → fallback dùng n_seqs
            n_running_missing += 1
            n_running = f.get("n_seqs", 0)
        key = (
            bucket_of(n_running,         BUCKET_EDGES["n_running"]),
            bucket_of(f.get("n_decode", 0), BUCKET_EDGES["n_decode"]),
            bucket_of(f.get("n_prefill", 0), BUCKET_EDGES["n_prefill"]),
            bucket_of(f.get("n_tokens", 0), BUCKET_EDGES["n_tokens"]),
        )
        cells[key].append(p["lat_ms"])

    if n_running_missing:
        print(f"  WARN: {n_running_missing} events missing n_running "
              f"(dùng n_seqs fallback)")

    result_cells = []
    n_kept = 0
    n_dropped = 0
    for key, lats in cells.items():
        if len(lats) < MIN_SAMPLES_PER_CELL:
            n_dropped += 1
            continue
        result_cells.append({
            "key": {
                "n_running":  list(key[0]),
                "n_decode":   list(key[1]),
                "n_prefill":  list(key[2]),
                "n_tokens":   list(key[3]),
            },
            "n_samples":  len(lats),
            "latency_ms": float(np.median(lats)),
        })
        n_kept += 1

    print(f"  Cells kept (≥{MIN_SAMPLES_PER_CELL} samples): {n_kept}")
    print(f"  Cells dropped (sparse):                {n_dropped}")
    return result_cells


def plot_summary(all_pairs, png_path):
    """Marginal: cho từng feature, plot latency trung bình mỗi bucket."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                              gridspec_kw={"hspace": 0.3, "wspace": 0.25})
    fig.suptitle("LUT main model — marginal latency vs từng feature",
                  fontsize=14, fontweight="bold", y=0.99)

    features = ["n_running", "n_decode", "n_prefill", "n_tokens"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    for ax, feat, color in zip(axes.flat, features, colors):
        edges = BUCKET_EDGES[feat]
        bucket_lats = defaultdict(list)
        for p in all_pairs:
            v = p["features"].get(feat, p["features"].get("n_seqs", 0))
            b = bucket_of(v, edges)
            bucket_lats[b].append(p["lat_ms"])
        keys = sorted(bucket_lats.keys())
        x = list(range(len(keys)))
        median = [float(np.median(bucket_lats[k])) for k in keys]
        n = [len(bucket_lats[k]) for k in keys]
        labels = [f"{k[0]}-{k[1]}" for k in keys]

        ax.plot(x, median, "o-", color=color, linewidth=2, label="median")
        ax2 = ax.twinx()
        ax2.bar(x, n, alpha=0.15, color="gray")
        ax2.set_ylabel("# samples", color="gray", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(feat)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(feat, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig(png_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {png_path}")


def main():
    print("=" * 60)
    print(" Build LUT main model from bench traces")
    print("=" * 60)

    all_pairs = []
    for rate in RATES:
        csv_path = DATA_DIR / f"bench_r{rate}" / "trace_merged.csv"
        if not csv_path.exists():
            print(f"  r={rate}: MISSING {csv_path}")
            continue
        pairs = parse_trace(csv_path)
        print(f"  r={rate}: {len(pairs)} model_executor pairs")
        for p in pairs:
            p["rate"] = rate
        all_pairs.extend(pairs)
    print(f"\nTotal pairs: {len(all_pairs)}")

    if not all_pairs:
        print("ERROR: no data. Run run_bench_sweep_main_lut.sh first.")
        return

    print("\nAggregating into bucket cells...")
    cells = aggregate(all_pairs)

    out = {
        "schema_version": "v4_simple",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "device": "v100_sxm2_fp16",
        "scheduler_used": "opt-xxx",
        "feature_keys": ["n_running", "n_decode", "n_prefill", "n_tokens"],
        "bucket_edges": BUCKET_EDGES,
        "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
        "n_total_pairs": len(all_pairs),
        "cells": cells,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved LUT: {OUT_JSON}")

    plot_summary(all_pairs, OUT_PNG)
    print("\nDone.")


if __name__ == "__main__":
    main()
