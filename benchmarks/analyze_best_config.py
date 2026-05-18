#!/usr/bin/env python3
# analyze_best_config.py — Tìm cấu hình (chunk, T) tốt nhất per-rate cho
# opt-cpu-async-merged dựa trên composite score, so sánh với baseline trong
# SERVE_RES (opt-xxx, sjf, srtf, fcfs, tpt-class10-xxx, opt-cpu-warmup2.0).

import json
import os
import re
from glob import glob

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(BENCH_DIR, "TEMP_RES_ASYNC_MERGE_SWEEP")
SERVE = os.path.join(BENCH_DIR, "SERVE_RES")
RATES = [2, 4, 8, 16, 32, 64]
CHUNKS = [1, 2, 4, 8, 16, 32]
WARMUPS = [1, 2, 3, 4, 5]
STATS = ["mean", "median", "p99"]


def load_metrics(json_path):
    with open(json_path) as f:
        d = json.load(f)
    nlat = []
    for tt, it, n in zip(d["ttfts"], d["itls"], d["output_lens"]):
        if n > 0:
            nlat.append((tt + sum(it)) * 1000.0 / n)
    nlat = np.array(nlat) if nlat else np.array([np.nan])
    return {
        "ttft_mean": d["mean_ttft_ms"], "ttft_median": d["median_ttft_ms"], "ttft_p99": d["p99_ttft_ms"],
        "tpot_mean": d["mean_tpot_ms"], "tpot_median": d["median_tpot_ms"], "tpot_p99": d["p99_tpot_ms"],
        "nlat_mean": float(np.mean(nlat)), "nlat_median": float(np.median(nlat)), "nlat_p99": float(np.percentile(nlat, 99)),
    }


# ---------- Load sweep cells ----------
sweep = {}  # sweep[(r, c, t)] = metrics dict
for r in RATES:
    for c in CHUNKS:
        for t in WARMUPS:
            files = sorted(glob(os.path.join(SWEEP, f"r{r}", f"chunk{c}", f"warmup{t}", "*.json")))
            if files:
                sweep[(r, c, t)] = load_metrics(files[-1])


# ---------- Load baselines ----------
baselines = {}  # baselines[scheduler][rate] = metrics
for f in sorted(glob(os.path.join(SERVE, "*.json"))):
    name = os.path.basename(f)
    m = re.match(r"vllm-([\d.]+)qps-cv[\d.]+-Meta-Llama-3-8B-Instruct-(.+?)-\d{8}-\d{6}\.json", name)
    if not m:
        continue
    rate = int(float(m.group(1)))
    sched = m.group(2)
    baselines.setdefault(sched, {})[rate] = load_metrics(f)


def composite(metrics_table, stat, rate):
    """Tính composite normalize cho 1 rate, 1 stat.
    metrics_table: dict cấu_hình -> metrics dict
    Trả về dict cấu_hình -> composite_score (lower = better).
    Normalize bằng min-max trên cùng tập cấu hình (sweep + baselines)."""
    keys = [f"ttft_{stat}", f"tpot_{stat}", f"nlat_{stat}"]
    arr = np.array([[metrics_table[c][k] for k in keys] for c in metrics_table])
    cmin = arr.min(axis=0); cmax = arr.max(axis=0)
    span = np.where(cmax - cmin > 0, cmax - cmin, 1.0)
    norm = (arr - cmin) / span
    score = norm.sum(axis=1)
    return dict(zip(metrics_table.keys(), score))


# ---------- Per (rate, stat) ranking ----------
print("=" * 100)
print("BEST (chunk, T) for opt-cpu-async-merged per (rate, stat)")
print("=" * 100)
print()

# Cũng so sánh với baselines: thêm baseline schedulers vào ranking
results = {}  # results[stat][rate] = list of (config_name, score, metrics)

for stat in STATS:
    print(f"\n--- stat = {stat} ---")
    for r in RATES:
        # Build pool: sweep configs + baselines
        pool = {}
        for (rr, c, t), met in sweep.items():
            if rr == r:
                pool[f"merged-c{c}-T{t}"] = met
        for sched, by_rate in baselines.items():
            if r in by_rate:
                pool[sched] = by_rate[r]
        if not pool:
            continue
        scores = composite(pool, stat, r)
        ranked = sorted(scores.items(), key=lambda x: x[1])
        results.setdefault(stat, {})[r] = ranked

        # Lấy merged tốt nhất
        merged_only = [(k, v) for k, v in ranked if k.startswith("merged-")]
        best_merged = merged_only[0] if merged_only else None
        # Đứng thứ mấy trong tổng thể
        merged_rank = next(i for i, (k, _) in enumerate(ranked) if k == best_merged[0])

        keys = [f"ttft_{stat}", f"tpot_{stat}", f"nlat_{stat}"]
        bm_metrics = pool[best_merged[0]]
        print(f"\n  rate={r:>2}: best merged = {best_merged[0]:<18s} (score={best_merged[1]:.3f}, "
              f"rank #{merged_rank+1}/{len(ranked)})")
        print(f"          ttft={bm_metrics['ttft_'+stat]:>8.1f}  "
              f"tpot={bm_metrics['tpot_'+stat]:>7.2f}  "
              f"nlat={bm_metrics['nlat_'+stat]:>7.2f}")
        # Top 3 overall (cả baselines + sweep)
        print(f"          TOP-3 overall:")
        for i, (name, sc) in enumerate(ranked[:3]):
            m = pool[name]
            print(f"            {i+1}. {name:<22s} score={sc:.3f}  "
                  f"ttft={m['ttft_'+stat]:>7.1f}  tpot={m['tpot_'+stat]:>6.2f}  nlat={m['nlat_'+stat]:>6.2f}")


# ---------- Summary table ----------
print("\n" + "=" * 100)
print("SUMMARY TABLE — best merged (chunk, T) per (rate, stat)")
print("=" * 100)
header = f"{'rate':>4} | " + " | ".join(f"{s:^32}" for s in STATS)
print(header); print("-" * len(header))
for r in RATES:
    cells = []
    for stat in STATS:
        ranked = results[stat][r]
        merged_only = [(k, v) for k, v in ranked if k.startswith("merged-")]
        best = merged_only[0]
        # Tách c, T
        m = re.match(r"merged-c(\d+)-T(\d+)", best[0])
        c, t = m.group(1), m.group(2)
        opt_xxx_score = next((s for k, s in ranked if k == "opt-xxx"), None)
        delta = best[1] - (opt_xxx_score if opt_xxx_score is not None else 0)
        cells.append(f"c={c:>2} T={t}  s={best[1]:.2f} (Δopt-xxx={delta:+.2f})")
    print(f"{r:>4} | " + " | ".join(f"{c:^32}" for c in cells))


# ---------- CSV export ----------
out_csv = os.path.join(SWEEP, "best_config_per_rate.csv")
import csv as _csv
with open(out_csv, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["rate", "stat", "best_merged_config", "best_chunk", "best_T",
                "score_merged", "score_opt-xxx", "score_sjf", "score_srtf",
                "score_fcfs", "score_tpt-class10-xxx", "score_opt-cpu-warmup2.0",
                "merged_overall_rank", "n_configs_in_pool",
                f"merged_ttft", f"merged_tpot", f"merged_nlat"])
    for r in RATES:
        for stat in STATS:
            ranked = results[stat][r]
            scores_d = dict(ranked)
            merged_only = [(k, v) for k, v in ranked if k.startswith("merged-")]
            best = merged_only[0]
            m = re.match(r"merged-c(\d+)-T(\d+)", best[0])
            c, t = int(m.group(1)), int(m.group(2))
            # Lấy metrics
            met = sweep[(r, c, t)]
            rank = next(i for i, (k, _) in enumerate(ranked) if k == best[0]) + 1
            w.writerow([r, stat, best[0], c, t, f"{best[1]:.4f}",
                        f"{scores_d.get('opt-xxx', float('nan')):.4f}",
                        f"{scores_d.get('sjf', float('nan')):.4f}",
                        f"{scores_d.get('srtf', float('nan')):.4f}",
                        f"{scores_d.get('fcfs', float('nan')):.4f}",
                        f"{scores_d.get('tpt-class10-xxx', float('nan')):.4f}",
                        f"{scores_d.get('opt-cpu-warmup2.0', float('nan')):.4f}",
                        rank, len(ranked),
                        f"{met['ttft_'+stat]:.2f}",
                        f"{met['tpot_'+stat]:.2f}",
                        f"{met['nlat_'+stat]:.2f}"])
print(f"\nWrote {out_csv}")
