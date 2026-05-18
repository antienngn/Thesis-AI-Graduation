#!/usr/bin/env python3
# analyze_beat_optxxx.py — Với mỗi rate, liệt kê các config (chunk, T) của
# opt-cpu-async-merged đánh bại opt-xxx trên (TTFT, TPOT, n_latency).

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
METRICS = ["ttft", "tpot", "nlat"]


def load_metrics(p):
    with open(p) as f:
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


sweep = {}
for r in RATES:
    for c in CHUNKS:
        for t in WARMUPS:
            files = sorted(glob(os.path.join(SWEEP, f"r{r}", f"chunk{c}", f"warmup{t}", "*.json")))
            if files:
                sweep[(r, c, t)] = load_metrics(files[-1])

opt_xxx = {}
for f in sorted(glob(os.path.join(SERVE, "vllm-*-opt-xxx-*.json"))):
    m = re.match(r"vllm-([\d.]+)qps", os.path.basename(f))
    if m:
        opt_xxx[int(float(m.group(1)))] = load_metrics(f)


def fmt_pct(merged_v, opt_v):
    if opt_v == 0 or np.isnan(opt_v) or np.isnan(merged_v):
        return "  -  "
    delta_pct = (merged_v - opt_v) / opt_v * 100
    return f"{delta_pct:+5.1f}%"


for r in RATES:
    if r not in opt_xxx:
        continue
    opt = opt_xxx[r]
    print(f"\n{'=' * 110}")
    print(f"RATE = {r}    opt-xxx baseline:  "
          + "  ".join(f"{m}_{s}={opt[f'{m}_{s}']:.1f}"
                      for m in METRICS for s in ["mean"]))
    print("=" * 110)

    # For each config, count number of (metric, stat) combos beaten
    keys = [f"{m}_{s}" for m in METRICS for s in STATS]
    cfg_results = []  # (cfg, n_beat, beat_mask, metrics)
    for c in CHUNKS:
        for t in WARMUPS:
            if (r, c, t) not in sweep:
                continue
            met = sweep[(r, c, t)]
            beat = [met[k] < opt[k] for k in keys]
            cfg_results.append((f"c{c}-T{t}", sum(beat), beat, met))

    cfg_results.sort(key=lambda x: -x[1])

    # Print configs that beat opt-xxx on ALL 9 (metric, stat) combos
    full = [c for c in cfg_results if c[1] == 9]
    print(f"\n  Configs đánh bại opt-xxx trên TẤT CẢ 9 metric/stat: {len(full)}")
    for cfg, n, _, met in full:
        print(f"    {cfg}: " + ", ".join(f"{m}_mean={met[f'{m}_mean']:.1f}" for m in METRICS))

    # Configs that beat opt-xxx on ALL 3 mean metrics
    mean_beat = [c for c in cfg_results
                 if all(c[3][f"{m}_mean"] < opt[f"{m}_mean"] for m in METRICS)]
    print(f"\n  Configs thắng opt-xxx trên CẢ 3 mean (ttft, tpot, nlat): {len(mean_beat)}")
    for cfg, _, _, met in mean_beat:
        print(f"    {cfg}: ttft={fmt_pct(met['ttft_mean'], opt['ttft_mean'])}  "
              f"tpot={fmt_pct(met['tpot_mean'], opt['tpot_mean'])}  "
              f"nlat={fmt_pct(met['nlat_mean'], opt['nlat_mean'])}")

    # Configs that beat opt-xxx on ALL 3 median metrics
    med_beat = [c for c in cfg_results
                if all(c[3][f"{m}_median"] < opt[f"{m}_median"] for m in METRICS)]
    print(f"\n  Configs thắng opt-xxx trên CẢ 3 median: {len(med_beat)}")
    for cfg, _, _, met in med_beat:
        print(f"    {cfg}: ttft={fmt_pct(met['ttft_median'], opt['ttft_median'])}  "
              f"tpot={fmt_pct(met['tpot_median'], opt['tpot_median'])}  "
              f"nlat={fmt_pct(met['nlat_median'], opt['nlat_median'])}")

    # Configs that beat opt-xxx on ALL 3 p99 metrics
    p99_beat = [c for c in cfg_results
                if all(c[3][f"{m}_p99"] < opt[f"{m}_p99"] for m in METRICS)]
    print(f"\n  Configs thắng opt-xxx trên CẢ 3 p99: {len(p99_beat)}")
    for cfg, _, _, met in p99_beat:
        print(f"    {cfg}: ttft={fmt_pct(met['ttft_p99'], opt['ttft_p99'])}  "
              f"tpot={fmt_pct(met['tpot_p99'], opt['tpot_p99'])}  "
              f"nlat={fmt_pct(met['nlat_p99'], opt['nlat_p99'])}")

    # Top-5 by n_beat overall
    print(f"\n  Top-5 configs theo số metric/stat thắng opt-xxx (max=9):")
    print(f"    {'config':<10} | {'n_beat':>6} | {'ttft mean / med / p99':<35} | {'tpot mean / med / p99':<35} | {'nlat mean / med / p99':<35}")
    for cfg, n, beat, met in cfg_results[:5]:
        ttft_str = f"{fmt_pct(met['ttft_mean'], opt['ttft_mean'])} / {fmt_pct(met['ttft_median'], opt['ttft_median'])} / {fmt_pct(met['ttft_p99'], opt['ttft_p99'])}"
        tpot_str = f"{fmt_pct(met['tpot_mean'], opt['tpot_mean'])} / {fmt_pct(met['tpot_median'], opt['tpot_median'])} / {fmt_pct(met['tpot_p99'], opt['tpot_p99'])}"
        nlat_str = f"{fmt_pct(met['nlat_mean'], opt['nlat_mean'])} / {fmt_pct(met['nlat_median'], opt['nlat_median'])} / {fmt_pct(met['nlat_p99'], opt['nlat_p99'])}"
        print(f"    {cfg:<10} | {n:>6} | {ttft_str:<35} | {tpot_str:<35} | {nlat_str:<35}")
