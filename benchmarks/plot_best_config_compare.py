#!/usr/bin/env python3
# plot_best_config_compare.py — Vẽ best merged (chunk, T) per (rate, stat)
# so với baseline trong SERVE_RES.

import json
import os
import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(BENCH_DIR, "TEMP_RES_ASYNC_MERGE_SWEEP")
SERVE = os.path.join(BENCH_DIR, "SERVE_RES")
RATES = [2, 4, 8, 16, 32, 64]
CHUNKS = [1, 2, 4, 8, 16, 32]
WARMUPS = [1, 2, 3, 4, 5]
STATS = ["mean", "median", "p99"]
BASELINES = ["opt-xxx", "sjf", "srtf", "fcfs", "tpt-class10-xxx", "opt-cpu-warmup2.0"]


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

baselines = {}
for f in sorted(glob(os.path.join(SERVE, "*.json"))):
    name = os.path.basename(f)
    m = re.match(r"vllm-([\d.]+)qps-cv[\d.]+-Meta-Llama-3-8B-Instruct-(.+?)-\d{8}-\d{6}\.json", name)
    if not m:
        continue
    rate = int(float(m.group(1)))
    sched = m.group(2)
    baselines.setdefault(sched, {})[rate] = load_metrics(f)


def composite_pool(stat, rate):
    pool = {}
    for (rr, c, t), met in sweep.items():
        if rr == r:
            pool[f"merged-c{c}-T{t}"] = met
    for sched in BASELINES:
        if rate in baselines.get(sched, {}):
            pool[sched] = baselines[sched][rate]
    if not pool:
        return None, None
    keys = [f"ttft_{stat}", f"tpot_{stat}", f"nlat_{stat}"]
    arr = np.array([[pool[k][kk] for kk in keys] for k in pool])
    cmin = arr.min(axis=0); cmax = arr.max(axis=0)
    span = np.where(cmax - cmin > 0, cmax - cmin, 1.0)
    norm = (arr - cmin) / span
    score = norm.sum(axis=1)
    return pool, dict(zip(pool.keys(), score))


# ============================================================
# Plot 1 — composite score bars: best-merged vs each baseline per rate
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), squeeze=False)
axes = axes[0]
for j, stat in enumerate(STATS):
    ax = axes[j]
    rows = []
    for r in RATES:
        pool, scores = composite_pool(stat, r)
        if scores is None:
            continue
        merged_only = [(k, v) for k, v in scores.items() if k.startswith("merged-")]
        best = min(merged_only, key=lambda x: x[1])
        rows.append((r, best[0], best[1], scores))
    # Bar groups: x = rate, bars = [best-merged, opt-xxx, sjf, srtf, fcfs, tpt, opt-warmup2]
    series = ["best-merged"] + BASELINES
    width = 0.11
    x = np.arange(len(rows))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(series):
        ys = []
        for r, best_name, best_score, scs in rows:
            if name == "best-merged":
                ys.append(best_score)
            else:
                ys.append(scs.get(name, np.nan))
        ax.bar(x + (i - len(series) / 2) * width, ys, width,
               label=name, color=cmap(i))
    # Annotate best-merged with config
    for i, (r, best_name, best_score, _) in enumerate(rows):
        m = re.match(r"merged-c(\d+)-T(\d+)", best_name)
        ax.annotate(f"c{m.group(1)}T{m.group(2)}",
                    (x[i] - len(series) / 2 * width, best_score),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7, rotation=30)
    ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r, *_ in rows])
    ax.set_ylabel(f"composite score ({stat})  [lower=better]")
    ax.set_title(f"stat = {stat}")
    ax.grid(True, alpha=0.3, axis="y")
    if j == 0:
        ax.legend(fontsize=7, ncol=2, loc="upper left")

fig.suptitle("Composite score per rate — best-merged (annotated chunk/T) vs baselines",
             fontsize=14)
fig.tight_layout()
out = os.path.join(SWEEP, "best_config_compare_bars.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")


# ============================================================
# Plot 2 — heatmap: rows = rate, cols = (chunk, T) flat list, value = composite score per stat
# Đơn giản hóa: 3 panel (mean/median/p99) heatmap rate × (config_idx)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(13, 12), squeeze=False)
axes = axes[:, 0]
config_keys = [f"merged-c{c}-T{t}" for c in CHUNKS for t in WARMUPS]
for j, stat in enumerate(STATS):
    ax = axes[j]
    mat = np.full((len(RATES), len(config_keys)), np.nan)
    for ri, r in enumerate(RATES):
        pool, scores = composite_pool(stat, r)
        if scores is None:
            continue
        for ki, k in enumerate(config_keys):
            mat[ri, ki] = scores.get(k, np.nan)
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(RATES))); ax.set_yticklabels([f"r={r}" for r in RATES])
    ax.set_xticks(range(len(config_keys)))
    ax.set_xticklabels([k.replace("merged-", "") for k in config_keys],
                       rotation=70, fontsize=7)
    ax.set_title(f"composite score ({stat}) — lower=better; ★ = best per rate")
    fig.colorbar(im, ax=ax, fraction=0.025)
    # Mark best per rate
    for ri in range(len(RATES)):
        if np.all(np.isnan(mat[ri])):
            continue
        best_ki = int(np.nanargmin(mat[ri]))
        ax.text(best_ki, ri, "★", ha="center", va="center", color="red", fontsize=14)
fig.tight_layout()
out = os.path.join(SWEEP, "best_config_heatmap_score.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
