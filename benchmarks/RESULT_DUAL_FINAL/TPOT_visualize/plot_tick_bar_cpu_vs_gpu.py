#!/usr/bin/env python3
"""Bar chart: scheduler tick duration — CPU async overlap vs GPU sync.

For each rate, group ticks (DPS) by predictor activity that overlaps:
   - CPU async overlap   : predictor.worker.forward intersects the tick
   - GPU sync            : predictor.gpu_sync intersects the tick
Decode-only ticks are excluded — we focus on predictor-bearing ticks.

Two panels: median and mean per rate.  Empty CPU bar = no CPU calls
at that rate (e.g. r = 64).

Source: SERVE_DUAL_TEST/r{rate}/trace_merged.csv.
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "..", "SERVE_DUAL_TEST"))
RATES = [2, 4, 8, 16, 32, 64]

C_CPU = "#2ca02c"   # green – CPU overlap
C_GPU = "#1f77b4"   # blue  – GPU sync

plt.rcParams.update({
    "font.size": 11, "axes.labelweight": "bold",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def parse(path):
    pending = defaultdict(list)
    pairs   = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            ev, t = r["event"], float(r["t_rel"])
            if ev.endswith(".start"):
                pending[ev[:-6]].append(t)
            elif ev.endswith(".end"):
                k = ev[:-4]
                if pending[k]:
                    pairs[k].append((pending[k].pop(0), t))
    return pairs


def classify(rate):
    p = parse(f"{BASE}/r{rate}/trace_merged.csv")
    ticks = sorted(p.get("scheduler.tick", []))
    cpu   = sorted(p.get("predictor.worker.forward", []))
    gpu   = sorted(p.get("predictor.gpu_sync", []))
    cpu_d, gpu_d = [], []

    def overlaps(ts, te, intervals):
        for s, e in intervals:
            if e < ts: continue
            if s > te: break
            return True
        return False

    for ts, te in ticks:
        dur = (te - ts) * 1000
        if overlaps(ts, te, gpu):
            gpu_d.append(dur)
        elif overlaps(ts, te, cpu):
            cpu_d.append(dur)
    return np.array(cpu_d), np.array(gpu_d)


buckets = {r: classify(r) for r in RATES}

def stat(arr, fn):
    return float(fn(arr)) if len(arr) > 0 else 0.0

cpu_med = [stat(buckets[r][0], np.median) for r in RATES]
gpu_med = [stat(buckets[r][1], np.median) for r in RATES]
cpu_mean = [stat(buckets[r][0], np.mean)  for r in RATES]
gpu_mean = [stat(buckets[r][1], np.mean)  for r in RATES]

print(f"{'Rate':>4} | {'CPU med':>8} {'CPU mean':>8} | {'GPU med':>8} {'GPU mean':>8}")
for i, r in enumerate(RATES):
    n_cpu = len(buckets[r][0]); n_gpu = len(buckets[r][1])
    print(f"{r:>4} | {cpu_med[i]:>8.1f} {cpu_mean[i]:>8.1f} "
          f"| {gpu_med[i]:>8.1f} {gpu_mean[i]:>8.1f}   "
          f"(n_cpu={n_cpu}, n_gpu={n_gpu})")

# ============================================================================
# Plot — two panels (median, mean)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(RATES))
w = 0.38

def annotate(ax, bars):
    for b in bars:
        h = b.get_height()
        if h <= 0: continue
        ax.text(b.get_x() + b.get_width()/2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=8.5, color="#333")

for ax, cpu_vals, gpu_vals, title in zip(
    axes, [cpu_med, cpu_mean], [gpu_med, gpu_mean],
    ["Median tick duration", "Mean tick duration"]
):
    b1 = ax.bar(x - w/2, cpu_vals, w, color=C_CPU,
                label="CPU async overlap", edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w/2, gpu_vals, w, color=C_GPU,
                label="GPU sync", edgecolor="white", linewidth=0.6)
    annotate(ax, b1); annotate(ax, b2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Scheduler tick duration (ms)")
    ax.set_title(title, fontweight="bold")
    ymax = max(max(cpu_vals), max(gpu_vals)) * 1.25
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(fontsize=10, loc="upper left", frameon=False)

plt.tight_layout()
OUT_PDF = os.path.join(HERE, "tick_bar_cpu_vs_gpu.pdf")
OUT_PNG = os.path.join(HERE, "tick_bar_cpu_vs_gpu.png")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
