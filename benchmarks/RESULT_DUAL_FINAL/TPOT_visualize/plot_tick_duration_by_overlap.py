#!/usr/bin/env python3
"""Per-tick duration grouped by predictor overlap (DPS only).

For each scheduler.tick interval, classify into:
  - "Decode only"       : no predictor activity in this tick
  - "CPU async overlap" : predictor.worker.forward overlaps this tick
  - "GPU sync"          : predictor.gpu_sync occurs within this tick

Plot violin of tick durations per group, side-by-side per rate.
If CPU overlap truly hides predictor cost, the "CPU async overlap"
distribution should match "Decode only", while "GPU sync" should be
shifted upward by the predictor.submit time.

Source: SERVE_DUAL_TEST/r{rate}/trace_merged.csv.
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "..", "SERVE_DUAL_TEST"))
RATES_TO_PLOT = [2, 4, 8, 16]   # rates where CPU is meaningfully used

C_DEC = "#bdbdbd"   # gray  – decode only
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
    """Return (decode, cpu, gpu) lists of tick durations in ms."""
    p = parse(f"{BASE}/r{rate}/trace_merged.csv")
    ticks = sorted(p.get("scheduler.tick", []))
    cpu = sorted(p.get("predictor.worker.forward", []))
    gpu = sorted(p.get("predictor.gpu_sync", []))

    dec_d, cpu_d, gpu_d = [], [], []

    def overlaps_any(ts, te, intervals):
        for s, e in intervals:
            if e < ts: continue
            if s > te: break
            return True
        return False

    for ts, te in ticks:
        dur = (te - ts) * 1000
        has_gpu = overlaps_any(ts, te, gpu)
        has_cpu = overlaps_any(ts, te, cpu)
        if has_gpu:
            gpu_d.append(dur)
        elif has_cpu:
            cpu_d.append(dur)
        else:
            dec_d.append(dur)
    return np.array(dec_d), np.array(cpu_d), np.array(gpu_d)


# Collect per rate
buckets = {r: classify(r) for r in RATES_TO_PLOT}

print(f"{'Rate':>4} | {'Decode':>20} | {'CPU overlap':>22} | {'GPU sync':>20}")
for r in RATES_TO_PLOT:
    dec, cpu, gpu = buckets[r]
    fmt = lambda a: f"n={len(a):>3d}, med={np.median(a):>5.1f}ms" if len(a) else f"n=0"
    print(f"{r:>4} | {fmt(dec):>20} | {fmt(cpu):>22} | {fmt(gpu):>20}")

# ============================================================================
# Plot — violin per group, grouped by rate
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

n_rates = len(RATES_TO_PLOT)
group_w = 0.7
sub_w   = group_w / 3
positions_base = np.arange(n_rates)

def vplot(pos, vals, color):
    if len(vals) == 0:
        return
    parts = ax.violinplot([vals], positions=[pos], widths=sub_w * 0.95,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.55); pc.set_edgecolor("#333")
    ax.boxplot([vals], positions=[pos], widths=sub_w * 0.35,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="white", linewidth=1.6),
               boxprops=dict(facecolor=color, edgecolor="#333"),
               whiskerprops=dict(color="#333"),
               capprops=dict(color="#333"))

for i, r in enumerate(RATES_TO_PLOT):
    dec, cpu, gpu = buckets[r]
    vplot(i - sub_w, dec, C_DEC)
    vplot(i,         cpu, C_CPU)
    vplot(i + sub_w, gpu, C_GPU)

# Legend (proxy patches)
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(facecolor=C_DEC, label="Decode only (no predictor)"),
    mpatches.Patch(facecolor=C_CPU, label="CPU async overlap"),
    mpatches.Patch(facecolor=C_GPU, label="GPU sync"),
], loc="upper left", fontsize=10, frameon=False)

ax.set_xticks(positions_base)
ax.set_xticklabels([f"r = {r}" for r in RATES_TO_PLOT])
ax.set_xlabel("Request rate (req/s)")
ax.set_ylabel("Scheduler tick duration (ms)")

# clip ylim to focus on bulk
ymax = max(np.percentile(buckets[r][j], 95)
           for r in RATES_TO_PLOT for j in range(3) if len(buckets[r][j])) * 1.15
ax.set_ylim(0, ymax)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT_PDF = os.path.join(HERE, "tick_duration_by_overlap.pdf")
OUT_PNG = os.path.join(HERE, "tick_duration_by_overlap.png")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
