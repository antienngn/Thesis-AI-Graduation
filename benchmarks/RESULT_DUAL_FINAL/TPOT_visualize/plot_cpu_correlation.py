#!/usr/bin/env python3
"""Correlation: % CPU-routed decisions  ↔  TPOT improvement over Fu et al.

If CPU overlap is the cause of TPOT improvement, the two should move
together: when the router uses the CPU more (low/medium rate), the gap
to Fu et al. is larger; when the router falls back to GPU sync entirely
(r=64), the gap collapses.

Data sources (raw-token TPOT):
  - opt-xxx, dual1.0 at r in {32, 64} → SERVE_OPTXXX, SERVE_DUAL_TEST
  - everything else                  → json_pool
Router decisions:                    → SERVE_DUAL_TEST trace_merged.csv
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE       = os.path.dirname(os.path.abspath(__file__))
POOL       = os.path.normpath(os.path.join(HERE, "..", "json_pool"))
DUAL_TEST  = os.path.normpath(os.path.join(HERE, "..", "..", "SERVE_DUAL_TEST"))
OPT_XXX    = os.path.normpath(os.path.join(HERE, "..", "..", "SERVE_OPTXXX"))
RATES      = [2, 4, 8, 16, 32, 64]

plt.rcParams.update({
    "font.size": 11, "axes.labelweight": "bold",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load(sched, r):
    if sched == "dual1.0" and r in (32, 64):
        files = sorted(glob.glob(f"{DUAL_TEST}/r{r}/vllm-{r}.0qps-*-dual1.0-*.json"))
    elif sched == "opt-xxx" and r in (32, 64):
        files = sorted(glob.glob(f"{OPT_XXX}/r{r}/vllm-{r}.0qps-*-opt-xxx-*.json"))
    else:
        files = sorted(glob.glob(f"{POOL}/vllm-{r}.0qps-*-{sched}-*.json"))
    return json.load(open(files[-1])) if files else None


def raw_mean_tpot_ms(d):
    vals = [sum(itl) / len(itl) * 1000 for itl in d["itls"] if len(itl) > 0]
    return float(np.mean(vals))


def cpu_pct(r):
    """% of router decisions routed to CPU at this rate."""
    c = defaultdict(int)
    with open(f"{DUAL_TEST}/r{r}/trace_merged.csv") as f:
        for row in csv.DictReader(f):
            if row["event"].endswith(".start"):
                c[row["event"][:-6]] += 1
    cpu = c["predictor.submit"]
    gpu = c["predictor.gpu_sync"]
    return 100 * cpu / (cpu + gpu)


cpu_p = []
tpot_imp = []
for r in RATES:
    cpu_p.append(cpu_pct(r))
    dual_tpot = raw_mean_tpot_ms(load("dual1.0", r))
    opt_tpot  = raw_mean_tpot_ms(load("opt-xxx", r))
    tpot_imp.append(100 * (opt_tpot - dual_tpot) / opt_tpot)

print(f"{'Rate':>4} | {'%CPU':>6} | {'TPOT improvement vs Fu et al.':>30}")
for i, r in enumerate(RATES):
    print(f"{r:>4} | {cpu_p[i]:>5.1f}% | {tpot_imp[i]:>15.1f}%")

# Pearson r
r_pearson = np.corrcoef(cpu_p, tpot_imp)[0, 1]
print(f"\nPearson r = {r_pearson:.3f}")

# ============================================================================
# Scatter + linear fit
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

cpu_p_np = np.array(cpu_p)
tpot_np  = np.array(tpot_imp)

# Linear fit
slope, intercept = np.polyfit(cpu_p_np, tpot_np, 1)
xs = np.linspace(0, max(cpu_p_np) * 1.1, 100)
ax.plot(xs, slope * xs + intercept, "--", color="#888", linewidth=1.5,
        label=f"Linear fit (r = {r_pearson:.2f})")

# Scatter, colored by rate
norm = plt.Normalize(min(RATES), max(RATES))
cmap = plt.get_cmap("viridis")
for r, x_, y_ in zip(RATES, cpu_p, tpot_imp):
    ax.scatter(x_, y_, s=140, color=cmap(norm(r)),
               edgecolor="black", linewidth=1.0, zorder=3)
    ax.annotate(f"r={r}", (x_, y_),
                xytext=(8, 6), textcoords="offset points",
                fontsize=10, fontweight="bold")

ax.set_xlabel("Router decisions sent to CPU (%)")
ax.set_ylabel("Mean TPOT improvement over Fu et al. (%)")
ax.axhline(0, color="#aaa", linewidth=0.8)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(loc="lower right", fontsize=10, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

OUT_PDF = os.path.join(HERE, "tpot_cpu_correlation.pdf")
OUT_PNG = os.path.join(HERE, "tpot_cpu_correlation.png")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
