#!/usr/bin/env python3
"""DPS router decisions per rate — stacked bar (CPU async vs GPU sync).

A single router decision = one scheduler tick that meets an unscored batch.
Parsed from SERVE_DUAL_TEST/r{rate}/trace_merged.csv:
  predictor.gpu_sync  → routed to GPU AUX-LLM (sync)
  predictor.submit    → routed to CPU OpenVINO (async, worker.forward)
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

RATES = [2, 4, 8, 16, 32, 64]
C_CPU = "#2ca02c"   # green — CPU async
C_GPU = "#1f77b4"   # blue  — GPU sync

plt.rcParams.update({
    "font.size": 11, "axes.labelweight": "bold",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def count_events(path):
    c = defaultdict(int)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["event"].endswith(".start"):
                c[r["event"][:-6]] += 1
    return c


cpu, gpu = [], []
for r in RATES:
    c = count_events(f"{BASE}/SERVE_DUAL_TEST/r{r}/trace_merged.csv")
    cpu.append(c["predictor.submit"])
    gpu.append(c["predictor.gpu_sync"])

cpu = np.array(cpu)
gpu = np.array(gpu)
total = cpu + gpu
pct_cpu = 100 * cpu / total

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(RATES))
w = 0.55

ax.bar(x, cpu, w, color=C_CPU, label="CPU (async)")
ax.bar(x, gpu, w, bottom=cpu, color=C_GPU, label="GPU (sync)")

# Single annotation: % CPU above each bar
for i, (t, p) in enumerate(zip(total, pct_cpu)):
    ax.text(i, t + max(total) * 0.02, f"{p:.0f}% CPU",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#222")

ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in RATES])
ax.set_xlabel("Request rate (req/s)")
ax.set_ylabel("Router decisions")
ax.set_ylim(0, max(total) * 1.18)
ax.legend(loc="upper right", fontsize=10, frameon=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT_PDF = os.path.join(OUT_DIR, "predictor_usage.pdf")
OUT_PNG = os.path.join(OUT_DIR, "predictor_usage.png")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")

print(f"\n{'Rate':>4} | {'CPU':>5} | {'GPU':>5} | {'Total':>5} | {'%CPU':>5}")
for i, r in enumerate(RATES):
    print(f"{r:>4} | {cpu[i]:>5d} | {gpu[i]:>5d} | {total[i]:>5d} | {pct_cpu[i]:>4.1f}%")
