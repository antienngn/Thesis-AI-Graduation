#!/usr/bin/env python3
"""Per-CPU-call GPU saving distribution across rates.

For every CPU predictor.worker.forward [cs, ce] in the trace, sum the
duration of model_executor intervals that overlap with [cs, ce].  That
overlap is "GPU forward time hidden under this CPU call" — i.e. how much
of the main-model forward pass was free-running while CPU scored.

Plot: violin per rate (r=2,4,8,16,32,64), with median per-call saving on
top of each violin and total saving (seconds over the run) below.

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


def saving_per_call(path):
    p = parse(path)
    cpu = sorted(p.get("predictor.worker.forward", []))
    me  = sorted(p.get("model_executor", []))
    savings = []
    j_hint = 0
    for cs, ce in cpu:
        ov = 0.0
        # advance hint past ME ending before cs
        while j_hint < len(me) and me[j_hint][1] < cs:
            j_hint += 1
        k = j_hint
        while k < len(me) and me[k][0] <= ce:
            ms, mend = me[k]
            ov += max(0.0, min(ce, mend) - max(cs, ms))
            k += 1
        savings.append(ov * 1000)
    return np.array(savings)


# Collect
data = {r: saving_per_call(f"{BASE}/r{r}/trace_merged.csv") for r in RATES}

print(f"{'Rate':>4} | {'n calls':>7} | {'median (ms)':>11} | "
      f"{'mean (ms)':>9} | {'total (s)':>9}")
for r in RATES:
    d = data[r]
    if len(d) == 0:
        print(f"{r:>4} | {0:>7} | {'-':>11} | {'-':>9} | {'-':>9}")
    else:
        print(f"{r:>4} | {len(d):>7} | {np.median(d):>11.1f} | "
              f"{np.mean(d):>9.1f} | {d.sum()/1000:>9.2f}")

# ============================================================================
# Plot — violin / box per rate
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

rates_with_data = [r for r in RATES if len(data[r]) > 0]
positions = np.arange(len(rates_with_data))

# violin
parts = ax.violinplot(
    [data[r] for r in rates_with_data],
    positions=positions, widths=0.7,
    showmeans=False, showmedians=False, showextrema=False,
)
for pc in parts["bodies"]:
    pc.set_facecolor("#d62728")
    pc.set_alpha(0.45)
    pc.set_edgecolor("#7c1414")

# box overlay
bp = ax.boxplot(
    [data[r] for r in rates_with_data],
    positions=positions, widths=0.18,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="white", linewidth=2),
    boxprops=dict(facecolor="#7c1414", edgecolor="#7c1414"),
    whiskerprops=dict(color="#7c1414"),
    capprops=dict(color="#7c1414"),
)

# Median annotation on top of each violin
for i, r in enumerate(rates_with_data):
    med = float(np.median(data[r]))
    tot = data[r].sum() / 1000
    ax.text(i, max(data[r]) * 1.05,
            f"med {med:.0f} ms\nΣ {tot:.1f} s",
            ha="center", va="bottom", fontsize=9.5,
            color="#222")

# Annotate rates without CPU usage
for r in RATES:
    if r in rates_with_data: continue
    # nothing to draw — but note "0 CPU calls" if you want to keep slot
    pass

ax.set_xticks(positions)
ax.set_xticklabels([f"r = {r}" for r in rates_with_data])
ax.set_xlabel("Request rate (req/s)")
ax.set_ylabel("GPU forward time hidden per CPU call (ms)")

ymax = max(data[r].max() for r in rates_with_data) * 1.18
ax.set_ylim(0, ymax)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT_PDF = os.path.join(HERE, "cpu_saving_per_call.pdf")
OUT_PNG = os.path.join(HERE, "cpu_saving_per_call.png")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
