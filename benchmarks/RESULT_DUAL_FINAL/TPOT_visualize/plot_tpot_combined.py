#!/usr/bin/env python3
"""TPOT analysis — bar + line, combining sources:
   - r in {2,4,8,16}     → json_pool (all schedulers)
   - r in {32, 64} DPS   → SERVE_DUAL_TEST (per user request)
   - r in {32, 64} other → json_pool
"""
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE       = os.path.dirname(os.path.abspath(__file__))
POOL       = os.path.normpath(os.path.join(HERE, "..", "json_pool"))
DUAL_TEST  = os.path.normpath(os.path.join(HERE, "..", "..", "SERVE_DUAL_TEST"))
RATES      = [2, 4, 8, 16, 32, 64]

SCHEDS = [
    ("fcfs",    "FCFS",     "#1f77b4", "s", "--"),
    ("srtf",    "SRTF",     "#2ca02c", "D", "--"),
    ("opt-xxx", "Fu et al", "#ff7f0e", "^", "-"),
    ("dual1.0", "DPS",      "#d62728", "o", "-"),
]

plt.rcParams.update({
    "font.size": 11, "axes.labelweight": "bold",
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load(sched, r):
    """Combine source per user request."""
    if sched == "dual1.0" and r in (32, 64):
        files = sorted(glob.glob(f"{DUAL_TEST}/r{r}/vllm-{r}.0qps-*-dual1.0-*.json"))
    else:
        files = sorted(glob.glob(f"{POOL}/vllm-{r}.0qps-*-{sched}-*.json"))
    return json.load(open(files[-1])) if files else None


def collect(metric):
    out = {}
    for key, *_ in SCHEDS:
        out[key] = [load(key, r)[metric] if load(key, r) else np.nan
                    for r in RATES]
    return out


median = collect("median_tpot_ms")
mean   = collect("mean_tpot_ms")

print("Combined TPOT data:")
print(f"{'Rate':>4} | " + " | ".join(f"{lbl:>10s}" for _, lbl, *_ in SCHEDS))
for kind, data in [("median", median), ("mean", mean)]:
    print(f"--- {kind} ---")
    for i, r in enumerate(RATES):
        row = " | ".join(f"{data[k][i]:>10.0f}" for k, *_ in SCHEDS)
        print(f"{r:>4} | {row}")

# ============================================================================
# Plot 1 — Line chart (2 panel: median + mean)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, data, title in zip(axes, [median, mean], ["Median TPOT", "Mean TPOT"]):
    for key, label, c, m, ls in SCHEDS:
        lw = 2.2 if key in ("opt-xxx", "dual1.0") else 1.5
        ax.plot(RATES, data[key], color=c, marker=m, linestyle=ls,
                linewidth=lw, markersize=6, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(300, 1750)
    ax.set_yticks([400, 800, 1200, 1600])
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(fontsize=10, loc="lower right", frameon=False)

plt.tight_layout()
out_line_pdf = os.path.join(HERE, "tpot_line.pdf")
out_line_png = os.path.join(HERE, "tpot_line.png")
plt.savefig(out_line_pdf, bbox_inches="tight")
plt.savefig(out_line_png, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_line_pdf}")
plt.close()

# ============================================================================
# Plot 2 — Bar chart (2 panel: median + mean)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(RATES))
w = 0.20

for ax, data, title in zip(axes, [median, mean], ["Median TPOT", "Mean TPOT"]):
    for i, (key, label, c, _, _) in enumerate(SCHEDS):
        offset = (i - 1.5) * w
        ax.bar(x + offset, data[key], w, color=c, label=label,
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_ylim(0, 1800)
    ax.set_yticks([0, 400, 800, 1200, 1600])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(fontsize=10, loc="upper left", frameon=False, ncol=2)

plt.tight_layout()
out_bar_pdf = os.path.join(HERE, "tpot_bar.pdf")
out_bar_png = os.path.join(HERE, "tpot_bar.png")
plt.savefig(out_bar_pdf, bbox_inches="tight")
plt.savefig(out_bar_png, dpi=200, bbox_inches="tight")
print(f"Saved: {out_bar_pdf}")
plt.close()
