#!/usr/bin/env python3
"""TPOT median + mean line chart (4 schedulers, log scale) — same style as
ttft_median_mean.png.

Output: plot_for_report_final/tpot_median_mean_test.{pdf,png}
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
OUT_DIR = os.path.join(HERE, "plot_for_report_final")

RATES = [2, 4, 8, 16, 32, 64]
SCHEDS = [
    ("fcfs",    "FCFS",          "#1f77b4", "s"),
    ("srtf",    "SRTF (Oracle)", "#2ca02c", "D"),
    ("opt-xxx", "Fu et al",      "#ff7f0e", "^"),
    ("dual1.0", "DPS (Ours)",    "#d62728", "o"),
]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(sched, rate):
    files = sorted(glob.glob(os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")))
    return json.load(open(files[-1])) if files else None


def raw_tpots_ms(d):
    if not d:
        return []
    return [sum(x) / len(x) * 1000 for x in d["itls"] if len(x) > 0]


def collect():
    med = {s: [] for s, *_ in SCHEDS}
    mn  = {s: [] for s, *_ in SCHEDS}
    for sched, *_ in SCHEDS:
        for r in RATES:
            vals = raw_tpots_ms(load(sched, r))
            med[sched].append(float(np.median(vals)) if vals else float("nan"))
            mn[sched].append(float(np.mean(vals))   if vals else float("nan"))
    return med, mn


def plot_panel(ax, data, title):
    for sched, lbl, color, marker in SCHEDS:
        ys = data[sched]
        lw = 2.2 if sched == "dual1.0" else 1.5
        ms = 8  if sched == "dual1.0" else 6
        ax.plot(RATES, ys, marker=marker, color=color, lw=lw, ms=ms,
                markeredgecolor="black", markeredgewidth=0.5, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(title)
    ax.set_xticks(RATES); ax.set_xticklabels([str(r) for r in RATES])
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", alpha=0.4, color="#888")
    ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    med, mn = collect()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_panel(ax1, med, "Median TPOT")
    plot_panel(ax2, mn,  "Mean TPOT")
    ax1.legend(fontsize=10, loc="lower right", frameon=False)
    ax2.set_ylabel("")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"tpot_median_mean_test.{ext}")
        plt.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"  ✓ {out}")
    plt.close()

    # Print numbers
    print("\nMedian TPOT (ms/token):")
    print(f"{'rate':>4}", "  ".join(f"{s[0]:>10}" for s in SCHEDS))
    for j, r in enumerate(RATES):
        row = "  ".join(f"{med[s[0]][j]:>10.1f}" for s in SCHEDS)
        print(f"r={r:2d}  {row}")


if __name__ == "__main__":
    main()
