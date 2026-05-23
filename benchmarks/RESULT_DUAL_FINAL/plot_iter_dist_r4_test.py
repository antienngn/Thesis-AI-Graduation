#!/usr/bin/env python3
"""2 separate histograms: per-iteration cycle distribution at r=4.
   - fig_iter_dist_r4_opt_test.png   — Fu et al (opt-xxx)
   - fig_iter_dist_r4_dual_test.png  — DPS (dual1.0)
Same style as fig_per_call_saving_r4_test.png.
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = {
    "opt-xxx": os.path.normpath(os.path.join(HERE, "..", "SERVE_OPTXXX", "r4", "trace_merged.csv")),
    "dual1.0": os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST", "r4", "trace_merged.csv")),
}
OUT_DIR = os.path.join(HERE, "plot_for_report_final")

COLOR_OPT = "#ff7f0e"
COLOR_DUAL = "#d62728"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
})


def cycles_ms(path):
    starts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["event"] == "model_executor.start":
                starts.append(float(r["t_rel"]))
    starts.sort()
    return np.diff(np.array(starts)) * 1000


def plot_one(data, color, scheduler_label, out_name):
    n = len(data)
    med = float(np.median(data))
    mean = float(data.mean())
    total = float(data.sum()) / 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 310, 10)
    ax.hist(data, bins=bins, color=color, edgecolor="white", lw=1.2, alpha=0.9)

    ax.axvline(med, color="#1f6f1f", ls="--", lw=2.0, label=f"Median = {med:.1f} ms")
    ax.axvline(mean, color="#9467bd", ls="--", lw=2.0, label=f"Mean   = {mean:.1f} ms")

    ax.set_xlabel("GPU iteration cycle (ms)")
    ax.set_ylabel("Number of iterations (log scale)")
    ax.set_title(f"Per-iteration cycle  —  {scheduler_label}  —  r = 4 req/s")
    ax.set_xlim(0, max(bins))
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.2, color="#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, frameon=False)

    summary = f"n = {n} iterations   ·   total GPU time = {total:.2f} s"
    ax.text(0.99, 0.78, summary, transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.4"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    print(f"    n={n}  mean={mean:.2f}  med={med:.2f}  "
          f"p90={np.percentile(data,90):.2f}  p99={np.percentile(data,99):.2f}  "
          f"max={data.max():.1f}  total={total:.2f}s")


def main():
    plot_one(cycles_ms(CSV["opt-xxx"]), COLOR_OPT, "Fu et al",
             "fig_iter_dist_r4_opt_test.png")
    plot_one(cycles_ms(CSV["dual1.0"]), COLOR_DUAL, "DPS (Ours)",
             "fig_iter_dist_r4_dual_test.png")


if __name__ == "__main__":
    main()
