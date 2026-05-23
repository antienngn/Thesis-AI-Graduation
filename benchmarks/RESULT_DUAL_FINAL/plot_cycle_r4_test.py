#!/usr/bin/env python3
"""2 charts at r=4 — mechanism: DPS giảm predictor load trên GPU.

Chart 1: số lần predictor block GPU per run
Chart 2: tổng thời gian GPU mất cho predictor (với CPU "free" overlay)

Output PNG @300 dpi.
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_TRACE = {
    "opt-xxx": os.path.normpath(os.path.join(HERE, "..", "SERVE_OPTXXX", "r4", "trace_merged.csv")),
    "dual1.0": os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST", "r4", "trace_merged.csv")),
}
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_OPT = "#ff7f0e"
COLOR_DUAL = "#d62728"
COLOR_CPU = "#888888"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
})


def parse_pairs(path):
    """Return dict: key → list of (start,end). Key = event name without .start/.end."""
    pairs = defaultdict(list)
    pending = defaultdict(list)
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


def gpu_predictor(pairs, sched):
    """OPT: predictor.submit. DPS: predictor.gpu_sync (chỉ khi router fallback)."""
    key = "predictor.gpu_sync" if sched == "dual1.0" else "predictor.submit"
    durs = np.array([(e - s) * 1000 for s, e in pairs.get(key, [])])
    return durs


def cpu_predictor(pairs):
    """DPS only: predictor.worker.forward = CPU OV cost (overlapped, off-GPU)."""
    durs = np.array([(e - s) * 1000 for s, e in pairs.get("predictor.worker.forward", [])])
    return durs


def chart1_call_count():
    opt = gpu_predictor(parse_pairs(CSV_TRACE["opt-xxx"]), "opt-xxx")
    dual = gpu_predictor(parse_pairs(CSV_TRACE["dual1.0"]), "dual1.0")
    counts = [len(opt), len(dual)]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = np.arange(2)
    ax.bar(x, counts, width=0.55,
           color=[COLOR_OPT, COLOR_DUAL], edgecolor="white", lw=1.2)
    for xi, c in zip(x, counts):
        ax.text(xi, c + max(counts) * 0.025, f"{c}",
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    pct = (counts[1] - counts[0]) / counts[0] * 100
    y_arrow = max(counts) * 1.18
    ax.annotate("", xy=(1, y_arrow), xytext=(0, y_arrow),
                arrowprops=dict(arrowstyle="->", color="#1f6f1f", lw=1.8))
    ax.text(0.5, y_arrow * 1.04, f"{pct:+.0f}%",
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#1f6f1f")
    ax.set_xticks(x); ax.set_xticklabels(["Fu et al", "DPS (Ours)"])
    ax.set_ylabel("Number of GPU-blocking predictor calls")
    ax.set_title("Predictor calls that block the GPU  —  r = 4 req/s")
    ax.set_ylim(0, max(counts) * 1.35)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_predictor_calls_r4_test.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}  (opt={counts[0]}, dual={counts[1]}, Δ={pct:+.0f}%)")


def chart2_predictor_time():
    """Stacked: bar bottom = GPU predictor time (cost); DPS thêm CPU layer hatched."""
    opt_pairs = parse_pairs(CSV_TRACE["opt-xxx"])
    dual_pairs = parse_pairs(CSV_TRACE["dual1.0"])

    opt_gpu_s = gpu_predictor(opt_pairs, "opt-xxx").sum() / 1000
    dual_gpu_s = gpu_predictor(dual_pairs, "dual1.0").sum() / 1000
    dual_cpu_s = cpu_predictor(dual_pairs).sum() / 1000

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = np.arange(2)
    gpu_vals = [opt_gpu_s, dual_gpu_s]
    cpu_vals = [0, dual_cpu_s]

    ax.bar(x, gpu_vals, width=0.55,
           color=[COLOR_OPT, COLOR_DUAL], edgecolor="white", lw=1.2,
           label="GPU (blocks main model)")
    ax.bar(x, cpu_vals, width=0.55, bottom=gpu_vals,
           color=COLOR_CPU, edgecolor="white", lw=1.2,
           hatch="//", alpha=0.45,
           label="CPU (overlapped, free)")

    for xi, g, c in zip(x, gpu_vals, cpu_vals):
        ax.text(xi, g / 2, f"{g:.2f} s",
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")
        if c > 0:
            ax.text(xi, g + c / 2, f"{c:.2f} s\n(overlapped)",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="#333")

    pct = (gpu_vals[1] - gpu_vals[0]) / gpu_vals[0] * 100
    y_arrow = max(gpu_vals) * 1.5
    ax.annotate("", xy=(1, y_arrow), xytext=(0, y_arrow),
                arrowprops=dict(arrowstyle="->", color="#1f6f1f", lw=1.8))
    ax.text(0.5, y_arrow * 1.04, f"GPU cost {pct:+.0f}%",
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#1f6f1f")

    ax.set_xticks(x); ax.set_xticklabels(["Fu et al", "DPS (Ours)"])
    ax.set_ylabel("Total predictor time across run (s)")
    ax.set_title("Where does the predictor work go?  —  r = 4 req/s")
    top = max(gpu_vals + [g + c for g, c in zip(gpu_vals, cpu_vals)]) * 1.3
    ax.set_ylim(0, top)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_predictor_time_r4_test.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    print(f"    OPT GPU={opt_gpu_s:.2f}s")
    print(f"    DUAL GPU={dual_gpu_s:.2f}s  + CPU={dual_cpu_s:.2f}s (overlapped)")
    print(f"    Δ GPU cost = {pct:+.0f}%")


def main():
    chart1_call_count()
    chart2_predictor_time()


if __name__ == "__main__":
    main()
