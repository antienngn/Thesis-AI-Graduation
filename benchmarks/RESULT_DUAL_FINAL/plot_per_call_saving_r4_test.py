#!/usr/bin/env python3
"""Per-CPU-call saving at r=4: histogram of GPU forward time hidden by each
CPU predictor call.

Saving per call = portion of CPU forward that overlaps with model_executor.
102 calls at r=4.
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST", "r4", "trace_merged.csv"))
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR = "#d62728"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
})


def load_saving():
    cpu_pairs, me_pairs = [], []
    pend = {"predictor.worker.forward": [], "model_executor": []}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ev, t = r["event"], float(r["t_rel"])
            for k in pend:
                if ev == k + ".start":
                    pend[k].append(t)
                elif ev == k + ".end" and pend[k]:
                    pair = (pend[k].pop(0), t)
                    (cpu_pairs if k == "predictor.worker.forward" else me_pairs).append(pair)

    me_sorted = sorted(me_pairs)
    j = 0
    savings = []
    for cs, ce in sorted(cpu_pairs):
        while j < len(me_sorted) and me_sorted[j][1] < cs:
            j += 1
        overlap = 0.0
        k = j
        while k < len(me_sorted) and me_sorted[k][0] <= ce:
            ms, me = me_sorted[k]
            overlap += max(0.0, min(ce, me) - max(cs, ms))
            k += 1
        savings.append(overlap * 1000)
    return np.array(savings)


def main():
    sav = load_saving()
    n = len(sav)
    med = float(np.median(sav))
    mean = float(sav.mean())
    total = float(sav.sum()) / 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 320, 20)
    counts, edges, _ = ax.hist(sav, bins=bins, color=COLOR, edgecolor="white", lw=1.2,
                                alpha=0.9)

    ax.axvline(med, color="#1f6f1f", ls="--", lw=2.0, label=f"Median = {med:.1f} ms")
    ax.axvline(mean, color="#9467bd", ls="--", lw=2.0, label=f"Mean   = {mean:.1f} ms")

    ax.set_xlabel("GPU forward time hidden per CPU predictor call (ms)")
    ax.set_ylabel("Number of CPU predictor calls")
    ax.set_title("Per-call saving at r = 4 req/s")
    ax.set_xlim(0, max(bins))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, frameon=False)

    summary = f"n = {n} calls   ·   total saving = {total:.2f} s"
    ax.text(0.99, 0.78, summary, transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.4"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_per_call_saving_r4_test.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    print(f"    n={n}, median={med:.1f} ms, mean={mean:.1f} ms, total={total:.2f} s")


if __name__ == "__main__":
    main()
