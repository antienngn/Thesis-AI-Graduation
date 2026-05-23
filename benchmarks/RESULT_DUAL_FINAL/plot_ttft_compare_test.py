#!/usr/bin/env python3
"""TTFT analysis: 2 charts (bar comparison + predictor latency root cause).

Chart 1: Grouped bar TTFT DPS vs Fu et al, annotated with Δ%
Chart 2: Predictor latency CPU vs GPU (root cause explanation)
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
TRACE = {
    "opt-xxx": os.path.normpath(os.path.join(HERE, "..", "SERVE_OPTXXX", "r4", "trace_merged.csv")),
    "dual1.0": os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST", "r4", "trace_merged.csv")),
}
OUT_DIR = os.path.join(HERE, "plot_for_report_final")

RATES = [2, 4, 8, 16, 32, 64]
COLOR_OPT  = "#ff7f0e"
COLOR_DUAL = "#d62728"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_ttft(sched, rate):
    files = sorted(glob.glob(os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")))
    if not files:
        return float("nan")
    return json.load(open(files[-1])).get("median_ttft_ms", float("nan"))


def parse_pairs(path):
    pairs = defaultdict(list)
    pending = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            ev, t = r["event"], float(r["t_rel"])
            if ev.endswith(".start"):
                pending[ev[:-6]].append(t)
            elif ev.endswith(".end") and pending[ev[:-4]]:
                k = ev[:-4]
                pairs[k].append((pending[k].pop(0), t))
    return pairs


# ── Chart 1: TTFT bar comparison ──────────────────────────────────────────
def make_ttft_bar():
    opt_vals  = [load_ttft("opt-xxx", r) / 1000 for r in RATES]   # → seconds
    dual_vals = [load_ttft("dual1.0", r) / 1000 for r in RATES]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(RATES))
    w = 0.38

    ax.bar(x - w/2, opt_vals,  w, color=COLOR_OPT,  edgecolor="white", lw=0.7, label="Fu et al")
    ax.bar(x + w/2, dual_vals, w, color=COLOR_DUAL, edgecolor="white", lw=0.7, label="DPS (Ours)")

    for xi, ov, dv in zip(x, opt_vals, dual_vals):
        for val, offset in [(ov, -w/2), (dv, w/2)]:
            if val < 1:
                label = f"{val*1000:.0f} ms"
            else:
                label = f"{val:.0f} s"
            ax.text(xi + offset, val * 1.03, label,
                    ha="center", va="bottom", fontsize=8.5)

    # Δ% annotation above each pair
    y_max = max(max(opt_vals), max(dual_vals))
    for j, (ov, dv) in enumerate(zip(opt_vals, dual_vals)):
        pct = (dv - ov) / ov * 100
        color = "#b22222" if pct > 2 else ("#1f6f1f" if pct < -2 else "#666")
        ax.text(x[j], y_max * 1.22, f"{pct:+.0f}%",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Median TTFT (s, log scale)")
    ax.set_title("Median TTFT — DPS vs Fu et al. across request rates")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RATES])
    ax.set_ylim(1e-1, y_max * 3.5)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, color="#888")
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.2, color="#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="lower right", frameon=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"ttft_compare_bar_test.{ext}")
        plt.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"  ✓ {out}")
    plt.close()


# ── Chart 2: Predictor latency CPU vs GPU ─────────────────────────────────
def make_predictor_latency():
    opt_pairs  = parse_pairs(TRACE["opt-xxx"])
    dual_pairs = parse_pairs(TRACE["dual1.0"])

    gpu_durs = np.array([(e-s)*1000 for s,e in opt_pairs.get("predictor.submit", [])])
    cpu_durs = np.array([(e-s)*1000 for s,e in dual_pairs.get("predictor.worker.forward", [])])

    labels = ["GPU predictor\n(Fu et al)", "CPU predictor\n(DPS)"]
    medians = [float(np.median(gpu_durs)), float(np.median(cpu_durs))]
    q25 = [float(np.percentile(gpu_durs, 25)), float(np.percentile(cpu_durs, 25))]
    q75 = [float(np.percentile(gpu_durs, 75)), float(np.percentile(cpu_durs, 75))]
    colors = [COLOR_OPT, COLOR_DUAL]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = np.arange(2)
    ax.bar(x, medians, width=0.55, color=colors, edgecolor="white", lw=1.2)
    ax.errorbar(x, medians,
                yerr=[np.array(medians)-np.array(q25), np.array(q75)-np.array(medians)],
                fmt="none", ecolor="#333", capsize=7, lw=1.4)

    for xi, m in zip(x, medians):
        ax.text(xi, m + max(medians) * 0.04, f"{m:.1f} ms",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ratio = medians[1] / medians[0]
    y_arrow = max(q75) * 1.5
    ax.annotate("", xy=(1, y_arrow), xytext=(0, y_arrow),
                arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.8))
    ax.text(0.5, y_arrow * 1.08, f"{ratio:.1f}× slower",
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#b22222")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Predictor latency (ms)")
    ax.set_title("Predictor latency: CPU vs GPU  —  r = 4 req/s")
    ax.set_ylim(0, max(q75) * 2.2)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"ttft_predictor_latency_test.{ext}")
        plt.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"  ✓ {out}")
    plt.close()

    print(f"    GPU: n={len(gpu_durs)} med={medians[0]:.1f} IQR=[{q25[0]:.1f},{q75[0]:.1f}]")
    print(f"    CPU: n={len(cpu_durs)} med={medians[1]:.1f} IQR=[{q25[1]:.1f},{q75[1]:.1f}]")
    print(f"    Ratio: {ratio:.1f}×")


def main():
    make_ttft_bar()
    make_predictor_latency()


if __name__ == "__main__":
    main()
