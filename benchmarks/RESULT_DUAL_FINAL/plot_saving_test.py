#!/usr/bin/env python3
"""Test plots: CPU OV overlap saving per dispatch + aggregate breakdown.
Output: plot_for_report_final/fig_*_test.pdf
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST"))
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

RATES = [2, 4, 8, 16, 32]   # bỏ 64 (0 CPU calls)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_trace(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(float(r["t_rel"]), r["event"], r.get("extra_json", ""))
                for r in csv.DictReader(f)]


def pair_intervals(trace, ev_start, ev_end):
    starts = []
    out = []
    for t, ev, _ in trace:
        if ev == ev_start:
            starts.append(t)
        elif ev == ev_end and starts:
            out.append((starts.pop(0), t))
    return out


def compute_overlaps(tr):
    """Return list of per-dispatch saving (ms) and totals (seconds)."""
    cpu_pairs = pair_intervals(tr, "predictor.worker.forward.start",
                                    "predictor.worker.forward.end")
    me_pairs  = pair_intervals(tr, "model_executor.start", "model_executor.end")
    gpu_pairs = pair_intervals(tr, "predictor.gpu_sync.start", "predictor.gpu_sync.end")

    me_pairs_sorted = sorted(me_pairs)
    j = 0
    saving_per_dispatch_ms = []
    total_cpu_s = 0.0
    total_overlap_s = 0.0
    for cs, ce in sorted(cpu_pairs):
        cpu_dur = ce - cs
        total_cpu_s += cpu_dur
        # advance me until end >= cs
        while j < len(me_pairs_sorted) and me_pairs_sorted[j][1] < cs:
            j += 1
        overlap = 0.0
        k = j
        while k < len(me_pairs_sorted) and me_pairs_sorted[k][0] <= ce:
            ms, me = me_pairs_sorted[k]
            overlap += max(0.0, min(ce, me) - max(cs, ms))
            k += 1
        saving_per_dispatch_ms.append(overlap * 1000)
        total_overlap_s += overlap

    gpu_sync_durs_ms = [(e - s) * 1000 for s, e in gpu_pairs]
    median_gpu_sync_ms = float(np.median(gpu_sync_durs_ms)) if gpu_sync_durs_ms else 0.0

    return {
        "saving_per_dispatch_ms": saving_per_dispatch_ms,
        "total_cpu_s": total_cpu_s,
        "total_overlap_s": total_overlap_s,
        "n_cpu": len(cpu_pairs),
        "median_gpu_sync_ms": median_gpu_sync_ms,
    }


def gather():
    out = {}
    for r in RATES:
        tr = load_trace(os.path.join(TRACE_DIR, f"r{r}", "trace_merged.csv"))
        out[r] = compute_overlaps(tr) if tr else None
    return out


# ============================================================
# Plan A: Saving per CPU dispatch (median + IQR)
# ============================================================
def fig_planA(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [r for r in RATES if data[r] and data[r]["n_cpu"] > 0]
    medians   = [np.median(data[r]["saving_per_dispatch_ms"]) for r in rates]
    q25       = [np.percentile(data[r]["saving_per_dispatch_ms"], 25) for r in rates]
    q75       = [np.percentile(data[r]["saving_per_dispatch_ms"], 75) for r in rates]
    gpu_sync  = [data[r]["median_gpu_sync_ms"] for r in rates]
    n_calls   = [data[r]["n_cpu"] for r in rates]

    x = np.arange(len(rates))
    bars = ax.bar(x, medians, width=0.6, color="#2ca02c", edgecolor="white", lw=0.8,
                  label="Median overlap of CPU OV with model_executor")
    # IQR whiskers
    ax.errorbar(x, medians,
                yerr=[np.array(medians)-np.array(q25), np.array(q75)-np.array(medians)],
                fmt="none", ecolor="#1f6f1f", capsize=5, lw=1.2)

    # reference line per rate (median gpu_sync) — connect with thin grey line
    ax.plot(x, gpu_sync, marker="o", ms=6, color="#9467bd", lw=1.5,
            ls="--", label="Median GPU sync duration (theoretical max saving)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"r={r}\n(n={n})" for r, n in zip(rates, n_calls)])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Time saved per CPU dispatch (ms)")
    ax.set_title("Effective GPU time saved per CPU OV dispatch in DPS")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", frameon=False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_saving_planA_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ============================================================
# Plan B: Stacked bar (total CPU time + overlap) + violin
# ============================================================
def fig_planB(data):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    rates = [r for r in RATES if data[r] and data[r]["n_cpu"] > 0]

    # LEFT: stacked bar total CPU time = overlapped + non-overlapped
    cpu_total  = np.array([data[r]["total_cpu_s"]     for r in rates])
    overlapped = np.array([data[r]["total_overlap_s"] for r in rates])
    non_over   = cpu_total - overlapped
    x = np.arange(len(rates))
    axL.bar(x, overlapped, color="#2ca02c", edgecolor="white", lw=0.8,
            label="Overlapped with model_executor (effective saving)")
    axL.bar(x, non_over, bottom=overlapped, color="#cccccc", edgecolor="white", lw=0.8,
            label="Non-overlapping (CPU busy, GPU idle)")
    # ratio % label
    for xi, ov, tot in zip(x, overlapped, cpu_total):
        if tot > 0:
            axL.text(xi, tot + max(cpu_total) * 0.02,
                     f"{100*ov/tot:.0f}%", ha="center", fontsize=10, color="#1f6f1f")

    axL.set_xticks(x)
    axL.set_xticklabels([f"r={r}" for r in rates])
    axL.set_xlabel("Request rate (req/s)")
    axL.set_ylabel("Cumulative CPU OV time (s)")
    axL.set_title("Aggregate: effective vs wasted CPU OV time")
    axL.set_axisbelow(True)
    axL.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.legend(fontsize=10, loc="upper right", frameon=False)

    # RIGHT: violin or boxplot of saving per dispatch
    savings = [data[r]["saving_per_dispatch_ms"] for r in rates]
    parts = axR.violinplot(savings, positions=x, showmeans=False, showmedians=True,
                            widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor("#2ca02c"); pc.set_alpha(0.55); pc.set_edgecolor("#1f6f1f")
    parts['cmedians'].set_color("#1f6f1f"); parts['cmedians'].set_linewidth(1.5)

    # overlay reference line at gpu_sync per rate
    gpu_sync_ms = [data[r]["median_gpu_sync_ms"] for r in rates]
    axR.plot(x, gpu_sync_ms, marker="o", ms=6, color="#9467bd", lw=1.5, ls="--",
             label="Median GPU sync duration")
    axR.set_xticks(x)
    axR.set_xticklabels([f"r={r}" for r in rates])
    axR.set_xlabel("Request rate (req/s)")
    axR.set_ylabel("Saving per CPU dispatch (ms)")
    axR.set_title("Distribution: saving per CPU dispatch")
    axR.set_axisbelow(True)
    axR.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.legend(fontsize=10, loc="upper left", frameon=False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_saving_planB_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def main():
    print(f"Loading traces from {TRACE_DIR}")
    data = gather()
    print("\nSummary:")
    for r in RATES:
        d = data[r]
        if d is None or d["n_cpu"] == 0:
            print(f"  r={r}: no CPU dispatches")
            continue
        med = np.median(d["saving_per_dispatch_ms"])
        print(f"  r={r}: n_cpu={d['n_cpu']}, total_cpu={d['total_cpu_s']:.2f}s, "
              f"overlap={d['total_overlap_s']:.2f}s "
              f"({100*d['total_overlap_s']/d['total_cpu_s']:.0f}%), "
              f"median saving/dispatch={med:.1f}ms")
    print()
    fig_planA(data)
    fig_planB(data)


if __name__ == "__main__":
    main()
