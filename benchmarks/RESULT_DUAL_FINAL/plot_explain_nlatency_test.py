#!/usr/bin/env python3
"""Explain mean N_latency: 3 charts (decomp / ITL distribution / CPU saving).

Chain: mean N_lat = (TTFT + Σ ITL)/N ≈ mean(ITL) → DPS thắng vì ITL nhỏ hơn
       → root cause: mỗi CPU OV dispatch ẩn dưới GPU iteration → tiết kiệm thời
       gian sync trên đường itl của tất cả running tokens.

Output: plot_for_report_final/fig_explain_nlat_{1,2,3}_test.pdf
"""
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
DUAL_TRACE_DIR = os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_TEST"))
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

RATES = [2, 4, 8, 16, 32, 64]
TRACE_RATES = [2, 4, 8, 16, 32]   # r=64 có 0 CPU calls
SCHEDS = [
    ("fcfs",    "FCFS",          "#1f77b4"),
    ("srtf",    "SRTF (Oracle)", "#2ca02c"),
    ("opt-xxx", "Fu et al",      "#ff7f0e"),
    ("dual1.0", "DPS (Ours)",    "#d62728"),
]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_json(sched, rate):
    pat = os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")
    files = sorted(glob.glob(pat))
    return json.load(open(files[-1])) if files else None


def load_trace(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(float(r["t_rel"]), r["event"]) for r in csv.DictReader(f)]


# ============================================================
# Chart 1: Decomposition — TTFT/N vs ΣITL/N (= mean N_latency)
# ============================================================
def chart1_decomposition():
    fig, ax = plt.subplots(figsize=(11, 5))
    n_sched = len(SCHEDS)
    x = np.arange(len(RATES))
    group_w = 0.85
    w = group_w / n_sched

    ttft_part = np.zeros((n_sched, len(RATES)))
    itl_part = np.zeros((n_sched, len(RATES)))
    for i, (sched, *_) in enumerate(SCHEDS):
        for j, r in enumerate(RATES):
            d = load_json(sched, r)
            if not d:
                continue
            ttfts = d["ttfts"]; itls = d["itls"]
            tt = []; ii = []
            for k in range(len(ttfts)):
                n_raw = len(itls[k]) + 1
                if n_raw > 0:
                    tt.append(ttfts[k] / n_raw)
                    ii.append(sum(itls[k]) / n_raw)
            ttft_part[i, j] = float(np.mean(tt)) * 1000 if tt else 0
            itl_part[i, j] = float(np.mean(ii)) * 1000 if ii else 0

    for i, (sched, lbl, color) in enumerate(SCHEDS):
        offset = (i - (n_sched - 1) / 2) * w
        # bottom: TTFT/N (more saturated)
        ax.bar(x + offset, ttft_part[i], w, color=color, edgecolor="white", lw=0.6,
               alpha=1.0, label=f"{lbl} — TTFT/N" if i == 0 else None)
        # top: ITL/N (hatched / lighter)
        ax.bar(x + offset, itl_part[i], w, bottom=ttft_part[i],
               color=color, edgecolor="white", lw=0.6, alpha=0.55,
               hatch="//", label=f"{lbl} — ΣITL/N" if i == 0 else None)

    # custom legend: 4 scheduler colors + 2 segment patterns
    from matplotlib.patches import Patch
    sched_handles = [Patch(facecolor=c, label=lbl) for _, lbl, c in SCHEDS]
    seg_handles = [
        Patch(facecolor="#888", alpha=1.0, label="TTFT / N (solid)"),
        Patch(facecolor="#888", alpha=0.55, hatch="//", label="Σ ITL / N (hatched)"),
    ]
    leg1 = ax.legend(handles=sched_handles, loc="upper left", fontsize=10,
                     frameon=False, ncol=2, title="Scheduler",
                     title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=seg_handles, loc="upper right", fontsize=10,
              frameon=False, title="Segment", title_fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Contribution to mean N_latency (ms/token)")
    ax.set_title("Decomposition: mean N_latency = TTFT/N + Σ ITL/N")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_explain_nlat_1_decomp_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    # Print share %
    print("  Share of ITL/N in total N_lat (%):")
    for i, (sched, lbl, _) in enumerate(SCHEDS):
        shares = [100 * itl_part[i, j] / (itl_part[i, j] + ttft_part[i, j])
                  if (itl_part[i, j] + ttft_part[i, j]) > 0 else 0
                  for j in range(len(RATES))]
        print(f"    {lbl:14s}: " + " ".join(f"r{r}={s:.0f}%" for r, s in zip(RATES, shares)))


# ============================================================
# Chart 2: ITL CDF per scheduler at r=4, 16, 32
# ============================================================
def chart2_itl_cdf():
    sel_rates = [4, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, r in zip(axes, sel_rates):
        for sched, lbl, color in SCHEDS:
            d = load_json(sched, r)
            if not d:
                continue
            all_itls = np.concatenate([np.array(x) for x in d["itls"] if x]) * 1000
            if all_itls.size == 0:
                continue
            sorted_itls = np.sort(all_itls)
            cdf = np.arange(1, len(sorted_itls) + 1) / len(sorted_itls)
            lw = 2.2 if sched == "dual1.0" else 1.5
            ax.plot(sorted_itls, cdf, color=color, lw=lw, label=lbl)

        ax.set_title(f"r = {r} req/s")
        ax.set_xlabel("Inter-token latency (ms)")
        ax.set_xscale("log")
        ax.set_xlim(10, 2000)
        ax.set_ylim(0, 1.0)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", alpha=0.4, color="#888")
        ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="#aaa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # mark p50, p99 reference
        ax.axhline(0.5, color="#bbb", lw=0.6, ls=":")
        ax.axhline(0.99, color="#bbb", lw=0.6, ls=":")

    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(loc="lower right", fontsize=10, frameon=False)
    fig.suptitle("ITL distribution across schedulers — DPS shifts whole CDF left",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_explain_nlat_2_itl_cdf_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ============================================================
# Chart 3: Saving per CPU OV dispatch in DPS (DPS trace only)
# ============================================================
def pair_intervals(trace, ev_start, ev_end):
    starts, out = [], []
    for t, ev in trace:
        if ev == ev_start:
            starts.append(t)
        elif ev == ev_end and starts:
            out.append((starts.pop(0), t))
    return out


def compute_dispatch_savings(tr):
    """Return per-CPU-dispatch overlap (ms) and median GPU sync duration (ms)."""
    cpu_pairs = pair_intervals(tr, "predictor.worker.forward.start",
                                    "predictor.worker.forward.end")
    me_pairs = pair_intervals(tr, "model_executor.start", "model_executor.end")
    gpu_pairs = pair_intervals(tr, "predictor.gpu_sync.start",
                                    "predictor.gpu_sync.end")
    me_sorted = sorted(me_pairs)
    j = 0
    savings_ms = []
    for cs, ce in sorted(cpu_pairs):
        while j < len(me_sorted) and me_sorted[j][1] < cs:
            j += 1
        overlap = 0.0
        k = j
        while k < len(me_sorted) and me_sorted[k][0] <= ce:
            ms, me = me_sorted[k]
            overlap += max(0.0, min(ce, me) - max(cs, ms))
            k += 1
        savings_ms.append(overlap * 1000)
    gpu_sync_ms = [(e - s) * 1000 for s, e in gpu_pairs]
    return savings_ms, gpu_sync_ms


def chart3_dispatch_saving():
    """Bar median saving + IQR per rate, reference line = median GPU sync duration.

    Câu chuyện: mỗi CPU OV dispatch của DPS ẩn dưới 1 GPU iteration, tiết kiệm
    được X ms (≈ median GPU sync duration) — chính khoản này khiến ITL nhỏ hơn.
    """
    rates_used, medians, q25, q75, gpu_refs, n_calls = [], [], [], [], [], []
    for r in TRACE_RATES:
        tr = load_trace(os.path.join(DUAL_TRACE_DIR, f"r{r}", "trace_merged.csv"))
        if not tr:
            continue
        sav, gpu_s = compute_dispatch_savings(tr)
        if not sav:
            continue
        rates_used.append(r)
        n_calls.append(len(sav))
        medians.append(np.median(sav))
        q25.append(np.percentile(sav, 25))
        q75.append(np.percentile(sav, 75))
        gpu_refs.append(np.median(gpu_s) if gpu_s else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rates_used))
    ax.bar(x, medians, width=0.6, color="#d62728", edgecolor="white", lw=0.8,
           alpha=0.85, label="Median overlap of CPU OV with GPU iteration")
    ax.errorbar(x, medians,
                yerr=[np.array(medians) - np.array(q25),
                      np.array(q75) - np.array(medians)],
                fmt="none", ecolor="#8b0000", capsize=5, lw=1.2)
    ax.plot(x, gpu_refs, marker="o", ms=6, color="#9467bd", lw=1.5, ls="--",
            label="Median GPU sync duration (cost avoided)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"r={r}\n(n={n})" for r, n in zip(rates_used, n_calls)])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Time hidden per CPU OV dispatch (ms)")
    ax.set_title("DPS — GPU time saved per CPU OV dispatch via overlap")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_explain_nlat_3_dispatch_saving_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    print("  Per-CPU-dispatch saving (ms, median + IQR) and ref GPU sync:")
    for r, m, lo, hi, g, n in zip(rates_used, medians, q25, q75, gpu_refs, n_calls):
        print(f"    r={r:2d} (n={n:3d}): med={m:5.1f}  IQR=[{lo:.1f},{hi:.1f}]"
              f"  GPU-sync ref={g:.1f}")


def main():
    chart1_decomposition()
    print()
    chart2_itl_cdf()
    print()
    chart3_dispatch_saving()


if __name__ == "__main__":
    main()
