#!/usr/bin/env python3
"""Plot TTFT / TPOT / Nlatency: dual<T> scheduler vs các baseline."""
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "json_pool")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

SCHEDULER_ORDER = [
    "fcfs", "sjf", "srtf", "tpt-class10-xxx",
    "opt-xxx", "opt-cpu-warmup2.0", "dual1.0",
]
COLORS = {
    "fcfs":               "#888888",
    "sjf":                "#1f77b4",
    "srtf":               "#9467bd",
    "tpt-class10-xxx":    "#2ca02c",
    "opt-xxx":            "#ff7f0e",
    "opt-cpu-warmup2.0":  "#8c564b",
    "dual1.0":            "#d62728",
}


def load_runs():
    out = []
    for f in sorted(glob.glob(os.path.join(SRC, "vllm-*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(f)
        out.append(d)
    return out


def compute_nlatency(d):
    # Use RAW streamed count = len(itls[i]) + 1 (KHÔNG dùng re-tok output_lens).
    ttfts = d.get("ttfts") or []
    itls = d.get("itls") or []
    if not (ttfts and itls):
        return None, None
    vals = []
    for i in range(len(ttfts)):
        n_raw = len(itls[i]) + 1
        if n_raw > 0:
            vals.append((ttfts[i] + sum(itls[i])) / n_raw)
    if not vals:
        return None, None
    arr = np.asarray(vals) * 1000.0  # → ms/token
    return float(arr.mean()), float(np.percentile(arr, 90))


def compute_tpot(d):
    """TPOT_i = mean(itls[i]) (vì sum/len(itls) khi denominator = n_raw - 1 = len(itls))."""
    itls = d.get("itls") or []
    if not itls:
        return None, None, None
    vals = [sum(itl) / len(itl) for itl in itls if len(itl) > 0]
    if not vals:
        return None, None, None
    arr = np.asarray(vals) * 1000.0
    return (float(arr.mean()),
            float(np.median(arr)),
            float(np.percentile(arr, 99)))


def group_series(runs, value_fn):
    """value_fn(run) -> float | None. Returns {sched: [(qps, val), ...]} sorted."""
    series = defaultdict(list)
    for d in runs:
        v = value_fn(d)
        if v is None:
            continue
        series[d["schedule_type"]].append((float(d["request_rate"]), v))
    for k in series:
        series[k].sort()
    return series


def order_schedulers(seen):
    known = [s for s in SCHEDULER_ORDER if s in seen]
    extra = sorted(s for s in seen if s not in SCHEDULER_ORDER)
    return known + extra


def plot_panel(runs, panel_specs, title, fname, logy=True):
    """panel_specs: list of (subtitle, value_fn, ylabel)."""
    n = len(panel_specs)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), squeeze=False)
    axes = axes[0]
    for ax, (subtitle, vfn, ylabel) in zip(axes, panel_specs):
        series = group_series(runs, vfn)
        order = order_schedulers(series.keys())
        for sched in order:
            points = series[sched]
            xs, ys = zip(*points)
            lw = 2.5 if sched == "dual1.0" else 1.6
            ms = 8 if sched == "dual1.0" else 6
            ax.plot(
                xs, ys, marker="o", linewidth=lw, markersize=ms,
                label=sched, color=COLORS.get(sched, None),
            )
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {SRC}")

    plot_panel(
        runs,
        [
            ("Mean TTFT",   lambda d: d.get("mean_ttft_ms"),   "Mean TTFT (ms)"),
            ("Median TTFT", lambda d: d.get("median_ttft_ms"), "Median TTFT (ms)"),
            ("P99 TTFT",    lambda d: d.get("p99_ttft_ms"),    "P99 TTFT (ms)"),
        ],
        title="TTFT — dual<T> vs baselines",
        fname="ttft.png",
    )

    plot_panel(
        runs,
        [
            ("Mean TPOT",   lambda d: compute_tpot(d)[0], "Mean TPOT (ms/token, raw)"),
            ("Median TPOT", lambda d: compute_tpot(d)[1], "Median TPOT (ms/token, raw)"),
            ("P99 TPOT",    lambda d: compute_tpot(d)[2], "P99 TPOT (ms/token, raw)"),
        ],
        title="TPOT — dual<T> vs baselines  (raw token denominator)",
        fname="tpot.png",
    )

    plot_panel(
        runs,
        [
            ("Mean Nlatency", lambda d: compute_nlatency(d)[0], "Mean Nlatency (ms/token)"),
            ("P90 Nlatency",  lambda d: compute_nlatency(d)[1], "P90 Nlatency (ms/token)"),
        ],
        title="Normalized latency — dual<T> vs baselines",
        fname="nlatency.png",
    )


if __name__ == "__main__":
    main()
