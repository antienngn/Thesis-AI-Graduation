#!/usr/bin/env python3
"""Final clean PDFs cho báo cáo: median TTFT, TPOT, N_latency (raw token) vs rate.
Mỗi metric — 1 PDF riêng. Không annotate giá trị, không thêm comment thừa.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
OUT_DIR = os.path.join(HERE, "plot_for_report_final")
os.makedirs(OUT_DIR, exist_ok=True)

RATES = [2, 4, 8, 16, 32, 64]
SCHEDS = [
    ("fcfs",    "FCFS",              "#1f77b4", "s"),   # square xanh
    ("srtf",    "SRTF (Oracle)",     "#2ca02c", "D"),   # diamond xanh lá
    ("opt-xxx", "Fu et al",   "#ff7f0e", "^"),   # triangle cam
    ("dual1.0", "DPS (Ours)",    "#d62728", "o"),   # circle đỏ
]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_one(sched, rate):
    pat = os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")
    files = sorted(glob.glob(pat))
    return json.load(open(files[-1])) if files else None


def median_ttft_ms(d):
    return d.get("median_ttft_ms")


def median_tpot_ms(d):
    """Lấy thẳng field median_tpot_ms từ JSON (server-saved, re-tok denominator)."""
    return d.get("median_tpot_ms")


def mean_nlatency_ms(d):
    """Mean Nlatency — RAW token denominator.
        latency_i  = ttfts[i] + sum(itls[i])
        n_raw_i    = len(itls[i]) + 1
        Nlatency_i = latency_i / n_raw_i
        mean_Nlatency = mean(Nlatency_i) over all requests
    """
    ttfts = d["ttfts"]; itls = d["itls"]
    vals = []
    for i in range(len(ttfts)):
        n_raw = len(itls[i]) + 1
        if n_raw > 0:
            vals.append((ttfts[i] + sum(itls[i])) / n_raw)
    return float(np.mean(vals) * 1000) if vals else None


def collect_all(metric_fn):
    """Trả về (matrix shape (n_sched, n_rate), ylim) cho consistent axis."""
    mat = np.full((len(SCHEDS), len(RATES)), np.nan)
    for i, (sched, *_ ) in enumerate(SCHEDS):
        for j, r in enumerate(RATES):
            d = load_one(sched, r)
            v = metric_fn(d) if d else None
            if v is not None:
                mat[i, j] = v
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    # Mở rộng nhẹ ra ngoài để label không sát biên
    lo = 10 ** (np.floor(np.log10(vmin)) - 0.0)
    hi = 10 ** (np.ceil(np.log10(vmax)) + 0.0)
    return mat, (lo, hi)


def make_line(metric_fn, ylabel, title, fname, unit_div=1000.0):
    """Line plot style ví dụ user gửi: linear X + linear Y, markers có edge,
    title bold trên đầu, legend trên top, light grey grid.
    unit_div: chia metric (ms) cho 1000 → s khi unit_div=1000.
    """
    mat, _ = collect_all(metric_fn)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (sched, lbl, color, marker) in enumerate(SCHEDS):
        ys = [v / unit_div if not np.isnan(v) else None for v in mat[i]]
        lw = 2.2 if sched == "dual1.0" else 1.6
        ms = 9  if sched == "dual1.0" else 7
        ax.plot(RATES, ys, marker=marker, color=color, lw=lw, ms=ms,
                markeredgecolor="black", markeredgewidth=0.6,
                label=lbl)

    ax.set_xlabel("Request rate (req/s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, max(RATES) + 4)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", alpha=0.35, color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=len(SCHEDS),
              frameon=False, columnspacing=2.0,
              handletextpad=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def make_bar(metric_fn, ylabel, title, fname, log_y=None):
    """Grouped bar chart, bars touching within group.
    log_y=None: auto (log if range > 100×, linear otherwise).
    """
    mat, _ = collect_all(metric_fn)
    if log_y is None:
        vmin = float(np.nanmin(mat[mat > 0])) if (mat > 0).any() else 1
        vmax = float(np.nanmax(mat))
        log_y = (vmax / vmin) > 100

    fig, ax = plt.subplots(figsize=(8, 5))
    n_sched = len(SCHEDS)
    x = np.arange(len(RATES))
    group_w = 0.8
    w = group_w / n_sched

    for i, (sched, lbl, color, _) in enumerate(SCHEDS):
        ys = mat[i].copy()
        ys[np.isnan(ys)] = 0
        offset = (i - (n_sched - 1) / 2) * w
        ax.bar(x + offset, ys, w, color=color, edgecolor="white", lw=0.6,
               label=lbl)

    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATES])
    if log_y:
        ax.set_yscale("log")
        vmin = float(np.nanmin(mat[mat > 0]))
        vmax = float(np.nanmax(mat))
        ax.set_ylim(10 ** np.floor(np.log10(vmin)),
                    10 ** np.ceil(np.log10(vmax)))
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(v)))
        ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.5, color="#888")
        ax.grid(True, axis="y", which="minor", linestyle=":",  alpha=0.25, color="#aaa")
    else:
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, color="#888")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95, frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def _fmt(v):
    """Format tick value: 10, 100, 1k, 10k, 100k, ..."""
    if v <= 0: return ""
    if v >= 1e6: return f"{v/1e6:g}M"
    if v >= 1e3: return f"{v/1e3:g}k"
    return f"{v:g}"


def main():
    print(f"Generating final PDFs → {OUT_DIR}")
    metrics = [
        (median_ttft_ms,      "Median TTFT (s)",
         "Median Time-To-First-Token", "median_ttft"),
        (median_tpot_ms,      "TPOT (s/token)",
         "Median Time-Per-Output-Token", "median_tpot"),
        (mean_nlatency_ms,    "Latency (s/token)",
         "Mean Normalized Latency", "mean_nlatency"),
    ]
    for fn, ylbl, title, base in metrics:
        make_line(fn, ylbl, title, f"{base}_line.pdf")
        # bar dùng đơn vị ms gốc (linear bar trông OK với ms)
        make_bar (fn, ylbl.replace("(s)", "(ms)").replace("(s/", "(ms/"),
                  title, f"{base}_bar.pdf")


if __name__ == "__main__":
    main()
