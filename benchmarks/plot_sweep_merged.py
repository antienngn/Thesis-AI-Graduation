#!/usr/bin/env python3
# plot_sweep_merged.py — Phân tích sweep (rate × chunk × warmup) trong
# TEMP_RES_ASYNC_MERGE_SWEEP. Vẽ TTFT / TPOT / N_LATENCY (mean/median/p99)
# theo mọi tổ hợp param.

import json
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "TEMP_RES_ASYNC_MERGE_SWEEP")
RATES = [2, 4, 8, 16, 32, 64]
CHUNKS = [1, 2, 4, 8, 16, 32]
WARMUPS = [1, 2, 3, 4, 5]

# (metric_label, output_subdir)
METRICS = [
    ("ttft_ms", "TTFT"),
    ("tpot_ms", "TPOT"),
    ("n_latency_ms_per_tok", "N_LATENCY"),
]
STATS = ["mean", "median", "p99"]


def load_cell(rate, chunk, warmup):
    """Load 1 sweep cell → dict các stat. None nếu không có dữ liệu."""
    d = os.path.join(ROOT, f"r{rate}", f"chunk{chunk}", f"warmup{warmup}")
    files = sorted(glob(os.path.join(d, "*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)

    # n_latency = e2e_latency / output_len, per-request
    ttfts = data["ttfts"]
    itls = data["itls"]
    out_lens = data["output_lens"]
    n_lat = []
    for ttft, itl, n in zip(ttfts, itls, out_lens):
        if n is None or n <= 0:
            continue
        e2e_ms = (ttft + sum(itl)) * 1000.0  # ttft & itl ở giây trong file gốc?
        # Note: trong benchmark_serving_real, ttft/itl được lưu sec, nhưng
        # mean_ttft_ms = mean(ttfts)*1000 → ttft đang là sec.
        n_lat.append(e2e_ms / n)
    n_lat = np.array(n_lat) if n_lat else np.array([np.nan])

    return {
        "ttft_ms_mean":   data["mean_ttft_ms"],
        "ttft_ms_median": data["median_ttft_ms"],
        "ttft_ms_p99":    data["p99_ttft_ms"],
        "tpot_ms_mean":   data["mean_tpot_ms"],
        "tpot_ms_median": data["median_tpot_ms"],
        "tpot_ms_p99":    data["p99_tpot_ms"],
        "n_latency_ms_per_tok_mean":   float(np.mean(n_lat)),
        "n_latency_ms_per_tok_median": float(np.median(n_lat)),
        "n_latency_ms_per_tok_p99":    float(np.percentile(n_lat, 99)),
    }


# ----- Build 4-D table: [rate, chunk, warmup, metric_stat] -----
print("Loading sweep cells...")
TABLE = {}  # TABLE[(r,c,t)][metric_stat] = value
for r in RATES:
    for c in CHUNKS:
        for t in WARMUPS:
            row = load_cell(r, c, t)
            if row is not None:
                TABLE[(r, c, t)] = row
print(f"Loaded {len(TABLE)} / {len(RATES)*len(CHUNKS)*len(WARMUPS)} cells")


def get(r, c, t, key):
    row = TABLE.get((r, c, t))
    if row is None:
        return np.nan
    return row[key]


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# PLOT TYPE 1 — Heatmap (chunk × warmup) per (rate, stat) — 1 figure / metric
# ============================================================
def plot_heatmap_grid(metric_label, out_dir):
    """6 rates × 3 stats grid. Mỗi cell là heatmap chunk(x) × warmup(y)."""
    fig, axes = plt.subplots(len(RATES), len(STATS),
                             figsize=(4.0 * len(STATS), 3.0 * len(RATES)),
                             squeeze=False)
    fig.suptitle(f"{metric_label} — heatmap (chunk × warmup) per rate × stat",
                 fontsize=14, y=1.0)

    for i, r in enumerate(RATES):
        for j, s in enumerate(STATS):
            ax = axes[i][j]
            key = f"{metric_label}_{s}"
            mat = np.array([[get(r, c, t, key) for c in CHUNKS]
                            for t in WARMUPS])
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(CHUNKS)))
            ax.set_xticklabels(CHUNKS)
            ax.set_yticks(range(len(WARMUPS)))
            ax.set_yticklabels(WARMUPS)
            ax.set_title(f"r={r}  {s}", fontsize=10)
            if i == len(RATES) - 1:
                ax.set_xlabel("chunk_size")
            if j == 0:
                ax.set_ylabel("T_warmup")
            for ti in range(len(WARMUPS)):
                for ci in range(len(CHUNKS)):
                    v = mat[ti, ci]
                    if not np.isnan(v):
                        ax.text(ci, ti, f"{v:.0f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if v > np.nanmean(mat) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(out_dir, f"01_heatmap_grid_{metric_label}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# PLOT TYPE 2 — x=rate, lines=chunk, faceted by warmup
# ============================================================
def plot_rate_x_chunk_facet_warmup(metric_label, stat, out_dir):
    fig, axes = plt.subplots(1, len(WARMUPS),
                             figsize=(4.0 * len(WARMUPS), 4.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    key = f"{metric_label}_{stat}"
    cmap = plt.get_cmap("viridis")
    for j, t in enumerate(WARMUPS):
        ax = axes[j]
        for ci, c in enumerate(CHUNKS):
            ys = [get(r, c, t, key) for r in RATES]
            ax.plot(RATES, ys, marker="o", label=f"chunk={c}",
                    color=cmap(ci / max(1, len(CHUNKS) - 1)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(RATES)
        ax.set_xticklabels(RATES)
        ax.set_xlabel("request_rate")
        if j == 0:
            ax.set_ylabel(f"{metric_label} ({stat})")
        ax.set_title(f"T_warmup={t}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{metric_label} {stat} — rate vs chunk_size, faceted by T_warmup",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"02_rate_x_chunk_facet_warmup__{stat}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# PLOT TYPE 3 — x=rate, lines=warmup, faceted by chunk
# ============================================================
def plot_rate_x_warmup_facet_chunk(metric_label, stat, out_dir):
    fig, axes = plt.subplots(1, len(CHUNKS),
                             figsize=(3.5 * len(CHUNKS), 4.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    key = f"{metric_label}_{stat}"
    cmap = plt.get_cmap("plasma")
    for j, c in enumerate(CHUNKS):
        ax = axes[j]
        for ti, t in enumerate(WARMUPS):
            ys = [get(r, c, t, key) for r in RATES]
            ax.plot(RATES, ys, marker="o", label=f"T={t}",
                    color=cmap(ti / max(1, len(WARMUPS) - 1)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(RATES)
        ax.set_xticklabels(RATES)
        ax.set_xlabel("request_rate")
        if j == 0:
            ax.set_ylabel(f"{metric_label} ({stat})")
        ax.set_title(f"chunk={c}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{metric_label} {stat} — rate vs T_warmup, faceted by chunk_size",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"03_rate_x_warmup_facet_chunk__{stat}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# PLOT TYPE 4 — x=chunk, lines=warmup, faceted by rate
# ============================================================
def plot_chunk_x_warmup_facet_rate(metric_label, stat, out_dir):
    fig, axes = plt.subplots(1, len(RATES),
                             figsize=(3.5 * len(RATES), 4.0),
                             sharey=False, squeeze=False)
    axes = axes[0]
    key = f"{metric_label}_{stat}"
    cmap = plt.get_cmap("plasma")
    for j, r in enumerate(RATES):
        ax = axes[j]
        for ti, t in enumerate(WARMUPS):
            ys = [get(r, c, t, key) for c in CHUNKS]
            ax.plot(CHUNKS, ys, marker="o", label=f"T={t}",
                    color=cmap(ti / max(1, len(WARMUPS) - 1)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(CHUNKS)
        ax.set_xticklabels(CHUNKS)
        ax.set_xlabel("chunk_size")
        if j == 0:
            ax.set_ylabel(f"{metric_label} ({stat})")
        ax.set_title(f"rate={r}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{metric_label} {stat} — chunk_size vs T_warmup, faceted by rate",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"04_chunk_x_warmup_facet_rate__{stat}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# PLOT TYPE 5 — x=warmup, lines=chunk, faceted by rate
# ============================================================
def plot_warmup_x_chunk_facet_rate(metric_label, stat, out_dir):
    fig, axes = plt.subplots(1, len(RATES),
                             figsize=(3.5 * len(RATES), 4.0),
                             sharey=False, squeeze=False)
    axes = axes[0]
    key = f"{metric_label}_{stat}"
    cmap = plt.get_cmap("viridis")
    for j, r in enumerate(RATES):
        ax = axes[j]
        for ci, c in enumerate(CHUNKS):
            ys = [get(r, c, t, key) for t in WARMUPS]
            ax.plot(WARMUPS, ys, marker="o", label=f"chunk={c}",
                    color=cmap(ci / max(1, len(CHUNKS) - 1)))
        ax.set_xticks(WARMUPS)
        ax.set_xlabel("T_warmup")
        if j == 0:
            ax.set_ylabel(f"{metric_label} ({stat})")
        ax.set_title(f"rate={r}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{metric_label} {stat} — T_warmup vs chunk_size, faceted by rate",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"05_warmup_x_chunk_facet_rate__{stat}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# PLOT TYPE 6 — Aggregate (rate, value) collapsing chunk/warmup → boxplot
# ============================================================
def plot_aggregate_rate(metric_label, stat, out_dir):
    """1 figure: 2 subplot: (a) avg over warmup, lines = chunk;
                           (b) avg over chunk, lines = warmup."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    axes = axes[0]
    key = f"{metric_label}_{stat}"

    cmap = plt.get_cmap("viridis")
    ax = axes[0]
    for ci, c in enumerate(CHUNKS):
        ys = [np.nanmean([get(r, c, t, key) for t in WARMUPS]) for r in RATES]
        ax.plot(RATES, ys, marker="o", label=f"chunk={c}",
                color=cmap(ci / max(1, len(CHUNKS) - 1)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES); ax.set_xticklabels(RATES)
    ax.set_xlabel("request_rate")
    ax.set_ylabel(f"{metric_label} ({stat}) — avg over T_warmup")
    ax.set_title("Rate vs chunk_size (mean over warmup)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    cmap = plt.get_cmap("plasma")
    ax = axes[1]
    for ti, t in enumerate(WARMUPS):
        ys = [np.nanmean([get(r, c, t, key) for c in CHUNKS]) for r in RATES]
        ax.plot(RATES, ys, marker="o", label=f"T={t}",
                color=cmap(ti / max(1, len(WARMUPS) - 1)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES); ax.set_xticklabels(RATES)
    ax.set_xlabel("request_rate")
    ax.set_ylabel(f"{metric_label} ({stat}) — avg over chunk_size")
    ax.set_title("Rate vs T_warmup (mean over chunk)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"{metric_label} {stat} — aggregate views", fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"06_aggregate_rate__{stat}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Main: vẽ tất cả
# ============================================================
for metric_label, sub in METRICS:
    out_dir = ensure(os.path.join(ROOT, sub))
    print(f"\n=== {sub} ({metric_label}) → {out_dir} ===")
    plot_heatmap_grid(metric_label, out_dir)
    for stat in STATS:
        plot_rate_x_chunk_facet_warmup(metric_label, stat, out_dir)
        plot_rate_x_warmup_facet_chunk(metric_label, stat, out_dir)
        plot_chunk_x_warmup_facet_rate(metric_label, stat, out_dir)
        plot_warmup_x_chunk_facet_rate(metric_label, stat, out_dir)
        plot_aggregate_rate(metric_label, stat, out_dir)

print("\nAll plots saved.")
