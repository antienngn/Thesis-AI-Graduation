#!/usr/bin/env python3
"""TPOT line + bar charts comparing ONLY DPS vs Fu et al (2 schedulers).

Output: plot_for_report_final/median_tpot_two_{line,bar}_test.pdf
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
    ("opt-xxx", "Fu et al",   "#ff7f0e", "^"),
    ("dual1.0", "DPS (Ours)", "#d62728", "o"),
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


def median_tpot_ms(d):
    """TPOT recomputed from raw itls: TPOT_i = mean(itls[i]), median across i.
    Avoids re-tokenize bias in JSON field median_tpot_ms.
    """
    if not d:
        return None
    itls = d.get("itls") or []
    tpots = [sum(x) / len(x) for x in itls if len(x) > 0]
    if not tpots:
        return None
    return float(np.median(tpots)) * 1000


def collect():
    mat = np.full((len(SCHEDS), len(RATES)), np.nan)
    for i, (sched, *_) in enumerate(SCHEDS):
        for j, r in enumerate(RATES):
            v = median_tpot_ms(load_one(sched, r))
            if v is not None:
                mat[i, j] = v
    return mat


def make_line():
    mat = collect()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (sched, lbl, color, marker) in enumerate(SCHEDS):
        ys = [v / 1000 if not np.isnan(v) else None for v in mat[i]]
        lw = 2.4 if sched == "dual1.0" else 1.8
        ms = 10 if sched == "dual1.0" else 8
        ax.plot(RATES, ys, marker=marker, color=color, lw=lw, ms=ms,
                markeredgecolor="black", markeredgewidth=0.6, label=lbl)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (s/token)")
    ax.set_title("Median Time-Per-Output-Token", fontweight="bold")
    ax.set_xlim(0, max(RATES) + 4)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", alpha=0.35, color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, columnspacing=2.0, handletextpad=0.5)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "median_tpot_two_line_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def make_bar():
    mat = collect()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(RATES))
    w = 0.38
    for i, (sched, lbl, color, _) in enumerate(SCHEDS):
        offset = (i - 0.5) * w
        ax.bar(x + offset, mat[i], w, color=color, edgecolor="white", lw=0.7,
               label=lbl)
        for xi, v in zip(x, mat[i]):
            if not np.isnan(v):
                ax.text(xi + offset, v * 1.04, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("Median Time-Per-Output-Token", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RATES])
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "median_tpot_two_bar_test.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def main():
    make_line()
    make_bar()


if __name__ == "__main__":
    main()
