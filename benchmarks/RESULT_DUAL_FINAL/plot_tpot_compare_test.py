#!/usr/bin/env python3
"""Two-scheduler TPOT comparison: explicit visual proof DPS < Fu et al.

Output:
  - median_tpot_compare_bar_test.pdf : grouped bar per rate, Δ% annotated
  - tpot_dist_r4_test.pdf            : per-request TPOT distribution at r=4
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
    ("opt-xxx", "Fu et al",   "#ff7f0e"),
    ("dual1.0", "DPS (Ours)", "#d62728"),
]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_one(sched, rate):
    files = sorted(glob.glob(os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")))
    return json.load(open(files[-1])) if files else None


def tpots_ms(d):
    if not d:
        return []
    return [sum(x) / len(x) * 1000 for x in d["itls"] if len(x) > 0]


def make_compare_bar():
    medians = {}
    for sched, *_ in SCHEDS:
        medians[sched] = [float(np.median(tpots_ms(load_one(sched, r))))
                           for r in RATES]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(RATES))
    w = 0.38
    for i, (sched, lbl, color) in enumerate(SCHEDS):
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, medians[sched], w, color=color,
                       edgecolor="white", lw=0.7, label=lbl)
        for xi, v in zip(x, medians[sched]):
            ax.text(xi + offset, v * 1.02, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)

    # Δ% annotation above each pair
    y_max = max(max(v) for v in medians.values())
    for j, r in enumerate(RATES):
        opt_v = medians["opt-xxx"][j]
        dual_v = medians["dual1.0"][j]
        pct = (dual_v - opt_v) / opt_v * 100
        ax.text(x[j], y_max * 1.18,
                f"{pct:+.0f}%",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#1f6f1f")

    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Median TPOT (ms/token)")
    ax.set_title("DPS reduces TPOT vs Fu et al. at every request rate",
                 fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RATES])
    ax.set_ylim(0, y_max * 1.30)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="lower right", frameon=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"median_tpot_compare_bar_test.{ext}")
        plt.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"  ✓ {out}")
    plt.close()


def make_dist_r4():
    opt = tpots_ms(load_one("opt-xxx", 4))
    dual = tpots_ms(load_one("dual1.0", 4))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.arange(0, 310, 10)

    ax.hist(opt,  bins=bins, color="#ff7f0e", alpha=0.5, edgecolor="white", lw=0.5,
            label="Fu et al")
    ax.hist(dual, bins=bins, color="#d62728", alpha=0.5, edgecolor="white", lw=0.5,
            label="DPS (Ours)")

    om, dm = float(np.median(opt)), float(np.median(dual))
    ax.axvline(om, color="#cc5500", ls="--", lw=1.6)
    ax.axvline(dm, color="#8b0000", ls="--", lw=1.6)

    # small inline labels next to median lines
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 55
    ax.text(om + 2, ymax * 0.92, f"{om:.0f} ms", color="#cc5500", fontsize=9, va="top")
    ax.text(dm - 2, ymax * 0.92, f"{dm:.0f} ms", color="#8b0000", fontsize=9,
            va="top", ha="right")

    pct = (dm - om) / om * 100
    ax.set_xlabel("Per-request TPOT (ms/token)")
    ax.set_ylabel("Number of requests")
    ax.set_title(f"TPOT distribution at r = 4 req/s  —  DPS median {pct:+.1f}%",
                 fontweight="bold")
    ax.set_xlim(0, max(bins))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper right", frameon=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"tpot_dist_r4_test.{ext}")
        plt.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"  ✓ {out}")
    plt.close()
    print(f"    opt n={len(opt)} med={om:.1f}  dual n={len(dual)} med={dm:.1f}  Δ={pct:+.1f}%")


def main():
    make_compare_bar()
    make_dist_r4()


if __name__ == "__main__":
    main()
