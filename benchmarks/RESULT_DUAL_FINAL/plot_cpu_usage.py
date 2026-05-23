#!/usr/bin/env python3
"""Stacked bar chart: số lần CPU vs GPU predictor được dispatch theo request rate.

Trình bày cho thesis: rõ ràng, không chồng chéo, label tiếng Việt.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots")
SUMMARY_LOG = os.path.normpath(
    os.path.join(HERE, "..", "SERVE_DUAL_1", "summary.log")
)


def parse_router_stats(log_path):
    text = open(log_path).read()
    stats = {}
    for m in re.finditer(
        r"rate=(\d+):\s*DONE(.*?)(?=rate=\d+:|\Z)",
        text, flags=re.S,
    ):
        rate = int(m.group(1))
        chunk = m.group(2)
        m_dec = re.search(r"(\d+)\s*decisions", chunk)
        m_cpu = re.search(r"(\d+)\s*CPU submits", chunk)
        m_gpu = re.search(r"(\d+)\s*GPU sync",   chunk)
        if m_dec and m_cpu and m_gpu:
            stats[rate] = {
                "decisions": int(m_dec.group(1)),
                "cpu":       int(m_cpu.group(1)),
                "gpu":       int(m_gpu.group(1)),
            }
    return stats


def main():
    stats = parse_router_stats(SUMMARY_LOG)
    rates = sorted(stats.keys())
    n_total = np.array([stats[r]["decisions"] for r in rates])
    n_cpu   = np.array([stats[r]["cpu"]       for r in rates])
    n_gpu   = np.array([stats[r]["gpu"]       for r in rates])
    cpu_pct = 100.0 * n_cpu / np.maximum(1, n_total)

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14,
        "axes.titleweight": "bold", "axes.labelweight": "bold",
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = np.arange(len(rates))
    width = 0.6

    C_CPU = "#2ca02c"   # green
    C_GPU = "#1f77b4"   # blue

    bars_cpu = ax.bar(x, n_cpu, width=width,
                      color=C_CPU, edgecolor="#1f6f1f", lw=0.8,
                      label="CPU predictor (OpenVINO async)")
    bars_gpu = ax.bar(x, n_gpu, width=width, bottom=n_cpu,
                      color=C_GPU, edgecolor="#0f4f8f", lw=0.8,
                      label="GPU predictor (AUX-LLM sync)")

    # Số count trong mỗi segment (chỉ in nếu segment đủ to để fit)
    for xi, bc, bg, nc, ng, ntot in zip(x, bars_cpu, bars_gpu, n_cpu, n_gpu, n_total):
        if bc.get_height() >= max(n_total) * 0.06:
            ax.text(xi, bc.get_height() / 2, f"{nc}",
                    ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
        else:
            # Quá nhỏ — dán bên cạnh
            ax.annotate(f"CPU={nc}",
                        xy=(xi, bc.get_height()),
                        xytext=(xi + 0.42, bc.get_height() + max(n_total) * 0.02),
                        ha="left", va="bottom",
                        fontsize=10, color=C_CPU, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=C_CPU, lw=0.6))
        if bg.get_height() >= max(n_total) * 0.06:
            ax.text(xi, bc.get_height() + bg.get_height() / 2, f"{ng}",
                    ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")

        # Total trên đỉnh + tỉ lệ CPU%
        ax.text(xi, ntot + max(n_total) * 0.03,
                f"tổng = {ntot}",
                ha="center", va="bottom",
                fontsize=10, color="#222")
        ax.text(xi, ntot + max(n_total) * 0.09,
                f"CPU {cpu_pct[xi]:.1f}%",
                ha="center", va="bottom",
                fontsize=12, color=C_CPU, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"r = {r}" for r in rates], fontsize=12)
    ax.set_xlabel("Request rate (request/giây)", fontsize=13)
    ax.set_ylabel("Số lần router ra quyết định", fontsize=13)
    ax.set_title(
        "Phân bố quyết định của router dual1.0 theo request rate\n"
        "(CPU OpenVINO async  vs  GPU AUX-LLM sync)",
        pad=14,
    )
    ax.set_ylim(0, max(n_total) * 1.28)
    ax.grid(True, axis="y", alpha=0.25)

    # Legend với mô tả rõ ràng
    handles = [
        Patch(facecolor=C_CPU, edgecolor="#1f6f1f",
              label="CPU predictor được chọn  (chạy OpenVINO async trên CPU)"),
        Patch(facecolor=C_GPU, edgecolor="#0f4f8f",
              label="GPU predictor được chọn  (chạy AUX-LLM sync trên GPU)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              framealpha=0.95, edgecolor="#666")

    # Note dưới chart
    fig.text(0.5, -0.02,
             "Một quyết định = 1 lần scheduler nhận unscored batch và phải chọn route. "
             "Router so sánh T_main (LUT 4D) vs T_cpu (LUT 2D) để quyết.",
             ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cpu_usage.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")

    # In bảng dữ liệu cho thesis
    print()
    print(f"{'rate':>6}  {'tổng QĐ':>10}  {'CPU':>6}  {'GPU':>6}  {'CPU %':>7}")
    print("-" * 45)
    for r in rates:
        s = stats[r]
        p = 100.0 * s["cpu"] / max(1, s["decisions"])
        print(f"r={r:>3}    {s['decisions']:>10}  {s['cpu']:>6}  {s['gpu']:>6}  {p:>6.1f}%")


if __name__ == "__main__":
    main()
