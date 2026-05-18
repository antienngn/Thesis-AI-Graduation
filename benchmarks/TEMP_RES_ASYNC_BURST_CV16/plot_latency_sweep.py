"""Latency vs req/s — sweep summary cho cả hai scheduler.

2x2 grid:
  (TL) TTFT mean ·  (TR) TTFT p99
  (BL) TPOT mean ·  (BR) TPOT p99

Đọc trực tiếp từ vllm-*.json (đã có mean/median/p99 tính sẵn).
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

C_MM = "#2ca02c"
C_XX = "#1f77b4"
LBL_MM = "opt-cpu-async-merged"
LBL_XX = "opt-xxx"


def load_sweep():
    out = {"merged": {}, "optxxx": {}}
    for path in glob.glob("vllm-*.json"):
        m = re.match(r"vllm-(\d+\.\d+)qps-cv[\d.]+-.+-(opt-cpu-async-merged1\.0|opt-xxx)-",
                     os.path.basename(path))
        if not m:
            continue
        rate = float(m.group(1))
        sched_key = "merged" if "merged" in m.group(2) else "optxxx"
        with open(path) as f:
            d = json.load(f)
        out[sched_key][rate] = d
    return out


data = load_sweep()
rates = sorted(set(data["merged"]) & set(data["optxxx"]))


def series(sched, field):
    return np.array([data[sched][r][field] for r in rates])


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                          gridspec_kw={"hspace": 0.35, "wspace": 0.25})

panels = [
    (axes[0, 0], "mean_ttft_ms", "(A) TTFT mean", "TTFT (ms, log)"),
    (axes[0, 1], "p99_ttft_ms",  "(B) TTFT p99",  "TTFT (ms, log)"),
    (axes[1, 0], "mean_tpot_ms", "(C) TPOT mean", "TPOT (ms/tok, log)"),
    (axes[1, 1], "p99_tpot_ms",  "(D) TPOT p99",  "TPOT (ms/tok, log)"),
]

for ax, field, title, ylabel in panels:
    y_mm = series("merged", field)
    y_xx = series("optxxx", field)
    ax.plot(rates, y_xx, "o-", color=C_XX, lw=2.4, ms=9, label=LBL_XX)
    ax.plot(rates, y_mm, "s-", color=C_MM, lw=2.4, ms=9, label=LBL_MM)
    for r, vx, vm in zip(rates, y_xx, y_mm):
        ax.annotate(f"{vx:.0f}", (r, vx), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9, color=C_XX)
        ax.annotate(f"{vm:.0f}", (r, vm), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=C_MM)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{int(r)}" for r in rates])
    ax.set_xlabel("offered rate (QPS)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left" if "TPOT" in title else "upper left")

fig.suptitle(
    "Burst (cv=16) sweep — TTFT/TPOT vs offered rate · Meta-Llama-3-8B-Instruct\n"
    f"rates = {[int(r) for r in rates]} QPS · t=60s warmup · ShareGPT",
    fontsize=14, fontweight="bold", y=1.0,
)

plt.savefig("latency_sweep.png", dpi=140,
             bbox_inches="tight", facecolor="white")
print("Saved latency_sweep.png")
for r in rates:
    print(f"  r={int(r):>2} QPS | "
          f"TTFT mean: xx={data['optxxx'][r]['mean_ttft_ms']:>7.0f} mm={data['merged'][r]['mean_ttft_ms']:>7.0f} | "
          f"TPOT mean: xx={data['optxxx'][r]['mean_tpot_ms']:>5.0f} mm={data['merged'][r]['mean_tpot_ms']:>5.0f}")
