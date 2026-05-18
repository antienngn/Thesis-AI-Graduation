"""TTFT rolling-mean per request — một panel mỗi rate."""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

C_MM = "#2ca02c"
C_XX = "#1f77b4"
LBL_MM = "opt-cpu-async-merged"
LBL_XX = "opt-xxx"


def load_lat(pt_file):
    data = torch.load(pt_file, weights_only=False)
    return np.array(data[0]) * 1000.0, np.array(data[1]) * 1000.0


def collect():
    out = {}
    for path in glob.glob("latency-*.pt"):
        m = re.match(
            r"latency-(opt-cpu-async-merged1\.0|opt-xxx)-.+-r(\d+\.\d+)-",
            os.path.basename(path))
        if not m:
            continue
        sched_key = "merged" if "merged" in m.group(1) else "optxxx"
        rate = float(m.group(2))
        ttft, tpot = load_lat(path)
        out.setdefault(rate, {})[sched_key] = (ttft, tpot)
    return out


sweep = collect()
rates = sorted(sweep)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

n = len(rates)
ncol = 3
nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.0 * nrow),
                          gridspec_kw={"hspace": 0.45, "wspace": 0.25})
axes = np.atleast_2d(axes).reshape(nrow, ncol)

for i, r in enumerate(rates):
    ax = axes[i // ncol, i % ncol]
    ttft_xx, _ = sweep[r]["optxxx"]
    ttft_mm, _ = sweep[r]["merged"]
    idx_xx = np.arange(len(ttft_xx))
    idx_mm = np.arange(len(ttft_mm))

    win = max(20, len(ttft_xx) // 30)
    ax.plot(idx_xx,
             pd.Series(ttft_xx).rolling(win, min_periods=1).mean(),
             color=C_XX, lw=2.4,
             label=f"{LBL_XX}  mean={ttft_xx.mean():.0f} p99={np.percentile(ttft_xx, 99):.0f}")
    ax.plot(idx_mm,
             pd.Series(ttft_mm).rolling(win, min_periods=1).mean(),
             color=C_MM, lw=2.4,
             label=f"{LBL_MM}  mean={ttft_mm.mean():.0f} p99={np.percentile(ttft_mm, 99):.0f}")

    ax.set_yscale("log")
    ax.set_xlabel("Request arrival index")
    ax.set_ylabel("TTFT (ms, log)")
    ax.set_title(f"r = {int(r)} QPS  ·  n_req = {len(ttft_xx)}",
                  fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.3)

for j in range(n, nrow * ncol):
    axes[j // ncol, j % ncol].axis("off")

fig.suptitle(
    "TTFT per request — rolling mean — burst (cv=16) sweep · Meta-Llama-3-8B-Instruct",
    fontsize=14, fontweight="bold", y=1.0,
)

plt.savefig("ttft_per_rate.png", dpi=140,
             bbox_inches="tight", facecolor="white")
print("Saved ttft_per_rate.png")
