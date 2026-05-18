"""TPOT distribution — một panel mỗi rate."""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
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
    _, tpot_xx = sweep[r]["optxxx"]
    _, tpot_mm = sweep[r]["merged"]
    hi = max(np.percentile(tpot_xx, 99.5), np.percentile(tpot_mm, 99.5))
    bins = np.linspace(0, hi, 50)

    ax.hist(tpot_xx, bins=bins, alpha=0.6, color=C_XX,
             label=f"{LBL_XX}  mean={tpot_xx.mean():.0f} med={np.median(tpot_xx):.0f}")
    ax.hist(tpot_mm, bins=bins, alpha=0.6, color=C_MM,
             label=f"{LBL_MM}  mean={tpot_mm.mean():.0f} med={np.median(tpot_mm):.0f}")

    ax.axvline(tpot_xx.mean(), color=C_XX, ls="--", lw=1.2, alpha=0.85)
    ax.axvline(tpot_mm.mean(), color=C_MM, ls="--", lw=1.2, alpha=0.85)

    ax.set_xlabel("Real TPOT (ms/token)")
    ax.set_ylabel("# requests")
    ax.set_title(f"r = {int(r)} QPS  ·  n_req = {len(tpot_xx)}",
                  fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.3)

for j in range(n, nrow * ncol):
    axes[j // ncol, j % ncol].axis("off")

fig.suptitle(
    "TPOT distribution — burst (cv=8) sweep · Meta-Llama-3-8B-Instruct",
    fontsize=14, fontweight="bold", y=1.0,
)

plt.savefig("tpot_per_rate.png", dpi=140,
             bbox_inches="tight", facecolor="white")
print("Saved tpot_per_rate.png")
