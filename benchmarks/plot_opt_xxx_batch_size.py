#!/usr/bin/env python3
"""plot_opt_xxx_batch_size.py — Visualize batch size distribution của
predictor calls trong opt-xxx scheduler (AUXLLM GPU).

Đọc PRE_LAT_E2E/predictor_latency_raw.csv, filter scheduler=opt-xxx,
plot:
  - Panel 1: histogram batch_size theo rate
  - Panel 2: batch_size vs latency (scatter, batch-amortization curve)
  - Panel 3: batch size CDF theo rate (log-x)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CSV = ROOT / "PRE_LAT_E2E/predictor_latency_raw.csv"
OUT = ROOT / "PRE_LAT_E2E/opt_xxx_batch_size_profile.png"

with open(CSV) as f:
    rows = [r for r in csv.DictReader(f) if r["scheduler"] == "opt-xxx"]

# Group by rate
by_rate = defaultdict(list)  # rate -> [(n, lat_ms)]
for r in rows:
    by_rate[int(r["rate"])].append((int(r["n_per_chunk"]),
                                     float(r["latency_ms"])))

RATES = sorted(by_rate.keys())
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: histogram batch size (log-y)
ax = axes[0]
bins = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257]
for i, rate in enumerate(RATES):
    ns = [n for n, _ in by_rate[rate]]
    ax.hist(ns, bins=bins, label=f"r={rate}",
            color=cmap(i / max(1, len(RATES) - 1)),
            histtype="step", linewidth=2)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xlabel("batch size (n_per_chunk)")
ax.set_ylabel("# forward calls (log)")
ax.set_title("Distribution batch size per rate")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# Panel 2: latency vs batch size (amortization)
ax = axes[1]
for i, rate in enumerate(RATES):
    ns = np.array([n for n, _ in by_rate[rate]])
    lats = np.array([l for _, l in by_rate[rate]])
    ax.scatter(ns, lats, label=f"r={rate}",
               color=cmap(i / max(1, len(RATES) - 1)),
               s=20, alpha=0.4, edgecolors="none")
# Reference: per-req cost projection
ns_axis = np.array([1, 2, 4, 8, 16, 32, 64, 128])
ax.plot(ns_axis, 10 * np.ones_like(ns_axis), "k--",
        alpha=0.4, label="10ms baseline (n=1)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xlabel("batch size")
ax.set_ylabel("forward latency (ms, log)")
ax.set_title("Latency vs batch size — batch amortization scaling")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, which="both")

# Panel 3: CDF batch size per rate
ax = axes[2]
for i, rate in enumerate(RATES):
    ns = sorted(n for n, _ in by_rate[rate])
    cdf = np.arange(1, len(ns) + 1) / len(ns)
    ax.step(ns, cdf, label=f"r={rate}",
            color=cmap(i / max(1, len(RATES) - 1)), where="post",
            linewidth=2)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xlabel("batch size")
ax.set_ylabel("CDF (cumulative fraction of calls)")
ax.set_title("CDF batch size per rate")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(0, 1.02)

fig.suptitle(
    "opt-xxx scheduler — predictor (OPT-125m AUXLLM) batch size profile",
    fontsize=13, y=1.00,
)
fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"Wrote {OUT}")
