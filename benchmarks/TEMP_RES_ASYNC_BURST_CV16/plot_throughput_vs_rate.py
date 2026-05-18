"""Throughput vs offered rate — kiểm tra điểm saturate.

(L) request throughput  (req/s actually completed)
(R) output token throughput (tok/s actually generated)
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
rates = np.array(sorted(set(data["merged"]) & set(data["optxxx"])))


def series(sched, field):
    return np.array([data[sched][r][field] for r in rates])


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                          gridspec_kw={"wspace": 0.25})

# Panel L: request throughput
ax = axes[0]
req_xx = series("optxxx", "request_throughput")
req_mm = series("merged", "request_throughput")
ax.plot(rates, rates, color="black", ls="--", lw=1.2, alpha=0.6,
         label="offered = completed (ideal)")
ax.plot(rates, req_xx, "o-", color=C_XX, lw=2.4, ms=9, label=LBL_XX)
ax.plot(rates, req_mm, "s-", color=C_MM, lw=2.4, ms=9, label=LBL_MM)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(rates)
ax.set_xticklabels([f"{int(r)}" for r in rates])
ax.set_xlabel("offered rate (QPS)")
ax.set_ylabel("Completed request throughput (req/s)")
ax.set_title("(A) Request throughput — gap với ideal = backlog",
              fontweight="bold", loc="left")
ax.legend(loc="upper left")
ax.grid(True, which="both", alpha=0.3)

# Panel R: output token throughput
ax = axes[1]
tok_xx = series("optxxx", "output_throughput")
tok_mm = series("merged", "output_throughput")
ax.plot(rates, tok_xx, "o-", color=C_XX, lw=2.4, ms=9,
         label=f"{LBL_XX}")
ax.plot(rates, tok_mm, "s-", color=C_MM, lw=2.4, ms=9,
         label=f"{LBL_MM}")
for r, vx, vm in zip(rates, tok_xx, tok_mm):
    ax.annotate(f"{vx:.0f}", (r, vx), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9, color=C_XX)
    ax.annotate(f"{vm:.0f}", (r, vm), textcoords="offset points",
                 xytext=(0, -16), ha="center", fontsize=9, color=C_MM)
ax.set_xscale("log", base=2)
ax.set_xticks(rates)
ax.set_xticklabels([f"{int(r)}" for r in rates])
ax.set_xlabel("offered rate (QPS)")
ax.set_ylabel("Output token throughput (tok/s)")
ax.set_title("(B) Output token throughput — plateau = saturate",
              fontweight="bold", loc="left")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Burst (cv=16) sweep — throughput vs offered rate · Meta-Llama-3-8B-Instruct\n"
    f"rates = {[int(r) for r in rates]} QPS",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.savefig("throughput_vs_rate.png", dpi=140,
             bbox_inches="tight", facecolor="white")
print("Saved throughput_vs_rate.png")
for r in rates:
    print(f"  r={int(r):>2} | req_thr: xx={data['optxxx'][r]['request_throughput']:.2f} mm={data['merged'][r]['request_throughput']:.2f} | "
          f"tok_thr: xx={data['optxxx'][r]['output_throughput']:>6.1f} mm={data['merged'][r]['output_throughput']:>6.1f}")
