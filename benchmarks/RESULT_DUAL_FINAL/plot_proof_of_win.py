#!/usr/bin/env python3
"""Vẽ biểu đồ "tại sao dual1.0 win opt-xxx" — combine router routing decision
với scheduler improvement vs rate.

Giả thuyết: dual1.0 thắng opt-xxx ở vùng rate mà router quyết định ROUTE CPU
(giải phóng GPU cho main model). Ở vùng saturated, router fallback 100% GPU
sync → dual1.0 = opt-xxx.

Chart layout (2 axes overlapping):
  - Bar: % CPU routing per rate (từ SERVE_DUAL_1/summary.log)
  - Line: Δ Nlatency, TPOT, TTFT của dual1.0 vs opt-xxx (negative = dual win)
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
OUT_DIR = os.path.join(HERE, "plots")
SUMMARY_LOG = os.path.normpath(
    os.path.join(HERE, "..", "SERVE_DUAL_1", "summary.log")
)


def parse_router_stats(log_path):
    """Parse 'rate=X' + 'Router stats: N decisions, M CPU submits, K GPU sync'.

    Robust với case bash heredoc tách output (r=64 trong summary.log của ta).
    Flatten cả block thành 1 chuỗi rồi regex.
    """
    text = open(log_path).read()
    stats = {}
    # Mỗi rate có 1 chunk text giữa 'rate=N: DONE' và 'rate=N+1' hoặc EOF
    for m in re.finditer(
        r"rate=(\d+):\s*DONE(.*?)(?=rate=\d+:|\Z)",
        text, flags=re.S,
    ):
        rate = int(m.group(1))
        chunk = m.group(2)
        # Tìm 4 số: decisions, CPU submits, GPU sync, OV forwards
        m_dec = re.search(r"(\d+)\s*decisions", chunk)
        m_cpu = re.search(r"(\d+)\s*CPU submits", chunk)
        m_gpu = re.search(r"(\d+)\s*GPU sync",   chunk)
        if m_dec and m_cpu and m_gpu:
            n_dec = int(m_dec.group(1))
            n_cpu = int(m_cpu.group(1))
            n_gpu = int(m_gpu.group(1))
            stats[rate] = {
                "decisions": n_dec,
                "cpu": n_cpu,
                "gpu": n_gpu,
                "cpu_pct": 100.0 * n_cpu / max(1, n_dec),
            }
    return stats


def compute_nlatency_ms(d):
    ttfts = d.get("ttfts") or []
    itls = d.get("itls") or []
    if not (ttfts and itls):
        return float("nan")
    vals = []
    for i in range(len(ttfts)):
        n_raw = len(itls[i]) + 1
        if n_raw > 0:
            vals.append((ttfts[i] + sum(itls[i])) / n_raw)
    return float(np.mean(vals) * 1000.0) if vals else float("nan")


def compute_mean_tpot_ms(d):
    itls = d.get("itls") or []
    vals = [sum(itl) / len(itl) for itl in itls if len(itl) > 0]
    return float(np.mean(vals) * 1000.0) if vals else float("nan")


def load_one(sched, rate):
    pat = os.path.join(JSON_POOL, f"vllm-{rate}.0qps-*-{sched}-*.json")
    files = sorted(glob.glob(pat))
    return json.load(open(files[-1])) if files else None


def main():
    router_stats = parse_router_stats(SUMMARY_LOG)
    rates = sorted(router_stats.keys())
    print(f"Router stats: {router_stats}")

    cpu_pct = [router_stats[r]["cpu_pct"] for r in rates]

    # Delta % (dual - optxxx) / optxxx — negative = dual win
    nlat_delta = []
    ttft_delta = []
    tpot_delta = []
    for r in rates:
        do = load_one("opt-xxx", r)
        dd = load_one("dual1.0", r)
        if not (do and dd):
            nlat_delta.append(0); ttft_delta.append(0); tpot_delta.append(0)
            continue
        n_o = compute_nlatency_ms(do); n_d = compute_nlatency_ms(dd)
        t_o = do["mean_ttft_ms"]; t_d = dd["mean_ttft_ms"]
        p_o = compute_mean_tpot_ms(do); p_d = compute_mean_tpot_ms(dd)
        nlat_delta.append((n_d / n_o - 1) * 100)
        ttft_delta.append((t_d / t_o - 1) * 100)
        tpot_delta.append((p_d / p_o - 1) * 100)

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "legend.fontsize": 10,
    })

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rates))
    width = 0.55

    # === BAR: CPU routing % ===
    bars = ax1.bar(x, cpu_pct, width=width, color="#2ca02c",
                   alpha=0.65, edgecolor="#1f6f1f",
                   label="CPU route % (dual1.0 router decisions)")
    for xi, v, n_cpu, n_dec in zip(x, cpu_pct,
                                    [router_stats[r]["cpu"] for r in rates],
                                    [router_stats[r]["decisions"] for r in rates]):
        ax1.text(xi, v + 1.5, f"{v:.0f}%\n({n_cpu}/{n_dec})",
                 ha="center", va="bottom", fontsize=9,
                 color="#1f6f1f", fontweight="bold")

    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("CPU routing % (router decisions)",
                   color="#1f6f1f", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"r={r}" for r in rates])
    ax1.tick_params(axis="y", labelcolor="#1f6f1f")
    ax1.set_ylim(0, max(cpu_pct) * 1.4 + 5)
    ax1.grid(True, axis="y", alpha=0.3)

    # === LINE: Δ% ===
    ax2 = ax1.twinx()
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.plot(x, nlat_delta, "o-", color="#d62728", lw=2.4, ms=9,
             label="Δ Nlatency (dual vs opt-xxx)")
    ax2.plot(x, ttft_delta, "s--", color="#ff7f0e", lw=1.8, ms=7,
             label="Δ TTFT")
    ax2.plot(x, tpot_delta, "^--", color="#9467bd", lw=1.8, ms=7,
             label="Δ TPOT")
    for xi, v in zip(x, nlat_delta):
        ax2.annotate(f"{v:+.0f}%", (xi, v),
                     textcoords="offset points", xytext=(8, -3),
                     fontsize=9, color="#d62728", fontweight="bold")
    ax2.set_ylabel("Δ % vs opt-xxx  (negative = dual1.0 win)",
                   color="#d62728", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    y_lo = min(nlat_delta + ttft_delta + tpot_delta) - 5
    y_hi = max(nlat_delta + ttft_delta + tpot_delta) + 8
    ax2.set_ylim(y_lo, y_hi)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower left", fontsize=9, framealpha=0.95)

    plt.title("Tại sao dual1.0 thắng opt-xxx: CPU routing chiếm bao nhiêu → win bấy nhiêu",
              pad=14)

    # Shading: indicate "CPU active regime" vs "GPU-only regime"
    cpu_active_idx = [i for i, p in enumerate(cpu_pct) if p > 30]
    if cpu_active_idx:
        ax1.axvspan(min(cpu_active_idx) - 0.5,
                    max(cpu_active_idx) + 0.5,
                    color="#2ca02c", alpha=0.06,
                    label="_CPU-active regime")
        ax1.text((min(cpu_active_idx) + max(cpu_active_idx)) / 2,
                 ax1.get_ylim()[1] * 0.95,
                 "CPU branch active\n→ dual offloads predictor\n→ win 10-18%",
                 ha="center", va="top",
                 fontsize=9, color="#1f6f1f", style="italic")

    gpu_only_idx = [i for i, p in enumerate(cpu_pct) if p < 15]
    if gpu_only_idx:
        ax1.axvspan(min(gpu_only_idx) - 0.5,
                    max(gpu_only_idx) + 0.5,
                    color="#888888", alpha=0.08)
        ax1.text((min(gpu_only_idx) + max(gpu_only_idx)) / 2,
                 ax1.get_ylim()[1] * 0.95,
                 "GPU-only regime\n→ dual = opt-xxx\n(noise + swap-space)",
                 ha="center", va="top",
                 fontsize=9, color="#444", style="italic")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "proof_of_win.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
