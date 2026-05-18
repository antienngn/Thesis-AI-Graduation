"""
Phase 2 — Focused analysis: opt-xxx vs opt-cpu-warmup2.0

Mục tiêu: tìm hiểu CỤ THỂ tại sao opt-cpu có TPOT cao hơn ở rate=16, dùng
cả 2 nguồn dữ liệu:
- benchmarks/SERVE_RES/latency-{sched}-...-r{rate}.0-c1.0-t60.0-o-1.pt
  → per-request: ttfts, real_tpots, latencies, actual_output_lens, input_lens,
    est_lens, aux_model_scores (predict score), pred_scores.
- benchmarks/SERVE_RES/vllm-{rate}.0qps-cv1.0-...{sched}-{date}.json
  → per-token ITLs, duration_s, errors, generated_texts (full text).

Chỉ phân tích 2 schedulers: opt-xxx (paper baseline, GPU sync) vs
opt-cpu-warmup2.0 (CPU OV async + 2s warmup).

Output: 8 plots tập trung vào hành vi runtime, KHÔNG redundant với xlsx.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent
SERVE_RES = ROOT / "SERVE_RES"
OUT = ROOT / "bench_analysis"
OUT.mkdir(exist_ok=True)

PT_KEYS = ["ttfts", "real_tpots", "latencies", "nlatencies", "actual_output_lens",
           "input_lens", "est_lens", "texts", "aux_model_scores", "pred_scores", "itl"]
SCHEDS = ["opt-xxx", "opt-cpu-warmup2.0"]
COLORS = {"opt-xxx": "#d62728", "opt-cpu-warmup2.0": "#1f77b4"}
RATES = [2, 4, 8, 16]
MODEL = "Meta-Llama-3-8B-Instruct"


def load_pt(sched, rate):
    """Load per-request arrays từ .pt file."""
    p = SERVE_RES / f"latency-{sched}-{MODEL}-p0-r{float(rate)}-c1.0-t60.0-o-1.pt"
    if not p.exists():
        return None
    raw = torch.load(p)
    return {k: raw[i] for i, k in enumerate(PT_KEYS) if i < len(raw)}


def load_json(sched, rate):
    """Load JSON output (chứa duration, errors, ITLs per-token).
    JSON có timestamp trong tên — chọn file mới nhất nếu có nhiều."""
    pattern = str(SERVE_RES / f"vllm-{float(rate)}qps-cv1.0-{MODEL}-{sched}-*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        return json.load(f)


def merge(sched, rate):
    """Trả dict gộp .pt + .json — alignment theo request order."""
    pt = load_pt(sched, rate)
    js = load_json(sched, rate)
    if pt is None or js is None:
        return None
    n = min(len(pt["ttfts"]), len(js["itls"]))
    return {
        "sched": sched, "rate": rate, "n": n,
        "duration": js["duration"],
        "ttft": np.asarray(pt["ttfts"][:n], dtype=float),
        "tpot": np.asarray(pt["real_tpots"][:n], dtype=float),
        "latency": np.asarray(pt["latencies"][:n], dtype=float),
        "nlatency": np.asarray(pt["nlatencies"][:n], dtype=float),
        "input_len": np.asarray(pt["input_lens"][:n], dtype=float),
        "output_len": np.asarray(pt["actual_output_lens"][:n], dtype=float),
        "est_len": np.asarray(pt["est_lens"][:n], dtype=float),
        "aux_score": np.asarray(pt.get("aux_model_scores", [np.nan]*n)[:n], dtype=float),
        "itls": [list(x) for x in js["itls"][:n]],
    }


def cdf(arr):
    arr = np.sort(np.asarray(arr, dtype=float))
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([]), np.array([])
    return arr, np.arange(1, len(arr)+1) / len(arr)


# =============================================================================
# Plot 1 — Cumulative output tokens over experiment time
# Trả lời: throughput thật theo thời gian, có "stall" nào không?
# =============================================================================
def plot_throughput_timeline(runs):
    """Cộng dồn token sinh ra theo wall-clock time của experiment.

    Cách build timeline (vì không có timestamp tuyệt đối):
    - Giả định arrival uniform: arrival_i = i / rate.
    - first_token_time_i = arrival_i + ttft_i.
    - Token thứ k của request i: first_token_time_i + sum(itls_i[:k]).
    - Bin theo 1s windows, đếm token rơi vào mỗi bin.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            r = runs.get((sched, rate))
            if r is None:
                continue
            timestamps = []
            for i in range(r["n"]):
                arrival = i / rate
                t0 = arrival + r["ttft"][i]
                # Token đầu tiên ở t0; token tiếp theo cộng dồn ITL
                t = t0
                timestamps.append(t)
                for itl in r["itls"][i]:
                    t += itl
                    timestamps.append(t)
            timestamps = np.array(timestamps)
            t_max = max(timestamps.max(), r["duration"]) + 1
            bins = np.arange(0, t_max, 1.0)
            hist, _ = np.histogram(timestamps, bins=bins)
            cum = np.cumsum(hist)
            ax.plot(bins[:-1], cum, label=f"{sched} (dur={r['duration']:.0f}s)",
                    color=COLORS[sched], linewidth=1.5)
        ax.set_xlabel("experiment time (s)")
        ax.set_ylabel("cumulative output tokens")
        ax.set_title(f"qps={rate}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Cumulative output tokens over time — slope = effective throughput")
    fig.tight_layout()
    fig.savefig(OUT / "01_throughput_timeline.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 2 — Concurrent active requests over time (batch size proxy)
# Trả lời: opt-cpu có chạy concurrent batch lớn hơn không?
# =============================================================================
def plot_concurrency_timeline(runs):
    """Đếm số request đang active (đã first_token, chưa done) tại mỗi giây.
    active window = [arrival + ttft, arrival + latency].
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            r = runs.get((sched, rate))
            if r is None:
                continue
            arrivals = np.arange(r["n"]) / rate
            start = arrivals + r["ttft"]
            end = arrivals + r["latency"]
            t_max = max(end.max(), r["duration"]) + 1
            ts = np.arange(0, t_max, 0.5)
            active = np.zeros_like(ts)
            for s, e in zip(start, end):
                mask = (ts >= s) & (ts < e)
                active[mask] += 1
            ax.plot(ts, active, label=sched, color=COLORS[sched], linewidth=1.0,
                    alpha=0.8)
        ax.set_xlabel("experiment time (s)")
        ax.set_ylabel("# concurrent active requests")
        ax.set_title(f"qps={rate}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Concurrent active requests — proxy cho running batch size")
    fig.tight_layout()
    fig.savefig(OUT / "02_concurrency_timeline.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 3 — ITL distribution CDF (toàn bộ token)
# Trả lời: opt-cpu có ITL distribution shift đều, hay chỉ tail dài?
# =============================================================================
def plot_itl_cdf(runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            r = runs.get((sched, rate))
            if r is None:
                continue
            all_itls = []
            for seq in r["itls"]:
                all_itls.extend(seq)
            x, y = cdf(all_itls)
            ax.plot(x, y, label=f"{sched} (n_tok={len(all_itls)})",
                    color=COLORS[sched], linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("inter-token latency (s)")
        ax.set_ylabel("CDF")
        ax.set_title(f"qps={rate}")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(0.99, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("ITL distribution — all tokens across all requests")
    fig.tight_layout()
    fig.savefig(OUT / "03_itl_cdf.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 4 — ITL spike trajectory for top longest requests at rate=16
# Trả lời: spike biệt lập (1 token) hay cluster (preempt swap-in)?
# =============================================================================
def plot_itl_spikes(runs, rate=16, n_sample=8):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    for ax, sched in zip(axes, SCHEDS):
        r = runs.get((sched, rate))
        if r is None:
            continue
        # Pick top n_sample requests by output_len
        order = np.argsort(-r["output_len"])[:n_sample]
        for idx in order:
            seq = r["itls"][idx]
            if not seq:
                continue
            ax.plot(range(len(seq)), seq, alpha=0.6, linewidth=0.6,
                    label=f"req#{idx} n={len(seq)} score={r['aux_score'][idx]:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("token index")
        ax.set_ylabel("ITL (s)")
        ax.set_title(f"{sched} @ qps={rate}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"ITL spike trajectory — top-{n_sample} longest requests @ qps={rate}")
    fig.tight_layout()
    fig.savefig(OUT / "04_itl_spikes_r16.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 5 — Decode time vs output_len; slope = TPOT
# Trả lời: TPOT khác nhau là vì decode rate khác, hay vì output ngắn?
# =============================================================================
def plot_decode_vs_output(runs, rate=16):
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    for sched in SCHEDS:
        r = runs.get((sched, rate))
        if r is None:
            continue
        decode_t = r["latency"] - r["ttft"]
        ax.scatter(r["output_len"], decode_t, s=8, alpha=0.4, color=COLORS[sched],
                   label=f"{sched}")
        # Robust slope via simple lstsq
        mask = (r["output_len"] > 0) & np.isfinite(decode_t)
        if mask.sum() > 5:
            x_, y_ = r["output_len"][mask], decode_t[mask]
            slope, intercept = np.polyfit(x_, y_, 1)
            xs = np.linspace(x_.min(), x_.max(), 50)
            ax.plot(xs, slope * xs + intercept, color=COLORS[sched], linewidth=2,
                    linestyle="--", label=f"{sched} fit: slope={slope*1000:.1f}ms/tok")
    ax.set_xlabel("actual_output_len (tokens)")
    ax.set_ylabel("decode time = latency - ttft (s)")
    ax.set_title(f"Decode time vs output length @ qps={rate}\nslope ≈ TPOT")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"05_decode_vs_output_r{rate}.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 6 — Per-request stacked breakdown: queue (ttft) vs decode
# Trả lời: latency cao là do queue lâu hay decode chậm?
# =============================================================================
def plot_latency_breakdown(runs, rate=16, n_sample=200):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, sched in zip(axes, SCHEDS):
        r = runs.get((sched, rate))
        if r is None:
            continue
        # Sort theo total latency, lấy top n_sample
        order = np.argsort(r["latency"])[::-1][:n_sample]
        ttft = r["ttft"][order]
        decode = r["latency"][order] - r["ttft"][order]
        x = np.arange(len(order))
        ax.bar(x, ttft, color="#888888", label="queue/TTFT", width=1.0)
        ax.bar(x, decode, bottom=ttft, color=COLORS[sched], label="decode", width=1.0)
        ax.set_xlabel(f"top-{n_sample} slowest requests (sorted desc)")
        ax.set_ylabel("time (s)")
        ax.set_title(f"{sched} @ qps={rate}  "
                     f"mean queue={ttft.mean():.1f}s  mean decode={decode.mean():.1f}s")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Latency breakdown — queue vs decode (top-{n_sample} slowest)")
    fig.tight_layout()
    fig.savefig(OUT / f"06_latency_breakdown_r{rate}.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 7 — Predictor score → outcome correlations
# Trả lời: aux_score predict được cái gì? OPT-OV có numeric drift so OPT-GPU?
# =============================================================================
def plot_score_vs_outcomes(runs, rate=16):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, sched in enumerate(SCHEDS):
        r = runs.get((sched, rate))
        if r is None:
            continue
        score = r["aux_score"]
        if not np.any(np.isfinite(score)):
            for ax in axes[row]:
                ax.text(0.5, 0.5, f"no aux_score\n{sched}", ha="center", va="center")
            continue
        targets = [
            ("output_len", r["output_len"], "actual_output_len (tok)"),
            ("ttft", r["ttft"], "TTFT (s)"),
            ("decode_time", r["latency"] - r["ttft"], "decode time (s)"),
        ]
        for ax, (key, target, ylabel) in zip(axes[row], targets):
            mask = np.isfinite(score) & np.isfinite(target)
            x, y = score[mask], target[mask]
            ax.scatter(x, y, s=6, alpha=0.4, color=COLORS[sched])
            if len(x) > 5:
                corr = np.corrcoef(x, y)[0, 1]
                ax.set_title(f"{sched}: score vs {key}\nPearson r = {corr:.3f}",
                             fontsize=10)
            ax.set_xlabel("aux_model_score")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Predictor score → outcomes @ qps={rate}")
    fig.tight_layout()
    fig.savefig(OUT / f"07_score_vs_outcomes_r{rate}.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Plot 8 — TTFT vs request rank (admission order)
# Trả lời: queue có drain ổn định, hay tích lũy theo thời gian?
# =============================================================================
def plot_ttft_vs_rank(runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            r = runs.get((sched, rate))
            if r is None:
                continue
            # Sort theo arrival index (= request rank trong order gửi)
            ranks = np.arange(r["n"])
            ax.plot(ranks, r["ttft"], label=sched, color=COLORS[sched],
                    alpha=0.6, linewidth=0.5)
            # Smooth trend
            if r["n"] > 30:
                window = max(10, r["n"] // 30)
                kernel = np.ones(window) / window
                smooth = np.convolve(r["ttft"], kernel, mode="valid")
                ax.plot(ranks[:len(smooth)], smooth, color=COLORS[sched],
                        linewidth=2.0, label=f"{sched} (smoothed)")
        ax.set_xlabel("request arrival rank")
        ax.set_ylabel("TTFT (s)")
        ax.set_title(f"qps={rate}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("TTFT vs arrival rank — queue accumulation pattern")
    fig.tight_layout()
    fig.savefig(OUT / "08_ttft_vs_rank.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Summary print
# =============================================================================
def print_summary(runs):
    print("\n" + "=" * 90)
    print(f"{'metric':<22}{'qps':>4} {'opt-xxx':>15} {'opt-cpu':>15} {'delta %':>10}")
    print("=" * 90)
    metrics = [
        ("TTFT mean (s)", lambda r: r["ttft"].mean()),
        ("TTFT p99 (s)", lambda r: np.percentile(r["ttft"], 99)),
        ("TPOT mean (s)", lambda r: r["tpot"].mean()),
        ("TPOT p99 (s)", lambda r: np.percentile(r["tpot"], 99)),
        ("decode mean (s)", lambda r: (r["latency"] - r["ttft"]).mean()),
        ("output_len mean", lambda r: r["output_len"].mean()),
        ("output_len p99", lambda r: np.percentile(r["output_len"], 99)),
        ("duration_s", lambda r: r["duration"]),
        ("output tok/s",
         lambda r: r["output_len"].sum() / r["duration"]),
    ]
    for label, fn in metrics:
        for rate in RATES:
            a = runs.get(("opt-xxx", rate))
            b = runs.get(("opt-cpu-warmup2.0", rate))
            if a is None or b is None:
                continue
            va, vb = fn(a), fn(b)
            delta = 100 * (vb - va) / va if va != 0 else float("nan")
            flag = " *" if abs(delta) > 30 else ""
            print(f"{label:<22}{rate:>4} {va:>15.3f} {vb:>15.3f} {delta:>+9.1f}%{flag}")
        print("-" * 90)


def main():
    print("Loading runs (.pt + .json)...")
    runs = {}
    for sched in SCHEDS:
        for rate in RATES:
            r = merge(sched, rate)
            if r:
                runs[(sched, rate)] = r
                print(f"  {sched:25} qps={rate:>3}  n={r['n']:4}  duration={r['duration']:.0f}s")
            else:
                print(f"  {sched:25} qps={rate:>3}  MISSING")

    print(f"\nSaving plots to {OUT}/")
    plot_throughput_timeline(runs);     print("  01_throughput_timeline.png")
    plot_concurrency_timeline(runs);    print("  02_concurrency_timeline.png")
    plot_itl_cdf(runs);                 print("  03_itl_cdf.png")
    plot_itl_spikes(runs, rate=16);     print("  04_itl_spikes_r16.png")
    plot_decode_vs_output(runs, rate=16); print("  05_decode_vs_output_r16.png")
    plot_latency_breakdown(runs, rate=16); print("  06_latency_breakdown_r16.png")
    plot_score_vs_outcomes(runs, rate=16); print("  07_score_vs_outcomes_r16.png")
    plot_ttft_vs_rank(runs);            print("  08_ttft_vs_rank.png")

    print_summary(runs)
    print("\nDone.")


if __name__ == "__main__":
    main()
