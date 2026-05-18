"""
Phân tích kết quả bench_sweep.sh — sweep max_num_seqs cho opt-cpu @ rate=16.

Mục tiêu: Confirm hypothesis "batch contention là cause chính của TPOT cao".
Decision rule:
  - Nếu TPOT giảm monotonic khi cap nhỏ → H1 (batch contention) CONFIRMED.
  - Nếu TPOT plateau / không đổi → H1 không phải cause, cần điều tra khác.

Reference: opt-xxx baseline ở rate=16 có TPOT mean = 1.63s.
  Mean concurrent decoding của opt-xxx = 150 (computed exact from earlier).

Inputs: SWEEP_RES/cap{N}/{latency-*.pt, vllm-*.json}
Outputs: bench_analysis/sweep_*.png
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent
SWEEP = ROOT / "SWEEP_RES"
SERVE = ROOT / "SERVE_RES"
OUT = ROOT / "bench_analysis"
OUT.mkdir(exist_ok=True)

PT_KEYS = ["ttfts", "real_tpots", "latencies", "nlatencies", "actual_output_lens",
           "input_lens", "est_lens", "texts", "aux_model_scores", "pred_scores"]

# Caps đã có trong SWEEP_RES
CAPS = [32, 64, 128, 256, 512]


def load_run(folder):
    """Load 1 run từ folder (vd SWEEP_RES/cap32)."""
    pt_files = sorted(folder.glob("latency-*.pt"))
    js_files = sorted(folder.glob("vllm-*.json"))
    if not pt_files or not js_files:
        return None
    raw = torch.load(pt_files[-1])
    pt = {k: raw[i] for i, k in enumerate(PT_KEYS) if i < len(raw)}
    js = json.load(open(js_files[-1]))
    n = min(len(pt["ttfts"]), len(js["itls"]))
    return {
        "n": n,
        "duration": js["duration"],
        "ttft": np.asarray(pt["ttfts"][:n], dtype=float),
        "tpot": np.asarray(pt["real_tpots"][:n], dtype=float),
        "latency": np.asarray(pt["latencies"][:n], dtype=float),
        "output_len": np.asarray(pt["actual_output_lens"][:n], dtype=float),
        "input_len": np.asarray(pt["input_lens"][:n], dtype=float),
        "throughput_tok_s": js["output_throughput"],
        "throughput_req_s": js["request_throughput"],
        "completed": js["completed"],
    }


def load_baseline(sched, rate=16):
    """Load opt-xxx hoặc opt-cpu default từ SERVE_RES để compare."""
    p = SERVE / f"latency-{sched}-Meta-Llama-3-8B-Instruct-p0-r{float(rate)}-c1.0-t60.0-o-1.pt"
    if not p.exists():
        return None
    raw = torch.load(p)
    pt = {k: raw[i] for i, k in enumerate(PT_KEYS) if i < len(raw)}
    js_paths = sorted(glob.glob(str(SERVE / f"vllm-{float(rate)}qps-cv1.0-*-{sched}-*.json")))
    js = json.load(open(js_paths[-1]))
    n = min(len(pt["ttfts"]), len(js["itls"]))
    return {
        "n": n,
        "duration": js["duration"],
        "ttft": np.asarray(pt["ttfts"][:n], dtype=float),
        "tpot": np.asarray(pt["real_tpots"][:n], dtype=float),
        "latency": np.asarray(pt["latencies"][:n], dtype=float),
        "output_len": np.asarray(pt["actual_output_lens"][:n], dtype=float),
        "throughput_tok_s": js["output_throughput"],
        "throughput_req_s": js["request_throughput"],
        "completed": js["completed"],
    }


def compute_metrics(r):
    """Compute aggregate metrics từ 1 run."""
    decode_dur = r["latency"] - r["ttft"]
    return {
        "n_completed": r["completed"],
        "duration_s": r["duration"],
        "ttft_mean": r["ttft"].mean(),
        "ttft_p99": np.percentile(r["ttft"], 99),
        "tpot_mean": r["tpot"].mean(),
        "tpot_p50": np.median(r["tpot"]),
        "tpot_p99": np.percentile(r["tpot"], 99),
        "output_len_mean": r["output_len"].mean(),
        "decode_mean": decode_dur.mean(),
        "throughput_tok_s": r["throughput_tok_s"],
        "throughput_req_s": r["throughput_req_s"],
        "mean_concurrent": decode_dur.sum() / r["duration"],   # EXACT
    }


def main():
    # ───── Load all sweep runs ─────
    print(f"Loading sweep runs from {SWEEP}...")
    runs = {}
    for cap in CAPS:
        folder = SWEEP / f"cap{cap}"
        r = load_run(folder)
        if r is not None:
            runs[cap] = compute_metrics(r)
            print(f"  cap={cap:>4}: n={r['n']}, duration={r['duration']:.0f}s, "
                  f"TPOT mean={runs[cap]['tpot_mean']:.3f}s")

    # ───── Load baselines for reference ─────
    baseline_xxx = load_baseline("opt-xxx", 16)
    baseline_cpu_default = load_baseline("opt-cpu-warmup2.0", 16)

    bxxx = compute_metrics(baseline_xxx) if baseline_xxx else None
    bcpu = compute_metrics(baseline_cpu_default) if baseline_cpu_default else None

    print(f"\nBaselines @ rate=16:")
    if bxxx:
        print(f"  opt-xxx (sync block):  TPOT mean={bxxx['tpot_mean']:.3f}s, "
              f"mean_concurrent={bxxx['mean_concurrent']:.0f}")
    if bcpu:
        print(f"  opt-cpu (default cap): TPOT mean={bcpu['tpot_mean']:.3f}s, "
              f"mean_concurrent={bcpu['mean_concurrent']:.0f}")

    # ───── Print comprehensive table ─────
    print("\n" + "=" * 100)
    print(f"{'cap':<6}{'TPOT mean':>12}{'TPOT p50':>12}{'TPOT p99':>12}"
          f"{'TTFT mean':>12}{'thr tok/s':>12}{'mean conc':>12}"
          f"{'out_len':>10}{'duration':>10}")
    print("=" * 100)
    for cap in CAPS:
        if cap not in runs:
            continue
        m = runs[cap]
        print(f"{cap:<6}{m['tpot_mean']:>12.3f}{m['tpot_p50']:>12.3f}"
              f"{m['tpot_p99']:>12.3f}{m['ttft_mean']:>12.2f}"
              f"{m['throughput_tok_s']:>12.1f}{m['mean_concurrent']:>12.1f}"
              f"{m['output_len_mean']:>10.0f}{m['duration_s']:>10.0f}")
    if bxxx:
        m = bxxx
        print(f"{'xxx':<6}{m['tpot_mean']:>12.3f}{m['tpot_p50']:>12.3f}"
              f"{m['tpot_p99']:>12.3f}{m['ttft_mean']:>12.2f}"
              f"{m['throughput_tok_s']:>12.1f}{m['mean_concurrent']:>12.1f}"
              f"{m['output_len_mean']:>10.0f}{m['duration_s']:>10.0f}")
    print("=" * 100)

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 1 — Main result: TPOT vs max_num_seqs
    # ═══════════════════════════════════════════════════════════════════
    caps_sorted = sorted(runs.keys())
    tpot_means = [runs[c]["tpot_mean"] for c in caps_sorted]
    tpot_p50 = [runs[c]["tpot_p50"] for c in caps_sorted]
    tpot_p99 = [runs[c]["tpot_p99"] for c in caps_sorted]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(caps_sorted, tpot_means, "o-", label="TPOT mean", color="#1f77b4",
            linewidth=2.5, markersize=10)
    ax.plot(caps_sorted, tpot_p50, "s--", label="TPOT median", color="#2ca02c",
            linewidth=1.5, markersize=8, alpha=0.7)
    ax.plot(caps_sorted, tpot_p99, "^--", label="TPOT p99", color="#ff7f0e",
            linewidth=1.5, markersize=8, alpha=0.7)

    if bxxx:
        ax.axhline(bxxx["tpot_mean"], color="#d62728", linestyle=":", linewidth=2,
                   label=f"opt-xxx baseline TPOT={bxxx['tpot_mean']:.2f}s")
        ax.axhline(bxxx["tpot_p99"], color="#d62728", linestyle=":", linewidth=1, alpha=0.4,
                   label=f"opt-xxx p99 TPOT={bxxx['tpot_p99']:.2f}s")

    # Annotate values
    for c, v in zip(caps_sorted, tpot_means):
        ax.annotate(f"{v:.2f}s", (c, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color="#1f77b4")

    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_sorted)
    ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs (cap on running batch size)")
    ax.set_ylabel("TPOT (s/token)")
    ax.set_title(f"TPOT vs max_num_seqs — opt-cpu @ rate=16\n"
                 f"Decision rule: monotonic decrease as cap shrinks → H1 (batch contention) CONFIRMED")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "sweep_01_tpot_vs_cap.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: sweep_01_tpot_vs_cap.png")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 2 — Trade-off: TPOT vs Throughput
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 7))

    thrs = [runs[c]["throughput_tok_s"] for c in caps_sorted]
    ax.plot(thrs, tpot_means, "o-", color="#1f77b4", linewidth=2, markersize=12,
            label="opt-cpu @ different caps")

    # Annotate cap value next to each point
    for c, t, p in zip(caps_sorted, thrs, tpot_means):
        ax.annotate(f"cap={c}", (t, p), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)

    if bxxx:
        ax.scatter([bxxx["throughput_tok_s"]], [bxxx["tpot_mean"]],
                   color="#d62728", s=200, marker="*", zorder=5,
                   label=f"opt-xxx baseline")
        ax.annotate("opt-xxx", (bxxx["throughput_tok_s"], bxxx["tpot_mean"]),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=11, color="#d62728", weight="bold")

    ax.set_xlabel("output throughput (tok/s)")
    ax.set_ylabel("TPOT mean (s/token)")
    ax.set_title("Pareto trade-off: TPOT vs Throughput\n"
                 "Lower-left = better both. Shape của curve = trade-off opt-cpu phải chấp nhận.")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_02_pareto.png", dpi=130)
    plt.close(fig)
    print(f"Saved: sweep_02_pareto.png")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 3 — Mean concurrent (EXACT) vs cap — confirm cap thực sự work
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 3a — mean concurrent vs cap
    ax = axes[0]
    concs = [runs[c]["mean_concurrent"] for c in caps_sorted]
    ax.plot(caps_sorted, concs, "o-", color="#9467bd", linewidth=2, markersize=10)
    # ideal: y = x line (cap = mean concurrent if perfectly capped)
    ax.plot([min(caps_sorted), max(caps_sorted)],
            [min(caps_sorted), max(caps_sorted)],
            "k--", alpha=0.3, linewidth=1, label="y=x (perfect cap)")
    for c, v in zip(caps_sorted, concs):
        ax.annotate(f"{v:.0f}", (c, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    if bxxx:
        ax.axhline(bxxx["mean_concurrent"], color="#d62728", linestyle=":",
                   label=f"opt-xxx mean conc = {bxxx['mean_concurrent']:.0f}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(caps_sorted)
    ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs (cap)")
    ax.set_ylabel("mean concurrent decoding (EXACT)")
    ax.set_title("Verify: cap có thực sự giảm batch?\n"
                 "= Σ(latency-ttft) / duration")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # 3b — TPOT vs mean_concurrent
    ax = axes[1]
    ax.plot(concs, tpot_means, "o-", color="#1f77b4", linewidth=2, markersize=12)
    for c, x, y in zip(caps_sorted, concs, tpot_means):
        ax.annotate(f"cap={c}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=10)

    if bxxx:
        ax.scatter([bxxx["mean_concurrent"]], [bxxx["tpot_mean"]],
                   color="#d62728", s=200, marker="*", zorder=5)
        ax.annotate("opt-xxx", (bxxx["mean_concurrent"], bxxx["tpot_mean"]),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=11, color="#d62728", weight="bold")

    # Linear fit để show correlation
    if len(concs) >= 2:
        slope, intercept = np.polyfit(concs, tpot_means, 1)
        x_fit = np.linspace(min(concs)*0.9, max(concs)*1.1, 50)
        ax.plot(x_fit, slope * x_fit + intercept, "--", alpha=0.5, color="gray",
                label=f"linear fit: slope={slope*1000:.2f} ms/token per concurrent req")

    ax.set_xlabel("mean concurrent decoding")
    ax.set_ylabel("TPOT mean (s/token)")
    ax.set_title("TPOT vs concurrent batch — relationship\n"
                 "Linear correlation = batch contention dominant")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "sweep_03_concurrent_vs_tpot.png", dpi=130)
    plt.close(fig)
    print(f"Saved: sweep_03_concurrent_vs_tpot.png")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 4 — Side metrics: TTFT, throughput vs cap (trade-off chi tiết)
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a TTFT mean
    ax = axes[0][0]
    ttfts = [runs[c]["ttft_mean"] for c in caps_sorted]
    ax.plot(caps_sorted, ttfts, "o-", color="#ff7f0e", linewidth=2, markersize=10)
    if bxxx:
        ax.axhline(bxxx["ttft_mean"], color="#d62728", linestyle=":",
                   label=f"opt-xxx TTFT={bxxx['ttft_mean']:.1f}s")
    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_sorted); ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs"); ax.set_ylabel("TTFT mean (s)")
    ax.set_title("TTFT mean — cap nhỏ = queue dài hơn")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # 4b throughput
    ax = axes[0][1]
    ax.plot(caps_sorted, thrs, "o-", color="#2ca02c", linewidth=2, markersize=10)
    if bxxx:
        ax.axhline(bxxx["throughput_tok_s"], color="#d62728", linestyle=":",
                   label=f"opt-xxx thr={bxxx['throughput_tok_s']:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_sorted); ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs"); ax.set_ylabel("output throughput (tok/s)")
    ax.set_title("Throughput — cap nhỏ = throughput thấp")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # 4c output_len mean
    ax = axes[1][0]
    out_lens = [runs[c]["output_len_mean"] for c in caps_sorted]
    ax.plot(caps_sorted, out_lens, "o-", color="#9467bd", linewidth=2, markersize=10)
    if bxxx:
        ax.axhline(bxxx["output_len_mean"], color="#d62728", linestyle=":",
                   label=f"opt-xxx out_len={bxxx['output_len_mean']:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_sorted); ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs"); ax.set_ylabel("output_len mean")
    ax.set_title("Output length — bị ảnh hưởng bởi sampling stochastic")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # 4d duration
    ax = axes[1][1]
    durs = [runs[c]["duration_s"] for c in caps_sorted]
    ax.plot(caps_sorted, durs, "o-", color="#17becf", linewidth=2, markersize=10)
    if bxxx:
        ax.axhline(bxxx["duration_s"], color="#d62728", linestyle=":",
                   label=f"opt-xxx duration={bxxx['duration_s']:.0f}s")
    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_sorted); ax.set_xticklabels([str(c) for c in caps_sorted])
    ax.set_xlabel("max_num_seqs"); ax.set_ylabel("experiment duration (s)")
    ax.set_title("Duration — cap nhỏ → bench lâu hơn vì throughput thấp")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle("Side metrics — confirm trade-off pattern",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_04_side_metrics.png", dpi=130)
    plt.close(fig)
    print(f"Saved: sweep_04_side_metrics.png")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT — H1 (batch contention) test")
    print("=" * 70)

    if len(caps_sorted) >= 2:
        smallest_cap = caps_sorted[0]
        largest_cap = caps_sorted[-1]
        tpot_smallest = runs[smallest_cap]["tpot_mean"]
        tpot_largest = runs[largest_cap]["tpot_mean"]

        # Check monotonic
        is_monotonic = all(tpot_means[i] <= tpot_means[i+1] for i in range(len(tpot_means)-1))

        print(f"\n  TPOT @ cap={smallest_cap}: {tpot_smallest:.3f}s")
        print(f"  TPOT @ cap={largest_cap}:  {tpot_largest:.3f}s")
        print(f"  Monotonic increase với cap lớn: {'YES' if is_monotonic else 'NO'}")

        if bxxx:
            ratio_smallest_to_xxx = tpot_smallest / bxxx["tpot_mean"]
            print(f"  TPOT(cap={smallest_cap}) / TPOT(opt-xxx) = {ratio_smallest_to_xxx:.2f}×")
            if ratio_smallest_to_xxx < 1.3:
                print("  → ≈ opt-xxx baseline → H1 CONFIRMED, batch contention là cause chính")
            elif ratio_smallest_to_xxx < 1.7:
                print("  → Một phần gap giải thích bởi H1, còn lại từ cause khác")
            else:
                print("  → TPOT vẫn cao dù cap nhỏ → H1 KHÔNG đủ giải thích")
    print()


if __name__ == "__main__":
    main()
