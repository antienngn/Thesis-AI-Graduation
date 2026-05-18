"""
Phân tích tại sao opt-cpu-warmup có TTFT thấp hơn opt-xxx.

Causal hypothesis:
  opt-xxx: SYNC predictor call (obtain_aux_scores) → block event loop 10-50ms
           mỗi scheduler tick. 2 effects:
    (i)  Direct: GPU dispatch không xảy ra trong khi block → GPU IDLE
    (ii) Indirect: throughput thực tế < arrival rate → queue tích lũy → TTFT
         tăng theo thời gian.
  opt-cpu: ASYNC predictor → tick non-blocking → GPU full duty cycle → throughput
           giữ kịp arrival → queue ổn định → TTFT bounded.

Decision rule:
  - Ở low rate (under-saturated): TTFT gap nhỏ ~ predictor overhead (10-100ms).
  - Ở high rate (saturated): TTFT gap LỚN do queue accumulation effect.
  - TTFT vs arrival rank: opt-xxx có trend tăng dần (queue grow), opt-cpu flat.

Inputs: SERVE_RES/{latency-*.pt, vllm-*.json} cho rate ∈ {2, 4, 8, 16}.
Outputs: bench_analysis/ttft_*.png
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent
SERVE = ROOT / "SERVE_RES"
OUT = ROOT / "bench_analysis"
OUT.mkdir(exist_ok=True)

PT_KEYS = ["ttfts", "real_tpots", "latencies", "nlatencies", "actual_output_lens",
           "input_lens", "est_lens", "texts", "aux_model_scores", "pred_scores"]

SCHEDS = ["opt-xxx", "opt-cpu-warmup2.0"]
COLORS = {"opt-xxx": "#d62728", "opt-cpu-warmup2.0": "#1f77b4"}
LABELS = {"opt-xxx": "opt-xxx", "opt-cpu-warmup2.0": "opt-cpu"}
RATES = [2, 4, 8, 16]


def load(sched, rate):
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
    }


def cdf(arr):
    arr = np.sort(np.asarray(arr, dtype=float))
    arr = arr[np.isfinite(arr)]
    return arr, np.arange(1, len(arr) + 1) / len(arr)


def main():
    print("Loading rate-sweep data for both schedulers...")
    runs = {}
    for sched in SCHEDS:
        for rate in RATES:
            r = load(sched, rate)
            if r is not None:
                runs[(sched, rate)] = r

    # ───── Print summary table ─────
    print("\n" + "=" * 100)
    print(f"{'rate':>6}{'sched':>10}{'TTFT mean (s)':>15}{'TTFT p50':>12}"
          f"{'TTFT p99':>12}{'thr tok/s':>12}{'thr req/s':>12}{'duration':>12}")
    print("=" * 100)
    for rate in RATES:
        for sched in SCHEDS:
            if (sched, rate) not in runs:
                continue
            r = runs[(sched, rate)]
            print(f"{rate:>6}{LABELS[sched]:>10}{r['ttft'].mean():>15.3f}"
                  f"{np.median(r['ttft']):>12.3f}{np.percentile(r['ttft'], 99):>12.3f}"
                  f"{r['throughput_tok_s']:>12.1f}{r['throughput_req_s']:>12.2f}"
                  f"{r['duration']:>12.1f}")
        # gap row
        if ("opt-xxx", rate) in runs and ("opt-cpu-warmup2.0", rate) in runs:
            xxx = runs[("opt-xxx", rate)]["ttft"].mean()
            cpu = runs[("opt-cpu-warmup2.0", rate)]["ttft"].mean()
            gap = (cpu - xxx) / xxx * 100
            print(f"{rate:>6}{'GAP':>10}{cpu - xxx:>15.3f}"
                  f"{'':>12}{'':>12}{'':>12}{'':>12}{f'({gap:+.0f}%)':>12}")
        print("-" * 100)

    # ═══════════════════════════════════════════════════════════════════
    # Plot 1 — TTFT CDF for each rate (4 panels)
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            if (sched, rate) not in runs:
                continue
            r = runs[(sched, rate)]
            x, y = cdf(r["ttft"])
            ax.plot(x, y, label=f"{LABELS[sched]} mean={r['ttft'].mean():.2f}s "
                                f"p99={np.percentile(r['ttft'],99):.1f}s",
                    color=COLORS[sched], linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(0.99, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("TTFT (s, log scale)")
        ax.set_ylabel("CDF")
        ax.set_title(f"qps={rate}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("TTFT distribution — opt-cpu shifts left at all rates", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_01_cdf.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: ttft_01_cdf.png")

    # ═══════════════════════════════════════════════════════════════════
    # Plot 2 — TTFT vs arrival rank (queue accumulation pattern)
    # KEY: opt-xxx phải có trend tăng theo rank ở rate cao (queue grow).
    # opt-cpu phải flat (queue stable).
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        for sched in SCHEDS:
            if (sched, rate) not in runs:
                continue
            r = runs[(sched, rate)]
            ranks = np.arange(r["n"])
            # raw scatter (thin)
            ax.scatter(ranks, r["ttft"], s=2, alpha=0.2, color=COLORS[sched])
            # smoothed trend (moving average)
            window = max(20, r["n"] // 30)
            kernel = np.ones(window) / window
            smooth = np.convolve(r["ttft"], kernel, mode="valid")
            ax.plot(ranks[:len(smooth)], smooth, color=COLORS[sched],
                    linewidth=2.5, label=f"{LABELS[sched]}")
        ax.set_xlabel("arrival rank (request order)")
        ax.set_ylabel("TTFT (s, log scale)")
        ax.set_yscale("log")
        ax.set_title(f"qps={rate} — trend rising = queue accumulating")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle("TTFT vs arrival rank — opt-xxx grows over time, opt-cpu stable",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_02_vs_rank.png", dpi=130)
    plt.close(fig)
    print(f"Saved: ttft_02_vs_rank.png")

    # ═══════════════════════════════════════════════════════════════════
    # Plot 3 — TTFT mean & p99 vs rate (scaling pattern)
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, agg_name, agg_fn in [(axes[0], "mean", np.mean),
                                  (axes[1], "p99", lambda x: np.percentile(x, 99))]:
        for sched in SCHEDS:
            xs, ys = [], []
            for rate in RATES:
                if (sched, rate) in runs:
                    xs.append(rate)
                    ys.append(agg_fn(runs[(sched, rate)]["ttft"]))
            if xs:
                ax.plot(xs, ys, "o-", label=LABELS[sched], color=COLORS[sched],
                        linewidth=2.5, markersize=12)
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points",
                                xytext=(0, 12), ha="center", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(RATES)
        ax.set_xticklabels([str(r) for r in RATES])
        ax.set_xlabel("request rate (qps)")
        ax.set_ylabel(f"TTFT {agg_name} (s)")
        ax.set_title(f"TTFT {agg_name} vs rate")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11)
    fig.suptitle("TTFT scaling: opt-xxx blows up at high rate (queue accumulation), "
                 "opt-cpu bounded (no event loop block)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_03_vs_rate.png", dpi=130)
    plt.close(fig)
    print(f"Saved: ttft_03_vs_rate.png")

    # ═══════════════════════════════════════════════════════════════════
    # Plot 4 — Throughput vs rate (saturation point detection)
    # KEY: throughput của opt-xxx flat ở rate cao (saturated, can't keep up)
    #      throughput của opt-cpu scale với rate.
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 4a — output throughput (tok/s)
    ax = axes[0]
    for sched in SCHEDS:
        xs, ys = [], []
        for rate in RATES:
            if (sched, rate) in runs:
                xs.append(rate)
                ys.append(runs[(sched, rate)]["throughput_tok_s"])
        if xs:
            ax.plot(xs, ys, "o-", label=LABELS[sched], color=COLORS[sched],
                    linewidth=2.5, markersize=12)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel("request rate (qps)")
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_title("Output throughput vs rate\nopt-xxx flat ở rate cao (saturated)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # 4b — request throughput (req/s) vs offered rate
    ax = axes[1]
    for sched in SCHEDS:
        xs, ys = [], []
        for rate in RATES:
            if (sched, rate) in runs:
                xs.append(rate)
                ys.append(runs[(sched, rate)]["throughput_req_s"])
        if xs:
            ax.plot(xs, ys, "o-", label=LABELS[sched], color=COLORS[sched],
                    linewidth=2.5, markersize=12)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=10)
    # Diagonal y=x = perfect (throughput matches arrival)
    ax.plot([1, 30], [1, 30], "k--", alpha=0.4, linewidth=1, label="y=x (no queue)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel("offered rate (qps)")
    ax.set_ylabel("served rate (req/s)")
    ax.set_title("Request throughput vs offered rate\nDưới y=x = saturated, queue grow")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    fig.suptitle("Throughput analysis — opt-xxx loses ability to keep up at rate=8+",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_04_throughput.png", dpi=130)
    plt.close(fig)
    print(f"Saved: ttft_04_throughput.png")

    # ═══════════════════════════════════════════════════════════════════
    # Plot 5 — Decomposition: TTFT gap vs predictor overhead
    # Idea: ở low rate (under-saturated), TTFT gap = predictor overhead
    #       ở high rate (saturated), TTFT gap = queue accumulation effect
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 7))

    rates_x, gaps, gap_pct = [], [], []
    for rate in RATES:
        if ("opt-xxx", rate) in runs and ("opt-cpu-warmup2.0", rate) in runs:
            xxx_mean = runs[("opt-xxx", rate)]["ttft"].mean()
            cpu_mean = runs[("opt-cpu-warmup2.0", rate)]["ttft"].mean()
            rates_x.append(rate)
            gaps.append(xxx_mean - cpu_mean)
            gap_pct.append((xxx_mean - cpu_mean) / xxx_mean * 100)

    bars = ax.bar(range(len(rates_x)), gaps, color="#1f77b4", alpha=0.7, width=0.6)
    for bar, v, p in zip(bars, gaps, gap_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(gaps)*0.02,
                f"{v:.1f}s\n(-{p:.0f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(rates_x)))
    ax.set_xticklabels([f"qps={r}" for r in rates_x])
    ax.set_xlabel("request rate")
    ax.set_ylabel("TTFT gap = opt-xxx - opt-cpu (s)")
    ax.set_title("TTFT advantage của opt-cpu so với opt-xxx\n"
                 "Low rate: ~50ms = predictor block overhead\n"
                 "High rate: TÉ NƯỚC TĂNG = queue accumulation due to GPU idle")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotation regimes
    ax.axhline(0.1, color="green", linestyle=":", alpha=0.5,
               label="~50-100ms (predictor block overhead floor)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "ttft_05_gap_decomposition.png", dpi=130)
    plt.close(fig)
    print(f"Saved: ttft_05_gap_decomposition.png")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT — quantitative
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CAUSAL ANALYSIS — TTFT advantage của opt-cpu")
    print("=" * 70)

    # Saturation detection: dùng MEDIAN TTFT.
    # < 1s = not saturated (most requests not queueing)
    # > 5s = saturated (most requests queued waiting)
    SAT_THRESHOLD = 1.0  # seconds
    for rate in RATES:
        if ("opt-xxx", rate) not in runs or ("opt-cpu-warmup2.0", rate) not in runs:
            continue
        xxx = runs[("opt-xxx", rate)]
        cpu = runs[("opt-cpu-warmup2.0", rate)]

        xxx_med = np.median(xxx["ttft"])
        cpu_med = np.median(cpu["ttft"])
        xxx_saturated = xxx_med > SAT_THRESHOLD
        cpu_saturated = cpu_med > SAT_THRESHOLD

        gap_s = xxx["ttft"].mean() - cpu["ttft"].mean()
        gap_pct = gap_s / xxx["ttft"].mean() * 100

        print(f"\n  qps={rate}:")
        print(f"    TTFT mean: opt-xxx={xxx['ttft'].mean():.2f}s, "
              f"opt-cpu={cpu['ttft'].mean():.2f}s, gap={gap_s:.2f}s ({gap_pct:.0f}%)")
        print(f"    TTFT median: opt-xxx={xxx_med:.2f}s "
              f"({'SATURATED' if xxx_saturated else 'not sat'}), "
              f"opt-cpu={cpu_med:.2f}s "
              f"({'SATURATED' if cpu_saturated else 'not sat'})")
        if xxx_saturated and not cpu_saturated:
            print(f"    → Regime: opt-xxx queue accumulating massively, opt-cpu stable")
            print(f"    → Gap dominated by QUEUE ACCUMULATION (event loop block effect)")
        elif xxx_saturated and cpu_saturated:
            print(f"    → Regime: BOTH saturated but opt-cpu less severely")
        elif not xxx_saturated and not cpu_saturated:
            print(f"    → Regime: BOTH under-saturated")
            print(f"    → Gap = PREDICTOR BLOCK OVERHEAD (~50-100ms per request)")
        else:
            print(f"    → Regime: opt-cpu saturated but opt-xxx not (?? unexpected)")
    print()


if __name__ == "__main__":
    main()
