"""
Experiment thuyết phục: tại sao TPOT opt-cpu cao hơn opt-xxx ở rate=16?

2 hypothesis:
  Effect 1 — Batch contention: opt-cpu running batch lớn hơn → mỗi decode
             step lâu hơn → mỗi token chậm hơn → TPOT cao thật sự.
  Effect 2 — Selection effect: opt-cpu nhiều request output dài → mỗi
             request expose nhiều token decode khi batch full → mean TPOT
             bị pull up bởi long requests; per-token cost không khác.

Method — 2 test độc lập, cross-validate:
  A. ITL CDF: distribution per-token latency toàn bộ token.
     → Shift right uniform = Effect 1.
     → Distribution chồng nhau = Effect 2.

  B. TPOT bucketed by output_len: control biến output_len.
     → Trong cùng bucket output_len, opt-cpu TPOT > opt-xxx = Effect 1.
     → Trong cùng bucket TPOT bằng nhau = Effect 2 (chỉ khác do output_len mix).

Nguồn dữ liệu: SERVE_RES/{latency-*.pt, vllm-*.json} ở rate=16.
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
RATE = 16


def load(sched, rate=RATE):
    """Load .pt + .json, merge, truncate về min length."""
    pt_path = SERVE / f"latency-{sched}-Meta-Llama-3-8B-Instruct-p0-r{float(rate)}-c1.0-t60.0-o-1.pt"
    raw = torch.load(pt_path)
    pt = {k: raw[i] for i, k in enumerate(PT_KEYS) if i < len(raw)}

    js_path = sorted(glob.glob(str(SERVE / f"vllm-{float(rate)}qps-cv1.0-*-{sched}-*.json")))[-1]
    with open(js_path) as f:
        js = json.load(f)

    n = min(len(pt["ttfts"]), len(js["itls"]))
    return {
        "ttft": np.asarray(pt["ttfts"][:n], dtype=float),
        "tpot": np.asarray(pt["real_tpots"][:n], dtype=float),
        "latency": np.asarray(pt["latencies"][:n], dtype=float),
        "output_len": np.asarray(pt["actual_output_lens"][:n], dtype=float),
        "input_len": np.asarray(pt["input_lens"][:n], dtype=float),
        "itls": [list(x) for x in js["itls"][:n]],
        "duration": js["duration"],
        "n": n,
    }


def cdf(arr):
    arr = np.sort(np.asarray(arr, dtype=float))
    arr = arr[np.isfinite(arr)]
    return arr, np.arange(1, len(arr) + 1) / len(arr) if len(arr) > 0 else np.array([])


def main():
    print(f"Loading rate={RATE} data for both schedulers...")
    runs = {s: load(s) for s in SCHEDS}
    for s, r in runs.items():
        n_tokens = sum(len(seq) for seq in r["itls"])
        print(f"  {s}: n_req={r['n']}, total_tokens={n_tokens}, "
              f"output_len mean={r['output_len'].mean():.1f}, "
              f"TPOT mean={r['tpot'].mean():.3f}s")

    # ═══════════════════════════════════════════════════════════════════
    # TEST A — Mean concurrent decoding workload (EXACT computation)
    # ═══════════════════════════════════════════════════════════════════
    # Logic:
    #   Mỗi request occupies decode window độ dài = (latency - ttft) seconds.
    #   TỔNG decode-time-units (cộng dồn cho 960 requests) = Σ (latency - ttft).
    #   TIME-AVERAGE concurrent decoding = TỔNG decode-time / duration.
    #   → Đây là metric EXACT, KHÔNG phụ thuộc giả định arrival distribution.
    #
    # Bench dùng Poisson arrival (gamma(1, 1/rate) khi cv=1), nhưng số này
    # vẫn đúng vì chỉ phụ thuộc tổng decode time và experiment duration —
    # cả 2 đều đo trực tiếp.
    print("\n" + "=" * 70)
    print("TEST A — Mean concurrent decoding (EXACT, no arrival assumption)")
    print("=" * 70)

    schedulers = ["opt-xxx", "opt-cpu-warmup2.0"]
    metrics = {}
    for s in schedulers:
        r = runs[s]
        decode_dur = r["latency"] - r["ttft"]
        total_decode = decode_dur.sum()
        duration = r["duration"]
        mean_concurrent = total_decode / duration
        # Output throughput (cũng exact)
        output_thr = r["output_len"].sum() / duration
        metrics[s] = {
            "total_decode": total_decode,
            "duration": duration,
            "mean_concurrent": mean_concurrent,
            "output_thr": output_thr,
            "decode_dur": decode_dur,
        }
        print(f"  {LABELS[s]:<10}: Σdecode={total_decode:>9.0f}s / dur={duration:>5.0f}s "
              f"= mean_concurrent={mean_concurrent:>6.1f}  output_thr={output_thr:>6.1f} tok/s")

    # ───────────────── Plot A (3 panels — all EXACT) ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # A.1 — Mean concurrent decoding (BAR — exact single number)
    ax = axes[0]
    vals = [metrics[s]["mean_concurrent"] for s in schedulers]
    bars = ax.bar([LABELS[s] for s in schedulers], vals,
                  color=[COLORS[s] for s in schedulers], alpha=0.75, width=0.6)
    for bar, v, s in zip(bars, vals, schedulers):
        m = metrics[s]
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.0f}\n(={m['total_decode']:.0f}s\n / {m['duration']:.0f}s)",
                ha="center", va="bottom", fontsize=10)
    ratio = vals[1] / vals[0]
    ax.set_ylabel("mean concurrent decoding requests")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title(f"A.1 — Mean concurrent decoding (EXACT)\n"
                 f"opt-cpu cao gấp {ratio:.2f}× opt-xxx\n"
                 f"= Σ(latency-ttft) / experiment_duration")
    ax.grid(True, alpha=0.3, axis="y")

    # A.2 — Per-request decode_duration distribution (EXACT, per-request)
    ax = axes[1]
    bins = np.linspace(0, max(metrics[s]["decode_dur"].max() for s in schedulers), 40)
    for s in schedulers:
        m = metrics[s]
        ax.hist(m["decode_dur"], bins=bins, alpha=0.5,
                label=f"{LABELS[s]}: mean={m['decode_dur'].mean():.0f}s, "
                      f"Σ={m['total_decode']:.0f}s",
                color=COLORS[s])
    ax.set_xlabel("decode duration per request (s) = latency - ttft")
    ax.set_ylabel("count of requests")
    ax.set_title("A.2 — Per-request decode duration\n"
                 "Distribution opt-cpu rộng + dài hơn\n"
                 "→ tổng decode time lớn → mean concurrent cao")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # A.3 — Output throughput (EXACT)
    ax = axes[2]
    vals_thr = [metrics[s]["output_thr"] for s in schedulers]
    bars = ax.bar([LABELS[s] for s in schedulers], vals_thr,
                  color=[COLORS[s] for s in schedulers], alpha=0.75, width=0.6)
    for bar, v in zip(bars, vals_thr):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals_thr)*0.02,
                f"{v:.0f} tok/s", ha="center", va="bottom", fontsize=11)
    thr_ratio = vals_thr[1] / vals_thr[0]
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_ylim(0, max(vals_thr) * 1.2)
    ax.set_title(f"A.3 — Output throughput (EXACT)\n"
                 f"opt-cpu cao gấp {thr_ratio:.2f}× opt-xxx\n"
                 f"= Σ output_len / duration")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Test A — Why opt-cpu has higher TPOT: more concurrent decoding workload",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "tpot_test_A_exact.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: tpot_test_A_exact.png")
    print(f"\n  CAUSAL CHAIN (all metrics EXACT):")
    print(f"    1. Mean concurrent: opt-cpu {ratio:.2f}× opt-xxx")
    print(f"       → bigger running batch in vLLM")
    print(f"    2. Throughput: opt-cpu {thr_ratio:.2f}× opt-xxx")
    print(f"       → more tokens per second total, but each request shares with more")
    print(f"    3. → mỗi forward step phục vụ nhiều request hơn → step lâu hơn")
    print(f"       → mỗi request nhận token chậm hơn → TPOT cao")

    # ═══════════════════════════════════════════════════════════════════
    # TEST B — TPOT bucketed by output_len
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST B — TPOT bucketed by output_len (control biến output)")
    print("=" * 70)

    BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 100000)]
    bucket_labels = [f"{lo}-{hi if hi < 10000 else '∞'}" for lo, hi in BUCKETS]

    # Collect TPOT per bucket per scheduler
    bucket_data = {s: [[] for _ in BUCKETS] for s in SCHEDS}
    bucket_count = {s: [0] * len(BUCKETS) for s in SCHEDS}
    for sched, r in runs.items():
        for i in range(r["n"]):
            ol = r["output_len"][i]
            tpot = r["tpot"][i]
            if not np.isfinite(tpot) or not np.isfinite(ol):
                continue
            for bi, (lo, hi) in enumerate(BUCKETS):
                if lo <= ol < hi:
                    bucket_data[sched][bi].append(tpot)
                    bucket_count[sched][bi] += 1
                    break

    # Print table
    print(f"\n  {'bucket (output_len)':<22}", end="")
    print(f"{'opt-xxx':>22}{'opt-cpu':>22}{'ratio':>10}")
    print(f"  {'':22}", end="")
    print(f"{'n     mean(s)':>22}{'n     mean(s)':>22}{'cpu/xxx':>10}")
    print("  " + "-" * 76)
    for bi, label in enumerate(bucket_labels):
        x_data = bucket_data["opt-xxx"][bi]
        c_data = bucket_data["opt-cpu-warmup2.0"][bi]
        x_n, c_n = len(x_data), len(c_data)
        x_m = np.mean(x_data) if x_data else float("nan")
        c_m = np.mean(c_data) if c_data else float("nan")
        ratio = c_m / x_m if x_m > 0 else float("nan")
        print(f"  {label:<22}{x_n:>5} {x_m:>14.3f}{c_n:>5} {c_m:>14.3f}{ratio:>10.2f}×")

    # Plot B — boxplot grouped
    fig, ax = plt.subplots(figsize=(13, 7))
    n_buckets = len(BUCKETS)
    width = 0.35
    positions_x = np.arange(n_buckets) - width / 2
    positions_c = np.arange(n_buckets) + width / 2

    bp_x = ax.boxplot([bucket_data["opt-xxx"][bi] for bi in range(n_buckets)],
                      positions=positions_x, widths=width, patch_artist=True,
                      flierprops={"markersize": 3, "alpha": 0.4})
    bp_c = ax.boxplot([bucket_data["opt-cpu-warmup2.0"][bi] for bi in range(n_buckets)],
                      positions=positions_c, widths=width, patch_artist=True,
                      flierprops={"markersize": 3, "alpha": 0.4})

    for patch in bp_x["boxes"]:
        patch.set_facecolor(COLORS["opt-xxx"]); patch.set_alpha(0.6)
    for patch in bp_c["boxes"]:
        patch.set_facecolor(COLORS["opt-cpu-warmup2.0"]); patch.set_alpha(0.6)

    ax.set_xticks(np.arange(n_buckets))
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("output_len bucket (tokens)")
    ax.set_ylabel("TPOT (s/token, log scale)")
    ax.set_yscale("log")

    # Annotate counts above each pair
    for bi in range(n_buckets):
        x_n = len(bucket_data["opt-xxx"][bi])
        c_n = len(bucket_data["opt-cpu-warmup2.0"][bi])
        ymax = ax.get_ylim()[1]
        ax.text(bi, ymax * 0.7, f"n_xxx={x_n}\nn_cpu={c_n}",
                ha="center", fontsize=8, color="gray")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["opt-xxx"], alpha=0.6, label="opt-xxx"),
        Patch(facecolor=COLORS["opt-cpu-warmup2.0"], alpha=0.6, label="opt-cpu"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.set_title("B — TPOT distribution per output_len bucket (rate=16)\n"
                 "Box opt-cpu cao hơn opt-xxx trong CÙNG bucket = Effect 1 (batch contention)\n"
                 "Box gần như trùng = Effect 2 (chỉ output_len mix khác)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "tpot_test_B_bucketed.png", dpi=130)
    plt.close(fig)
    print(f"\n  Saved: tpot_test_B_bucketed.png")

    # ═══════════════════════════════════════════════════════════════════
    # COUNTER-FACTUAL — "Fair TPOT" via reweighting
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COUNTER-FACTUAL — Fair TPOT (giả sử output_len distribution như opt-xxx)")
    print("=" * 70)

    # Reweight: TPOT của opt-cpu nếu sample request theo distribution của opt-xxx
    xxx_buckets_pct = np.array([len(bucket_data["opt-xxx"][bi]) for bi in range(n_buckets)],
                               dtype=float)
    xxx_buckets_pct /= xxx_buckets_pct.sum() if xxx_buckets_pct.sum() > 0 else 1
    cpu_buckets_pct = np.array([len(bucket_data["opt-cpu-warmup2.0"][bi]) for bi in range(n_buckets)],
                               dtype=float)
    cpu_buckets_pct /= cpu_buckets_pct.sum() if cpu_buckets_pct.sum() > 0 else 1

    print(f"\n  output_len distribution (% requests):")
    print(f"  {'bucket':<14}{'opt-xxx %':>12}{'opt-cpu %':>12}")
    for bi, label in enumerate(bucket_labels):
        print(f"  {label:<14}{xxx_buckets_pct[bi]*100:>10.1f}%{cpu_buckets_pct[bi]*100:>10.1f}%")

    # Weighted mean using opt-xxx distribution
    cpu_means = np.array([np.mean(bucket_data["opt-cpu-warmup2.0"][bi])
                          if bucket_data["opt-cpu-warmup2.0"][bi] else 0
                          for bi in range(n_buckets)])
    xxx_means = np.array([np.mean(bucket_data["opt-xxx"][bi])
                          if bucket_data["opt-xxx"][bi] else 0
                          for bi in range(n_buckets)])

    cpu_actual = runs["opt-cpu-warmup2.0"]["tpot"].mean()
    xxx_actual = runs["opt-xxx"]["tpot"].mean()
    cpu_fair = (cpu_means * xxx_buckets_pct).sum()  # reweight cpu TPOT by xxx distribution
    xxx_fair = (xxx_means * xxx_buckets_pct).sum()

    print(f"\n  TPOT mean ACTUAL:")
    print(f"    opt-xxx: {xxx_actual:.3f}s")
    print(f"    opt-cpu: {cpu_actual:.3f}s  (gap +{(cpu_actual-xxx_actual)/xxx_actual*100:.0f}%)")
    print(f"\n  TPOT mean COUNTER-FACTUAL (cpu reweighted to xxx output_len distribution):")
    print(f"    opt-xxx: {xxx_fair:.3f}s (sanity check)")
    print(f"    opt-cpu: {cpu_fair:.3f}s  (gap +{(cpu_fair-xxx_fair)/xxx_fair*100:.0f}%)")
    print(f"\n  → Nếu gap reweighted ≪ gap actual → phần lớn là Effect 2 (selection)")
    print(f"  → Nếu gap reweighted ≈ gap actual → phần lớn là Effect 1 (batch contention)")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    xxx_concurrent = metrics["opt-xxx"]["mean_concurrent"]
    cpu_concurrent = metrics["opt-cpu-warmup2.0"]["mean_concurrent"]
    batch_ratio = cpu_concurrent / xxx_concurrent
    actual_gap = (cpu_actual - xxx_actual) / xxx_actual * 100
    fair_gap = (cpu_fair - xxx_fair) / xxx_fair * 100 if xxx_fair > 0 else float("nan")
    selection_share = 1 - (fair_gap / actual_gap) if actual_gap > 0 else 0

    print(f"\n  Test A (EXACT mean concurrent decoding): "
          f"opt-xxx={xxx_concurrent:.1f} vs opt-cpu={cpu_concurrent:.1f}  "
          f"({batch_ratio:.2f}× lớn hơn → batch contention)")
    print(f"  Counter-factual gap:       actual {actual_gap:+.0f}%, "
          f"fair {fair_gap:+.0f}%")
    print(f"  Selection effect contribution: ~{selection_share*100:.0f}% of gap")
    print(f"  Batch contention contribution: ~{(1-selection_share)*100:.0f}% of gap")
    print()


if __name__ == "__main__":
    main()
