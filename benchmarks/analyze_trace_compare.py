"""
Phân tích trace torch.profiler — Experiment 2.1
So sánh execute_model duration distribution giữa opt-xxx và opt-cpu-warmup2.0.

Câu hỏi:
- Decode step duration của opt-cpu có thực sự lớn hơn opt-xxx (Effect 1 — batch contention)?
- Hay 2 distribution gần như nhau (Effect 2 — selection effect)?

Bonus:
- Đo overlap ratio giữa ov_predictor.* và model_executor.execute_model
  để verify CPU/GPU concurrency assumption.

Caveat:
- Trace hiện tại bench ở rate=2 (KHÔNG phải rate=16) → server không saturated
  → kết quả không trực tiếp answer câu hỏi gốc (TPOT crossover ở rate=16).
  Output script vẫn hữu ích để verify infrastructure trace + check overlap.
- Phase label patch (decode/prefill/mixed) chưa apply → mọi execute_model
  block có cùng tên, không phân biệt được phase. Sẽ phân tích duration tổng.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "bench_analysis"
OUT.mkdir(exist_ok=True)

# Locate trace files
TRACE_XXX = ROOT / "server_tracee" / "moreh_3070636.1777969623200198734.pt.trace.json"
TRACE_CPU = ROOT / "server_tracee_ov" / "moreh_3016903.1777966092420915597.pt.trace.json"


def load_trace(path):
    """Trả list các traceEvents."""
    print(f"Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)...")
    with open(path) as f:
        d = json.load(f)
    return d["traceEvents"]


def extract_blocks(events, name_prefix):
    """Trả list (start_us, end_us, dur_us, name) cho events có name match prefix.

    torch.profiler trace có 'ph' (phase): 'X' = complete event với 'dur',
    'B'/'E' = begin/end pair. record_function blocks thường là 'X'.
    """
    blocks = []
    for e in events:
        name = e.get("name", "")
        if not name.startswith(name_prefix):
            continue
        if e.get("ph") != "X":
            continue
        if "dur" not in e or "ts" not in e:
            continue
        ts = e["ts"]
        dur = e["dur"]
        blocks.append((ts, ts + dur, dur, name))
    return blocks


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def compute_overlap_ratio(predictor_blocks, forward_blocks):
    """Tỷ lệ thời gian predictor compute trùng với model_executor.execute_model."""
    if not predictor_blocks:
        return 0.0, 0.0, 0.0
    total_pred = sum(b[2] for b in predictor_blocks)
    total_overlap = 0
    for ps, pe, _, _ in predictor_blocks:
        for fs, fe, _, _ in forward_blocks:
            total_overlap += overlap(ps, pe, fs, fe)
    return total_overlap, total_pred, total_overlap / total_pred if total_pred > 0 else 0.0


def main():
    # ───────────────── Load 2 traces ─────────────────
    events_xxx = load_trace(TRACE_XXX)
    events_cpu = load_trace(TRACE_CPU)

    # ───────────────── Extract execute_model blocks ─────────────────
    xxx_fwd = extract_blocks(events_xxx, "model_executor.execute_model")
    cpu_fwd = extract_blocks(events_cpu, "model_executor.execute_model")
    print(f"\nopt-xxx execute_model blocks: {len(xxx_fwd)}")
    print(f"opt-cpu execute_model blocks: {len(cpu_fwd)}")

    xxx_dur_ms = np.array([b[2] / 1000 for b in xxx_fwd])
    cpu_dur_ms = np.array([b[2] / 1000 for b in cpu_fwd])

    # ───────────────── Stats table ─────────────────
    print("\n" + "=" * 70)
    print(f"{'metric':<30}{'opt-xxx':>15}{'opt-cpu':>15}")
    print("=" * 70)
    for label, fn in [
        ("# blocks", lambda x: len(x)),
        ("mean (ms)", lambda x: np.mean(x)),
        ("median (ms)", lambda x: np.median(x)),
        ("p50 (ms)", lambda x: np.percentile(x, 50)),
        ("p90 (ms)", lambda x: np.percentile(x, 90)),
        ("p99 (ms)", lambda x: np.percentile(x, 99)),
        ("max (ms)", lambda x: np.max(x)),
        ("std (ms)", lambda x: np.std(x)),
        ("total (s)", lambda x: np.sum(x) / 1000),
    ]:
        print(f"{label:<30}{fn(xxx_dur_ms):>15.2f}{fn(cpu_dur_ms):>15.2f}")

    # ───────────────── Plot 1: histogram duration ─────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(min(xxx_dur_ms.min(), cpu_dur_ms.min()),
                       max(xxx_dur_ms.max(), cpu_dur_ms.max()), 50)
    ax.hist(xxx_dur_ms, bins=bins, alpha=0.5, label=f"opt-xxx (n={len(xxx_dur_ms)})",
            color="#d62728")
    ax.hist(cpu_dur_ms, bins=bins, alpha=0.5, label=f"opt-cpu (n={len(cpu_dur_ms)})",
            color="#1f77b4")
    ax.axvline(xxx_dur_ms.mean(), color="#d62728", linestyle="--", linewidth=1)
    ax.axvline(cpu_dur_ms.mean(), color="#1f77b4", linestyle="--", linewidth=1)
    ax.set_xlabel("model_executor.execute_model duration (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"Forward pass duration distribution\n"
                 f"opt-xxx mean={xxx_dur_ms.mean():.1f}ms vs opt-cpu mean={cpu_dur_ms.mean():.1f}ms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "trace_01_forward_duration_hist.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: trace_01_forward_duration_hist.png")

    # ───────────────── Plot 2: CDF duration ─────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, arr, color in [("opt-xxx", xxx_dur_ms, "#d62728"),
                               ("opt-cpu", cpu_dur_ms, "#1f77b4")]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, label=f"{label} (mean={arr.mean():.1f}ms)",
                color=color, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(0.99, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("model_executor.execute_model duration (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Forward duration CDF — kiểm tra distribution shift")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "trace_02_forward_duration_cdf.png", dpi=130)
    plt.close(fig)
    print(f"Saved: trace_02_forward_duration_cdf.png")

    # ───────────────── Plot 3: timeline duration ─────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    # Normalize timestamps relative to first event
    xxx_t0 = xxx_fwd[0][0]
    cpu_t0 = cpu_fwd[0][0]
    xxx_relative_s = np.array([(b[0] - xxx_t0) / 1e6 for b in xxx_fwd])
    cpu_relative_s = np.array([(b[0] - cpu_t0) / 1e6 for b in cpu_fwd])
    ax.plot(xxx_relative_s, xxx_dur_ms, "o-", label="opt-xxx", color="#d62728",
            alpha=0.7, markersize=4)
    ax.plot(cpu_relative_s, cpu_dur_ms, "o-", label="opt-cpu", color="#1f77b4",
            alpha=0.7, markersize=4)
    ax.set_xlabel("trace time (s, relative to first forward)")
    ax.set_ylabel("forward duration (ms)")
    ax.set_title("Forward duration over trace timeline (60 captured steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "trace_03_forward_duration_timeline.png", dpi=130)
    plt.close(fig)
    print(f"Saved: trace_03_forward_duration_timeline.png")

    # ───────────────── Predictor analysis (opt-cpu only) ─────────────────
    print("\n" + "=" * 70)
    print("OV Predictor analysis (opt-cpu only)")
    print("=" * 70)

    cpu_pred_total = extract_blocks(events_cpu, "ov_predictor.batch_total")
    cpu_pred_infer = extract_blocks(events_cpu, "ov_predictor.inference")
    cpu_pred_token = extract_blocks(events_cpu, "ov_predictor.tokenize")
    cpu_pred_submit = extract_blocks(events_cpu, "ov_predictor.submit_async")
    cpu_pred_poll = extract_blocks(events_cpu, "ov_predictor.poll_results")

    print(f"  ov_predictor.batch_total:    {len(cpu_pred_total):3} blocks")
    print(f"  ov_predictor.inference:      {len(cpu_pred_infer):3} blocks")
    print(f"  ov_predictor.tokenize:       {len(cpu_pred_token):3} blocks")
    print(f"  ov_predictor.submit_async:   {len(cpu_pred_submit):3} blocks")
    print(f"  ov_predictor.poll_results:   {len(cpu_pred_poll):3} blocks")

    if cpu_pred_total:
        print(f"\n  batch_total duration: mean={np.mean([b[2]/1000 for b in cpu_pred_total]):.1f}ms "
              f"max={max(b[2] for b in cpu_pred_total)/1000:.1f}ms")
    if cpu_pred_infer:
        print(f"  inference duration:   mean={np.mean([b[2]/1000 for b in cpu_pred_infer]):.1f}ms "
              f"max={max(b[2] for b in cpu_pred_infer)/1000:.1f}ms")
    if cpu_pred_submit:
        submit_durs = [b[2] for b in cpu_pred_submit]
        print(f"  submit_async duration: mean={np.mean(submit_durs):.1f}µs "
              f"max={max(submit_durs):.1f}µs (kỳ vọng <100µs)")
    if cpu_pred_poll:
        poll_durs = [b[2] for b in cpu_pred_poll]
        print(f"  poll_results duration: mean={np.mean(poll_durs):.1f}µs "
              f"max={max(poll_durs):.1f}µs (kỳ vọng <100µs)")

    # ───────────────── Compute overlap ratio ─────────────────
    if cpu_pred_total:
        ov_t, pt_t, ratio = compute_overlap_ratio(cpu_pred_total, cpu_fwd)
        print(f"\n  OVERLAP ANALYSIS (predictor vs forward):")
        print(f"    Total predictor compute: {pt_t/1000:.1f}ms")
        print(f"    Total overlap with forward: {ov_t/1000:.1f}ms")
        print(f"    Overlap ratio: {ratio*100:.1f}%")
        print(f"    → ≥80% kỳ vọng = predictor 'miễn phí' so với GPU forward")

    # ───────────────── Plot 4: Gantt-style overlap visualization ─────────────────
    if cpu_pred_total:
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(16, 4))

        # Normalize to first forward
        t0 = min(cpu_fwd[0][0], cpu_pred_total[0][0])
        T_END = max(cpu_fwd[-1][1], cpu_pred_total[-1][1])
        # Show first 5s window (or full if shorter)
        T_WIN = min(5.0, (T_END - t0) / 1e6)

        # Forward blocks (top row)
        for b in cpu_fwd:
            s = (b[0] - t0) / 1e6
            d = b[2] / 1e6
            if s > T_WIN:
                continue
            ax.add_patch(Rectangle((s, 1), d, 0.8, color="#1f77b4", alpha=0.7))

        # Predictor blocks (bottom row)
        for b in cpu_pred_total:
            s = (b[0] - t0) / 1e6
            d = b[2] / 1e6
            if s > T_WIN:
                continue
            ax.add_patch(Rectangle((s, 0), d, 0.8, color="#ff7f0e", alpha=0.7))

        ax.set_xlim(0, T_WIN)
        ax.set_ylim(-0.2, 2.0)
        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(["CPU predictor\n(ov_predictor.batch_total)",
                            "GPU forward\n(model_executor.execute_model)"])
        ax.set_xlabel("trace time (s)")
        ax.set_title(f"opt-cpu-warmup2.0 — CPU/GPU overlap Gantt (overlap ratio = {ratio*100:.1f}%)")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(OUT / "trace_04_overlap_gantt.png", dpi=130)
        plt.close(fig)
        print(f"\nSaved: trace_04_overlap_gantt.png")

    # ───────────────── Plot 5: Predictor duration breakdown ─────────────────
    if cpu_pred_infer and cpu_pred_token:
        fig, ax = plt.subplots(figsize=(10, 6))
        infer_dur = [b[2] / 1000 for b in cpu_pred_infer]
        token_dur = [b[2] / 1000 for b in cpu_pred_token]
        bins = np.linspace(0, max(max(infer_dur), max(token_dur)), 30)
        ax.hist(token_dur, bins=bins, alpha=0.6, label=f"tokenize (mean={np.mean(token_dur):.2f}ms)",
                color="#2ca02c")
        ax.hist(infer_dur, bins=bins, alpha=0.6, label=f"OV inference (mean={np.mean(infer_dur):.2f}ms)",
                color="#9467bd")
        ax.set_xlabel("duration (ms)")
        ax.set_ylabel("count")
        ax.set_title("Predictor compute breakdown — tokenize vs OV inference")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "trace_05_predictor_breakdown.png", dpi=130)
        plt.close(fig)
        print(f"Saved: trace_05_predictor_breakdown.png")

    # ───────────────── Conclusion text ─────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    delta_pct = (cpu_dur_ms.mean() - xxx_dur_ms.mean()) / xxx_dur_ms.mean() * 100
    print(f"opt-cpu mean forward duration vs opt-xxx: {delta_pct:+.1f}%")
    if abs(delta_pct) < 10:
        print("  → 2 distribution gần như IDENTICAL")
        print("  → Effect 1 (batch contention) KHÔNG đáng kể tại rate hiện tại")
        print("  → Nếu rate=2 → expected, vì server không saturated")
    elif delta_pct > 10:
        print("  → opt-cpu forward LÂU HƠN opt-xxx")
        print("  → Effect 1 (batch contention) ACTIVE — opt-cpu có batch lớn hơn")
    else:
        print("  → opt-cpu forward NHANH HƠN opt-xxx")
        print("  → opt-xxx có overhead khác (predictor cướp GPU?)")

    print("\nLưu ý: trace bench tại rate=2 (low load). Để verify Effect 1 ở rate=16,")
    print("cần re-run trace_ov.sh với --request-rate 16.")
    print()


if __name__ == "__main__":
    main()
