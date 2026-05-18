#!/usr/bin/env python3
"""analyze_trace_e2e.py — Phân tích trace_{merged,optxxx}.csv để trả lời 4 câu hỏi:
  Q1: GPU utilization & idle gap
  Q2: CPU/GPU overlap (worker thread parallel với GPU?)
  Q3: Predictor blocking (sync block tick bao lâu?)
  Q4: Predictor backpressure (queue grow không?)

Output 4 PNG + 1 TXT summary trong --dir.

Usage:
  python analyze_trace_e2e.py --dir TEMP_PROF_R8
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Parse trace CSV
# ============================================================
def parse_trace(csv_path: Path) -> List[Dict]:
    """Đọc trace CSV, return list of dict events."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                t = float(r["t_rel"])
                extra = json.loads(r["extra_json"]) if r["extra_json"] else {}
                rows.append({
                    "t_rel": t,
                    "event": r["event"],
                    "thread": r["thread"],
                    "extra": extra,
                })
            except (json.JSONDecodeError, ValueError):
                continue
    return rows


def pair_intervals(events: List[Dict], start_name: str,
                    end_name: str) -> List[Tuple[float, float, Dict]]:
    """Pair start/end events thành intervals (t0, t1, extra_from_end).
    Match theo thứ tự FIFO trên cùng thread.
    """
    intervals = []
    # Group by thread
    by_thread: Dict[str, List[Dict]] = {}
    for e in events:
        if e["event"] in (start_name, end_name):
            by_thread.setdefault(e["thread"], []).append(e)

    for thread_events in by_thread.values():
        stack = []
        for e in thread_events:
            if e["event"] == start_name:
                stack.append(e["t_rel"])
            elif e["event"] == end_name and stack:
                t0 = stack.pop(0)  # FIFO
                intervals.append((t0, e["t_rel"], e["extra"]))
    return intervals


def union_intervals(intervals: List[Tuple[float, float, Dict]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals. Return sorted disjoint."""
    if not intervals:
        return []
    sorted_intervals = sorted([(t0, t1) for t0, t1, _ in intervals])
    merged = [sorted_intervals[0]]
    for t0, t1 in sorted_intervals[1:]:
        last_t0, last_t1 = merged[-1]
        if t0 <= last_t1:
            merged[-1] = (last_t0, max(last_t1, t1))
        else:
            merged.append((t0, t1))
    return merged


def intersection_duration(a: List[Tuple[float, float]],
                           b: List[Tuple[float, float]]) -> float:
    """Total overlap duration giữa 2 lists of intervals."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ============================================================
# Metrics
# ============================================================
def _empty_metrics(label: str) -> Dict:
    """Return zero-filled metrics — dùng khi không có events."""
    return {
        "label": label, "n_events": 0, "t_max": 0,
        "n_ticks": 0, "n_gpu": 0, "n_worker": 0, "n_submit": 0,
        "gpu_busy_s": 0, "gpu_util_pct": 0,
        "worker_busy_s": 0, "worker_util_pct": 0,
        "overlap_s": 0, "overlap_pct": 0,
        "tick_durs_ms": [], "submit_durs_ms": [],
        "worker_durs_ms": [], "gpu_durs_ms": [], "idle_gaps_ms": [],
        "intervals": {"tick": [], "gpu": [], "worker": [], "submit": []},
    }


def compute_metrics(events: List[Dict], label: str) -> Dict:
    """Compute high-level metrics từ events."""
    if not events:
        return _empty_metrics(label)

    t_max = max(e["t_rel"] for e in events)

    tick_intervals = pair_intervals(events, "scheduler.tick.start",
                                     "scheduler.tick.end")
    gpu_intervals = pair_intervals(events, "model_executor.start",
                                    "model_executor.end")
    worker_intervals = pair_intervals(events,
                                       "predictor.worker.forward.start",
                                       "predictor.worker.forward.end")
    submit_intervals = pair_intervals(events, "predictor.submit.start",
                                       "predictor.submit.end")

    gpu_union = union_intervals(gpu_intervals)
    worker_union = union_intervals(worker_intervals)

    gpu_busy = sum(t1 - t0 for t0, t1 in gpu_union)
    worker_busy = sum(t1 - t0 for t0, t1 in worker_union)
    overlap = intersection_duration(gpu_union, worker_union)

    tick_durs = [(t1 - t0) * 1000 for t0, t1, _ in tick_intervals]  # ms
    submit_durs = [(t1 - t0) * 1000 for t0, t1, _ in submit_intervals]
    worker_durs = [(t1 - t0) * 1000 for t0, t1, _ in worker_intervals]
    gpu_durs = [(t1 - t0) * 1000 for t0, t1, _ in gpu_intervals]

    # Idle gap: gap giữa GPU intervals
    idle_gaps = []
    for i in range(1, len(gpu_union)):
        gap = gpu_union[i][0] - gpu_union[i - 1][1]
        if gap > 0:
            idle_gaps.append(gap * 1000)

    return {
        "label": label,
        "n_events": len(events),
        "t_max": t_max,
        "n_ticks": len(tick_intervals),
        "n_gpu": len(gpu_intervals),
        "n_worker": len(worker_intervals),
        "n_submit": len(submit_intervals),
        "gpu_busy_s": gpu_busy,
        "gpu_util_pct": gpu_busy / t_max * 100 if t_max > 0 else 0,
        "worker_busy_s": worker_busy,
        "worker_util_pct": worker_busy / t_max * 100 if t_max > 0 else 0,
        "overlap_s": overlap,
        "overlap_pct": (overlap / worker_busy * 100) if worker_busy > 0 else 0,
        "tick_durs_ms": tick_durs,
        "submit_durs_ms": submit_durs,
        "worker_durs_ms": worker_durs,
        "gpu_durs_ms": gpu_durs,
        "idle_gaps_ms": idle_gaps,
        "intervals": {
            "tick": tick_intervals,
            "gpu": gpu_intervals,
            "worker": worker_intervals,
            "submit": submit_intervals,
        },
    }


# ============================================================
# Plots
# ============================================================
def plot_gantt(metrics_m, metrics_o, out: Path, t_start=None, t_window=None):
    """Plot 1: Gantt timeline 3 lanes.

    Mặc định vẽ TOÀN BỘ benchmark (t=0 → t_max). Truyền t_start/t_window
    để zoom. Dùng broken_barh để xử lý hàng nghìn intervals nhanh.

    Nếu chỉ 1 scheduler có data thì chỉ vẽ scheduler đó (không pad panel rỗng).
    """
    panels = []
    for m, title in [(metrics_m, "opt-cpu-async-merged1.0"),
                      (metrics_o, "opt-xxx")]:
        if m["n_events"] > 0:
            panels.append((m, title))

    if not panels:
        print(f"  [skip] no data for either scheduler — gantt not written")
        return

    fig, axes = plt.subplots(len(panels), 1,
                              figsize=(18, 3.5 * len(panels)), sharex=False,
                              squeeze=False)
    for ax, (m, title) in zip(axes[:, 0], panels):
        # Window: full bench by default
        lo = t_start if t_start is not None else 0.0
        hi = (t_start + t_window) if (t_start is not None and t_window is not None) else m["t_max"]

        def _filter(intervals):
            return [(t0, t1 - t0) for t0, t1, _ in intervals
                    if t1 >= lo and t0 <= hi]

        tick_segs = _filter(m["intervals"]["tick"])
        worker_segs = _filter(m["intervals"]["worker"])
        gpu_segs = _filter(m["intervals"]["gpu"])

        # broken_barh: (y_low, height) + list of (x, width)
        if tick_segs:
            ax.broken_barh(tick_segs, (1.65, 0.7),
                            facecolors="#5B9BD5", edgecolors="none")
        if worker_segs:
            ax.broken_barh(worker_segs, (0.65, 0.7),
                            facecolors="#FFC000", edgecolors="none")
        if gpu_segs:
            ax.broken_barh(gpu_segs, (-0.35, 0.7),
                            facecolors="#70AD47", edgecolors="none")

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["GPU (model_executor)", "Worker (predictor)",
                             "Main (scheduler.tick)"])
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t_rel (seconds from serve_start)")
        ax.set_title(
            f"{title} — full bench [{lo:.1f}, {hi:.1f}]s "
            f"(ticks={len(tick_segs)}, worker={len(worker_segs)}, gpu={len(gpu_segs)})"
        )
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Plot 1 — Gantt timeline (CPU/GPU overlap) — full benchmark",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_summary_bars(metrics_m, metrics_o, out: Path):
    """Plot 2: 4 panel bar chart cho Q1+Q2 quantitative."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = ["merged", "opt-xxx"]
    colors = ["#4472C4", "#ED7D31"]

    # Panel A: GPU busy %
    vals = [metrics_m["gpu_util_pct"], metrics_o["gpu_util_pct"]]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_title("GPU utilization %")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    # Panel B: GPU idle gap mean (ms)
    g_m = np.mean(metrics_m["idle_gaps_ms"]) if metrics_m["idle_gaps_ms"] else 0
    g_o = np.mean(metrics_o["idle_gaps_ms"]) if metrics_o["idle_gaps_ms"] else 0
    axes[1].bar(labels, [g_m, g_o], color=colors)
    axes[1].set_title("GPU idle gap mean (ms)")
    axes[1].set_ylabel("ms")
    for i, v in enumerate([g_m, g_o]):
        axes[1].text(i, v + max(g_m, g_o) * 0.02, f"{v:.1f}", ha="center", fontsize=10)

    # Panel C: CPU/GPU overlap %
    vals = [metrics_m["overlap_pct"], metrics_o["overlap_pct"]]
    axes[2].bar(labels, vals, color=colors)
    axes[2].set_title("CPU/GPU overlap %\n(worker∩GPU / worker)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(vals):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    # Panel D: Worker util %
    vals = [metrics_m["worker_util_pct"], metrics_o["worker_util_pct"]]
    axes[3].bar(labels, vals, color=colors)
    axes[3].set_title("Worker utilization %")
    axes[3].set_ylabel("%")
    axes[3].set_ylim(0, max(100, max(vals) * 1.2))
    for i, v in enumerate(vals):
        axes[3].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Plot 2 — Summary bars (Q1 + Q2)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_tick_breakdown(metrics_m, metrics_o, out: Path):
    """Plot 3: Box plot + stacked bar breakdown cho Q3."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot of tick durations
    ax = axes[0]
    data = [metrics_m["tick_durs_ms"], metrics_o["tick_durs_ms"]]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                     labels=["merged", "opt-xxx"], showfliers=True,
                     flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], ["#4472C4", "#ED7D31"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xscale("log")
    ax.set_xlabel("scheduler.tick CPU duration (ms, log)")
    ax.set_title("Distribution tick durations")
    ax.grid(True, alpha=0.3, which="both")
    # Annotate median
    for i, durs in enumerate(data):
        if durs:
            med = np.median(durs)
            ax.text(med, i + 1.3, f"median={med:.2f}ms", ha="center",
                    fontsize=9, color="darkred")

    # Right: stacked bar of mean tick breakdown
    ax = axes[1]
    # Per scheduler: mean total tick, mean submit (predictor block in tick)
    mean_tick_m = np.mean(metrics_m["tick_durs_ms"]) if metrics_m["tick_durs_ms"] else 0
    mean_tick_o = np.mean(metrics_o["tick_durs_ms"]) if metrics_o["tick_durs_ms"] else 0
    mean_submit_m = np.mean(metrics_m["submit_durs_ms"]) if metrics_m["submit_durs_ms"] else 0
    mean_submit_o = np.mean(metrics_o["submit_durs_ms"]) if metrics_o["submit_durs_ms"] else 0
    # "scheduler logic" = tick - submit
    sched_m = max(0, mean_tick_m - mean_submit_m)
    sched_o = max(0, mean_tick_o - mean_submit_o)

    ax.barh(["merged", "opt-xxx"], [mean_submit_m, mean_submit_o],
             color="#ED7D31", label="predictor.submit (sync block for opt-xxx)")
    ax.barh(["merged", "opt-xxx"], [sched_m, sched_o],
             left=[mean_submit_m, mean_submit_o],
             color="#5B9BD5", label="scheduler logic (filter + sort)")
    ax.set_xlabel("ms")
    ax.set_title("Mean tick breakdown")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    for i, (s, t) in enumerate(zip([mean_submit_m, mean_submit_o],
                                     [mean_tick_m, mean_tick_o])):
        ax.text(t + max(mean_tick_m, mean_tick_o) * 0.02, i, f"{t:.2f}ms total",
                va="center", fontsize=9)

    fig.suptitle("Plot 3 — Tick duration analysis (Q3: predictor blocking)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_predictor_backpressure(metrics_m, metrics_o, tick_csv_m, tick_csv_o,
                                  out: Path):
    """Plot 4: Predictor backpressure time series cho Q4.

    Dùng tick_profile CSV để lấy stream_queue_depth + stream_in_flight.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    for ax, tick_csv, label in zip(axes,
                                     [tick_csv_m, tick_csv_o],
                                     ["opt-cpu-async-merged1.0", "opt-xxx"]):
        if not tick_csv.exists():
            ax.text(0.5, 0.5,
                     f"No tick_profile data for {label}\n"
                     "(opt-xxx không có streaming queue)",
                     ha="center", va="center", transform=ax.transAxes,
                     fontsize=10)
            ax.set_title(label)
            continue
        import csv as _csv
        ts, qd, inf = [], [], []
        with open(tick_csv) as f:
            for r in _csv.DictReader(f):
                try:
                    ts.append(float(r["t_rel"]))
                    qd.append(int(r.get("stream_queue_depth", 0) or 0))
                    inf.append(int(r.get("stream_in_flight", 0) or 0))
                except (ValueError, KeyError):
                    continue
        if not ts:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                     transform=ax.transAxes)
            continue
        ax.fill_between(ts, qd, alpha=0.4, color="tab:blue",
                          label="stream_queue_depth")
        ax.fill_between(ts, inf, alpha=0.4, color="tab:orange",
                          label="stream_in_flight")
        ax.set_ylabel("Count")
        ax.set_xlabel("t_rel (s)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Plot 4 — Predictor backpressure (Q4: queue grow không?)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_tick_rate_histogram(metrics_m, metrics_o, out: Path, bucket_s: float = 5.0):
    """Plot 5: tick/s theo time bucket — bar histogram + GPU forward median overlay.

    Cho thấy phase nào engine loop quay nhanh/chậm (thường tỉ lệ nghịch với GPU
    forward duration → prefill-heavy phase = tick/s thấp, decode steady = tick/s
    cao).
    """
    panels = []
    for m, title in [(metrics_m, "opt-cpu-async-merged1.0"),
                      (metrics_o, "opt-xxx")]:
        if m["n_events"] > 0:
            panels.append((m, title))
    if not panels:
        print(f"  [skip] no data — tick-rate histogram not written")
        return

    fig, axes = plt.subplots(len(panels), 1,
                              figsize=(16, 3.5 * len(panels)), sharex=False,
                              squeeze=False)
    for ax, (m, title) in zip(axes[:, 0], panels):
        tick_starts = [t0 for t0, _, _ in m["intervals"]["tick"]]
        gpu_intervals = m["intervals"]["gpu"]
        if not tick_starts:
            ax.text(0.5, 0.5, f"No ticks for {title}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        t_max = m["t_max"]
        edges = np.arange(0, t_max + bucket_s, bucket_s)
        counts, _ = np.histogram(tick_starts, bins=edges)
        rate = counts / bucket_s  # tick/s

        centers = edges[:-1] + bucket_s / 2

        # GPU forward median per bucket
        gpu_med = np.full_like(centers, np.nan, dtype=float)
        for i, c in enumerate(centers):
            lo, hi = edges[i], edges[i+1]
            durs = [(t1 - t0) * 1000 for t0, t1, _ in gpu_intervals
                    if lo <= t0 < hi]
            if durs:
                gpu_med[i] = np.median(durs)

        # Bar = tick rate (left axis)
        ax.bar(centers, rate, width=bucket_s * 0.9,
                color="#5B9BD5", edgecolor="navy", linewidth=0.3,
                label=f"tick/s (bucket={int(bucket_s)}s)")
        ax.set_xlabel("t_rel (seconds from serve_start)")
        ax.set_ylabel("ticks per second", color="navy")
        ax.tick_params(axis="y", labelcolor="navy")
        ax.set_xlim(0, t_max)
        ax.grid(True, alpha=0.3, axis="y")

        # Overlay GPU forward median (right axis, log scale)
        ax2 = ax.twinx()
        ax2.plot(centers, gpu_med, color="#C00000", marker=".",
                  markersize=4, linewidth=1.2, label="GPU forward median (ms)")
        ax2.set_ylabel("GPU forward median (ms, log)", color="#C00000")
        ax2.tick_params(axis="y", labelcolor="#C00000")
        ax2.set_yscale("log")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=9)

        ax.set_title(
            f"{title} — total ticks={len(tick_starts)} over {t_max:.1f}s "
            f"(avg {len(tick_starts)/t_max:.1f}/s)"
        )

    fig.suptitle("Plot 5 — Tick rate vs GPU forward duration (per time bucket)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Summary text
# ============================================================
def write_summary(metrics_m, metrics_o, out: Path):
    lines = []
    lines.append("=" * 78)
    lines.append(f" merged vs opt-xxx @ rate=8")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'Metric':<40} {'merged':>15} {'opt-xxx':>15}")
    lines.append("-" * 78)

    def fmt(v, unit="", prec=2):
        if v is None:
            return "n/a"
        if unit == "%":
            return f"{v:.1f}%"
        if unit == "ms":
            return f"{v:.{prec}f} ms"
        if unit == "s":
            return f"{v:.{prec}f} s"
        return f"{v:.{prec}f}"

    def stat(name, key, unit="", prec=2):
        return f"{name:<40} {fmt(metrics_m.get(key, 0), unit, prec):>15} {fmt(metrics_o.get(key, 0), unit, prec):>15}"

    def mean(name, key_list, unit="ms", prec=2):
        a = np.mean(metrics_m[key_list]) if metrics_m[key_list] else 0
        b = np.mean(metrics_o[key_list]) if metrics_o[key_list] else 0
        return f"{name:<40} {fmt(a, unit, prec):>15} {fmt(b, unit, prec):>15}"

    def percentile(name, key_list, p=99, unit="ms", prec=2):
        a = np.percentile(metrics_m[key_list], p) if metrics_m[key_list] else 0
        b = np.percentile(metrics_o[key_list], p) if metrics_o[key_list] else 0
        return f"{name:<40} {fmt(a, unit, prec):>15} {fmt(b, unit, prec):>15}"

    lines.append(stat("Total bench duration", "t_max", "s"))
    lines.append(stat("Scheduler ticks", "n_ticks"))
    lines.append(stat("Model executor calls", "n_gpu"))
    lines.append(stat("Predictor submit calls", "n_submit"))
    lines.append(stat("Predictor worker forwards", "n_worker"))
    lines.append("")
    lines.append("--- GPU (Q1) ---")
    lines.append(stat("GPU utilization", "gpu_util_pct", "%"))
    lines.append(mean("GPU forward duration mean", "gpu_durs_ms"))
    lines.append(mean("GPU idle gap mean", "idle_gaps_ms"))
    lines.append("")
    lines.append("--- CPU/GPU overlap (Q2) ---")
    lines.append(stat("Worker utilization", "worker_util_pct", "%"))
    lines.append(stat("CPU/GPU overlap %", "overlap_pct", "%"))
    lines.append("")
    lines.append("--- Scheduler tick (Q3) ---")
    lines.append(mean("Tick CPU duration mean", "tick_durs_ms"))
    lines.append(percentile("Tick CPU duration P50", "tick_durs_ms", 50))
    lines.append(percentile("Tick CPU duration P99", "tick_durs_ms", 99))
    lines.append(mean("Predictor submit duration mean", "submit_durs_ms"))
    lines.append("")
    lines.append("--- Predictor (Q4) ---")
    lines.append(mean("Predictor forward duration mean", "worker_durs_ms"))
    lines.append("")

    # Auto-generated conclusion
    lines.append("=" * 78)
    lines.append(" Auto-generated conclusion:")
    lines.append("=" * 78)
    ratio_tick = (np.mean(metrics_o["tick_durs_ms"])
                  / np.mean(metrics_m["tick_durs_ms"])
                  if metrics_m["tick_durs_ms"] and metrics_o["tick_durs_ms"] else 1)
    lines.append(f"- merged tick CPU work ngắn hơn opt-xxx {ratio_tick:.1f}× "
                 "(submit non-blocking vs sync block)")
    lines.append(f"- merged CPU/GPU overlap: {metrics_m['overlap_pct']:.1f}% "
                 "→ TRUE PARALLEL" if metrics_m['overlap_pct'] > 50 else
                 f"- merged CPU/GPU overlap: {metrics_m['overlap_pct']:.1f}% "
                 "→ overlap chưa optimal")
    lines.append(f"- GPU util: merged={metrics_m['gpu_util_pct']:.1f}% "
                 f"vs opt-xxx={metrics_o['gpu_util_pct']:.1f}%")
    if metrics_m["worker_durs_ms"] and metrics_o["submit_durs_ms"]:
        m_pred = np.mean(metrics_m["worker_durs_ms"])
        o_pred = np.mean(metrics_o["submit_durs_ms"])
        lines.append(f"- Predictor forward latency: merged={m_pred:.1f}ms "
                     f"vs opt-xxx={o_pred:.1f}ms")
        if m_pred > o_pred * 2:
            lines.append(f"  → BOTTLENECK: merged predictor chậm {m_pred/o_pred:.1f}× "
                         "(CPU OV vs GPU AUXLLM)")
            lines.append("  → Suggest: tăng OV_CHUNK_SIZE để amortize")

    out.write_text("\n".join(lines))
    print(f"  wrote {out}")
    print()
    print("\n".join(lines))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory chứa trace_*.csv")
    ap.add_argument("--gantt-start", type=float, default=None,
                     help="Start time (s) for Gantt zoom (default: 0 = full bench)")
    ap.add_argument("--gantt-window", type=float, default=None,
                     help="Window length (s) for Gantt zoom (default: t_max = full bench)")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"Error: {d} không tồn tại")
        return

    print(f"Loading traces from {d} ...")
    events_m = parse_trace(d / "trace_merged.csv")
    events_o = parse_trace(d / "trace_optxxx.csv")
    print(f"  merged: {len(events_m)} events")
    print(f"  opt-xxx: {len(events_o)} events")

    print("Computing metrics...")
    metrics_m = compute_metrics(events_m, "merged")
    metrics_o = compute_metrics(events_o, "opt-xxx")

    print("Generating plots...")
    plot_gantt(metrics_m, metrics_o, d / "gantt_timeline.png",
               t_start=args.gantt_start, t_window=args.gantt_window)
    plot_summary_bars(metrics_m, metrics_o, d / "summary_bars.png")
    plot_tick_breakdown(metrics_m, metrics_o, d / "tick_breakdown.png")
    plot_predictor_backpressure(
        metrics_m, metrics_o,
        d / "tick_profile_merged.csv", d / "tick_profile_optxxx.csv",
        d / "predictor_backpressure.png",
    )
    plot_tick_rate_histogram(metrics_m, metrics_o, d / "tick_rate_histogram.png")

    print("Writing summary...")
    write_summary(metrics_m, metrics_o, d / "analysis_summary.txt")


if __name__ == "__main__":
    main()
