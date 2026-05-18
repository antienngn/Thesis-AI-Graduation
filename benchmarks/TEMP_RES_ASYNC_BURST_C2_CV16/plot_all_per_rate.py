"""Generate all TEMP_PROF_R8-style plots cho mỗi rate trong CV16 sweep.

Per rate sinh ra:
  - predictor_batch_size_r<rate>.png
  - score_progress_r<rate>.png
  - score_progress_2_r<rate>.png
  - batch_composition_r<rate>.png
  - batch_composition_merged_r<rate>.png
  - root_cause_r<rate>.png    (A+C+D combined)
  - root_cause_panel_{A,C,D}_r<rate>.png

Output: <CWD>/plots_per_rate/r<rate>/
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = "plots_per_rate"
os.makedirs(OUT_ROOT, exist_ok=True)

C_MM = "#2ca02c"
C_XX = "#1f77b4"
C_DECODE = "#5B9BD5"
C_PREFILL = "#ED7D31"
C_GAP = "#d62728"
LBL_MM = "opt-cpu-async-merged"
LBL_XX = "opt-xxx"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
})


# ============================================================================
# Parsers
# ============================================================================
def parse_events(path, want=None):
    """Return list of (t_rel, event, thread, extra_dict)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ev = r["event"]
            if want and ev not in want:
                continue
            try:
                ex = json.loads(r["extra_json"])
            except Exception:
                ex = {}
            try:
                t = float(r["t_rel"])
            except Exception:
                continue
            out.append((t, ev, r["thread"], ex))
    return out


def parse_arrivals(path):
    seen = {}
    for t, _, _, ex in parse_events(path, want={"request.arrival"}):
        rid = ex.get("rid")
        if rid is None:
            continue
        if rid not in seen or t < seen[rid]:
            seen[rid] = t
    return np.sort(np.array(list(seen.values()))) if seen else np.array([])


def parse_batches(path):
    t, ns, nt = [], [], []
    for ti, _, _, ex in parse_events(path, want={"model_executor.start"}):
        try:
            t.append(ti); ns.append(ex["n_seqs"]); nt.append(ex["n_tokens"])
        except KeyError:
            continue
    return np.array(t), np.array(ns), np.array(nt)


def parse_scored(path, end_event, key):
    rows = []
    for t, _, _, ex in parse_events(path, want={end_event}):
        try:
            rows.append((t, int(ex[key])))
        except (KeyError, TypeError):
            continue
    rows.sort()
    if not rows:
        return np.array([]), np.array([])
    return (np.array([r[0] for r in rows]),
            np.cumsum(np.array([r[1] for r in rows])))


def pair_calls(events, start_ev, end_ev, key):
    """Pair start/end FIFO per thread → return (t_start, batch_size, lat_ms)."""
    pend = defaultdict(list)
    pairs = []
    for t, ev, th, ex in events:
        if ev == start_ev:
            pend[th].append((t, ex.get(key)))
        elif ev == end_ev and pend[th]:
            t0, bsz = pend[th].pop(0)
            lat = ex.get("lat_ms")
            if lat is None or bsz is None:
                continue
            pairs.append((t0, bsz, lat))
    pairs.sort()
    if not pairs:
        return np.array([]), np.array([]), np.array([])
    a = np.array(pairs)
    return a[:, 0], a[:, 1].astype(int), a[:, 2]


def tick_n_running(path):
    rows = []
    for t, _, _, ex in parse_events(path, want={"scheduler.tick.end"}):
        rows.append((t, ex.get("n_running", 0)))
    rows.sort()
    return (pd.DataFrame(rows, columns=["t_rel", "n_running"])
            if rows else pd.DataFrame(columns=["t_rel", "n_running"]))


def load_lat(pt_file):
    if not os.path.exists(pt_file):
        return None, None
    d = torch.load(pt_file, weights_only=False)
    return np.array(d[0]) * 1000.0, np.array(d[1]) * 1000.0


# ============================================================================
# Plot functions (per rate)
# ============================================================================
def plot_predictor_batch_size(trace_mm, trace_xx, out, rate):
    ev_mm = parse_events(trace_mm,
        want={"predictor.worker.forward.start", "predictor.worker.forward.end"})
    t_mm, bs_mm, lat_mm = pair_calls(
        ev_mm, "predictor.worker.forward.start",
        "predictor.worker.forward.end", "chunk_size")
    ev_xx = parse_events(trace_xx,
        want={"predictor.submit.start", "predictor.submit.end"})
    t_xx, bs_xx, lat_xx = pair_calls(
        ev_xx, "predictor.submit.start", "predictor.submit.end", "n_input")

    fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex="col",
                              gridspec_kw={"hspace": 0.15, "wspace": 0.2})

    def stem(ax, t, y, c, ylabel, title):
        if len(t):
            ax.vlines(t, 0, y, colors=c, alpha=0.6, lw=1.0)
            ax.scatter(t, y, c=c, s=14, alpha=0.85)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

    stem(axes[0, 0], t_mm, bs_mm, C_MM, "chunk_size", f"{LBL_MM} · OpenVINO CPU")
    stem(axes[1, 0], t_mm, lat_mm, C_MM, "forward latency (ms)", "")
    stem(axes[0, 1], t_xx, bs_xx, C_XX, "n_input", f"{LBL_XX} · AUXLLM GPU")
    stem(axes[1, 1], t_xx, lat_xx, C_XX, "submit latency (ms)", "")

    axes[1, 0].set_xlabel("t_rel (s)")
    axes[1, 1].set_xlabel("t_rel (s)")
    fig.suptitle(
        f"Predictor batch size & latency · r={rate} QPS · cv=16",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score_progress(trace_mm, trace_xx, out, rate):
    arr_mm = parse_arrivals(trace_mm)
    arr_xx = parse_arrivals(trace_xx)
    t_sc_mm, c_sc_mm = parse_scored(
        trace_mm, "predictor.worker.forward.end", "chunk_size")
    t_sc_xx, c_sc_xx = parse_scored(
        trace_xx, "predictor.submit.end", "n_input")

    T_END = max(
        arr_mm.max() if len(arr_mm) else 0,
        arr_xx.max() if len(arr_xx) else 0,
        t_sc_mm.max() if len(t_sc_mm) else 0,
        t_sc_xx.max() if len(t_sc_xx) else 0,
    ) + 2

    def step_at(xs, ys, t):
        idx = np.searchsorted(xs, t, side="right") - 1
        idx = np.clip(idx, 0, len(ys) - 1)
        return ys[idx]

    fig, ax = plt.subplots(figsize=(16, 8))

    arr_x_mm = np.concatenate([[0], arr_mm])
    arr_y_mm = np.arange(len(arr_x_mm))
    arr_x_xx = np.concatenate([[0], arr_xx])
    arr_y_xx = np.arange(len(arr_x_xx))

    ax.step(arr_x_xx, arr_y_xx, where="post", color=C_XX, lw=2.0, ls="--",
             alpha=0.85,
             label=f"arrivals · opt-xxx run (n={len(arr_xx)})")
    ax.step(arr_x_mm, arr_y_mm, where="post", color=C_MM, lw=2.0, ls="--",
             alpha=0.85,
             label=f"arrivals · merged run (n={len(arr_mm)})")
    if len(c_sc_xx):
        ax.step(np.concatenate([[0], t_sc_xx]),
                 np.concatenate([[0], c_sc_xx]),
                 where="post", color=C_XX, lw=3.0,
                 label=f"scored · opt-xxx (total={int(c_sc_xx[-1])})")
    if len(c_sc_mm):
        ax.step(np.concatenate([[0], t_sc_mm]),
                 np.concatenate([[0], c_sc_mm]),
                 where="post", color=C_MM, lw=3.0,
                 label=f"scored · merged (total={int(c_sc_mm[-1])})")

    if len(arr_mm) and len(t_sc_mm):
        t_grid = np.linspace(0, T_END, 2000)
        ag = np.array([step_at(arr_x_mm, arr_y_mm, t) for t in t_grid])
        sg = np.array([step_at(np.concatenate([[0], t_sc_mm]),
                                 np.concatenate([[0], c_sc_mm]), t)
                        for t in t_grid])
        gap = np.maximum(ag - sg, 0)
        ax.fill_between(t_grid, sg, ag, color=C_GAP, alpha=0.18,
                         label=f"merged unscored gap (peak={int(gap.max())})")

    ax.set_xlim(0, T_END)
    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("Cumulative request count")
    ax.set_title(f"Cumulative scored progress · r={rate} QPS · cv=16",
                  fontweight="bold", loc="left")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score_progress_2(trace_mm, trace_xx, out, rate):
    arr_mm = parse_arrivals(trace_mm)
    arr_xx = parse_arrivals(trace_xx)
    t_sc_mm, c_sc_mm = parse_scored(
        trace_mm, "predictor.worker.forward.end", "chunk_size")
    t_sc_xx, c_sc_xx = parse_scored(
        trace_xx, "predictor.submit.end", "n_input")

    T_END = max(
        arr_mm.max() if len(arr_mm) else 0,
        arr_xx.max() if len(arr_xx) else 0,
        t_sc_mm.max() if len(t_sc_mm) else 0,
        t_sc_xx.max() if len(t_sc_xx) else 0,
    ) + 2

    def step_at(xs, ys, t):
        idx = np.searchsorted(xs, t, side="right") - 1
        idx = np.clip(idx, 0, len(ys) - 1)
        return ys[idx]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"hspace": 0.25})

    def panel(ax, arr_t, t_sc, c_sc, c_main, title):
        if not len(arr_t):
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            return
        arr_x = np.concatenate([[0], arr_t])
        arr_y = np.arange(len(arr_x))
        sc_x = np.concatenate([[0], t_sc])
        sc_y = np.concatenate([[0], c_sc]) if len(c_sc) else np.array([0])
        ax.step(arr_x, arr_y, where="post", color=c_main, lw=2.3, ls="--",
                 alpha=0.85, label=f"arrivals (n={len(arr_t)})")
        ax.step(sc_x, sc_y, where="post", color=c_main, lw=3.2,
                 label=f"scored (total={int(sc_y[-1])})")
        t_grid = np.linspace(0, T_END, 2000)
        ag = np.array([step_at(arr_x, arr_y, t) for t in t_grid])
        sg = np.array([step_at(sc_x, sc_y, t) for t in t_grid])
        gap = np.maximum(ag - sg, 0)
        ax.fill_between(t_grid, sg, ag, color=C_GAP, alpha=0.22,
                         label=f"unscored gap (peak={int(gap.max())})")
        ax.set_xlim(0, T_END)
        ax.set_ylabel("Cumulative request count")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    panel(axes[0], arr_xx, t_sc_xx, c_sc_xx, C_XX,
           f"opt-xxx · AUXLLM GPU · r={rate} QPS")
    panel(axes[1], arr_mm, t_sc_mm, c_sc_mm, C_MM,
           f"opt-cpu-async-merged · OpenVINO CPU · r={rate} QPS")
    axes[1].set_xlabel("t_rel (s)")
    fig.suptitle(f"Cumulative arrivals vs scored · r={rate} QPS · cv=16",
                  fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_batch_composition(trace_mm, trace_xx, out, rate):
    t_mm, ns_mm, nt_mm = parse_batches(trace_mm)
    t_xx, ns_xx, nt_xx = parse_batches(trace_xx)
    if len(t_mm) == 0 and len(t_xx) == 0:
        return
    dec_mm = np.minimum(ns_mm, nt_mm); pre_mm = nt_mm - dec_mm
    dec_xx = np.minimum(ns_xx, nt_xx); pre_xx = nt_xx - dec_xx
    T_MAX = max(t_mm.max() if len(t_mm) else 0,
                t_xx.max() if len(t_xx) else 0)

    fig, axes = plt.subplots(3, 1, figsize=(22, 13),
                              gridspec_kw={"hspace": 0.4})
    fig.suptitle(
        f"Batch composition · r={rate} QPS · cv=16 · full bench [0,{T_MAX:.0f}s]",
        fontsize=13, fontweight="bold", y=0.995,
    )

    def timeline(ax, t, dec, pre, name):
        if len(t) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            return
        o = np.argsort(t); t = t[o]; dec = dec[o]; pre = pre[o]
        w = max(np.median(np.diff(t)), 0.02)
        n_pre = int((pre > 0).sum()); n_tot = len(t)
        tot_pre = int(pre.sum()); tot_dec = int(dec.sum())
        mean_frac = (pre / np.maximum(dec + pre, 1))[pre > 0].mean() \
            if n_pre > 0 else 0
        ax.bar(t, dec, width=w, color=C_DECODE, edgecolor="none",
                label=f"decode  total={tot_dec:,}")
        ax.bar(t, pre, width=w, bottom=dec, color=C_PREFILL, edgecolor="none",
                label=f"prefill total={tot_pre:,}  "
                      f"appear {n_pre}/{n_tot} ({n_pre/n_tot*100:.1f}%, frac {mean_frac:.0%})")
        ax.set_xlim(0, T_MAX)
        ax.set_xlabel("t_rel (s)"); ax.set_ylabel("Tokens/iter")
        ax.set_title(f"{name}", fontweight="bold", loc="left")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    timeline(axes[0], t_mm, dec_mm, pre_mm, "(A) opt-cpu-async-merged")
    timeline(axes[1], t_xx, dec_xx, pre_xx, "(B) opt-xxx")

    ax = axes[2]
    edges = np.arange(0, T_MAX + 1, 1)
    centers = edges[:-1] + 0.5
    sd_mm, _ = np.histogram(t_mm, bins=edges, weights=dec_mm)
    sp_mm, _ = np.histogram(t_mm, bins=edges, weights=pre_mm)
    sd_xx, _ = np.histogram(t_xx, bins=edges, weights=dec_xx)
    sp_xx, _ = np.histogram(t_xx, bins=edges, weights=pre_xx)
    w = 0.4
    ax.bar(centers - w / 2, sd_mm, width=w, color=C_DECODE, alpha=0.95,
            label="merged·decode")
    ax.bar(centers - w / 2, sp_mm, width=w, bottom=sd_mm, color=C_PREFILL,
            alpha=0.95, label="merged·prefill")
    ax.bar(centers + w / 2, sd_xx, width=w, color=C_DECODE, alpha=0.45,
            label="opt-xxx·decode")
    ax.bar(centers + w / 2, sp_xx, width=w, bottom=sd_xx, color=C_PREFILL,
            alpha=0.45, label="opt-xxx·prefill")
    ax.set_xlim(0, T_MAX); ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("Sum tokens/1s")
    ax.set_title("(C) Sum tokens/1s bucket", fontweight="bold", loc="left")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_batch_composition_merged(trace_mm, out, rate):
    t, ns, nt = parse_batches(trace_mm)
    if len(t) < 2:
        return
    dec = np.minimum(ns, nt); pre = nt - dec
    o = np.argsort(t); t = t[o]; dec = dec[o]; pre = pre[o]
    w = max(np.median(np.diff(t)), 0.02)
    T_MAX = t.max()
    n_pre = int((pre > 0).sum()); n_tot = len(t)
    tot_pre = int(pre.sum()); tot_dec = int(dec.sum())
    mean_frac = (pre / np.maximum(dec + pre, 1))[pre > 0].mean() \
        if n_pre > 0 else 0

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(t, dec, width=w, color=C_DECODE, edgecolor="none",
            label=f"decode total={tot_dec:,}")
    ax.bar(t, pre, width=w, bottom=dec, color=C_PREFILL, edgecolor="none",
            label=f"prefill total={tot_pre:,}  "
                  f"{n_pre}/{n_tot} iter ({n_pre/n_tot*100:.1f}%, frac {mean_frac:.0%})")
    ax.set_xlim(0, T_MAX)
    ax.set_xlabel("t_rel (s)"); ax.set_ylabel("Tokens/iter")
    ax.set_title(
        f"opt-cpu-async-merged · batch composition · r={rate} QPS · cv=16",
        fontweight="bold", loc="left")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_root_cause(trace_mm, trace_xx, pt_mm, pt_xx, out_dir, rate):
    ttft_xx, tpot_xx = load_lat(pt_xx)
    ttft_mm, tpot_mm = load_lat(pt_mm)
    if ttft_xx is None or ttft_mm is None:
        return
    xx_state = tick_n_running(trace_xx)
    mm_state = tick_n_running(trace_mm)
    T_MAX = max(xx_state.t_rel.max() if len(xx_state) else 0,
                mm_state.t_rel.max() if len(mm_state) else 0, 1)

    def draw_A(ax):
        ax.plot(np.arange(len(ttft_xx)),
                 pd.Series(ttft_xx).rolling(20, min_periods=1).mean(),
                 color=C_XX, lw=2.6, label=LBL_XX)
        ax.plot(np.arange(len(ttft_mm)),
                 pd.Series(ttft_mm).rolling(20, min_periods=1).mean(),
                 color=C_MM, lw=2.6, label=LBL_MM)
        ax.set_xlabel("Request arrival index"); ax.set_ylabel("TTFT (ms, log)")
        ax.set_yscale("log")
        ax.set_title(f"(A) TTFT rolling-20 · r={rate}",
                      fontweight="bold", loc="left")
        ax.legend(loc="lower right")
        ax.grid(True, which="both", alpha=0.3)

    def draw_C(ax):
        if len(xx_state):
            ax.plot(xx_state.t_rel,
                     xx_state.n_running.rolling(50, min_periods=1).mean(),
                     color=C_XX, lw=2.0, label=LBL_XX)
        if len(mm_state):
            ax.plot(mm_state.t_rel,
                     mm_state.n_running.rolling(50, min_periods=1).mean(),
                     color=C_MM, lw=2.0, label=LBL_MM)
        ax.axhline(256, color="black", ls="--", lw=1.0, alpha=0.6,
                    label="max_num_seqs=256")
        ax.set_xlabel("t_rel (s)"); ax.set_ylabel("n_running")
        ax.set_xlim(0, T_MAX)
        ax.set_title(f"(C) n_running · r={rate}",
                      fontweight="bold", loc="left")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    def draw_D(ax):
        bins = np.linspace(0, max(tpot_xx.max(), tpot_mm.max()), 50)
        ax.hist(tpot_xx, bins=bins, alpha=0.65, color=C_XX,
                 label=f"{LBL_XX}  mean={tpot_xx.mean():.0f} med={np.median(tpot_xx):.0f}")
        ax.hist(tpot_mm, bins=bins, alpha=0.65, color=C_MM,
                 label=f"{LBL_MM}  mean={tpot_mm.mean():.0f} med={np.median(tpot_mm):.0f}")
        ax.set_xlabel("Real TPOT (ms/token)"); ax.set_ylabel("# requests")
        ax.set_title(f"(D) TPOT distribution · r={rate}",
                      fontweight="bold", loc="left")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    # Combined
    fig, axes = plt.subplots(1, 3, figsize=(22, 6),
                              gridspec_kw={"wspace": 0.25})
    fig.suptitle(
        f"Root-cause · r={rate} QPS · cv=16 · Meta-Llama-3-8B-Instruct",
        fontsize=13, fontweight="bold", y=1.02,
    )
    draw_A(axes[0]); draw_C(axes[1]); draw_D(axes[2])
    fig.savefig(os.path.join(out_dir, f"root_cause_r{rate}.png"),
                 dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Standalone panels
    for lbl, fn in [("A", draw_A), ("C", draw_C), ("D", draw_D)]:
        fig, ax = plt.subplots(figsize=(10, 6))
        fn(ax); fig.tight_layout()
        fig.savefig(os.path.join(out_dir,
                                   f"root_cause_panel_{lbl}_r{rate}.png"),
                     dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ============================================================================
# Driver
# ============================================================================
RATES = [2, 4, 8, 16, 32, 64]


def pt_path(sched, rate):
    pat = (f"latency-{sched}-Meta-Llama-3-8B-Instruct-p0-"
           f"r{rate}.0-c16.0-t60.0-o-1.pt")
    return pat if os.path.exists(pat) else None


for rate in RATES:
    out_dir = os.path.join(OUT_ROOT, f"r{rate}")
    os.makedirs(out_dir, exist_ok=True)
    trace_mm = f"trace_async-merged_r{rate}.csv"
    trace_xx = f"trace_opt-xxx_r{rate}.csv"
    pt_mm = pt_path("opt-cpu-async-merged1.0", rate)
    pt_xx = pt_path("opt-xxx", rate)
    if not (os.path.exists(trace_mm) and os.path.exists(trace_xx)):
        print(f"[skip] r={rate}: missing traces")
        continue
    print(f"\n=== r={rate} QPS ===")

    print("  predictor_batch_size...", end=" ", flush=True)
    plot_predictor_batch_size(trace_mm, trace_xx,
        os.path.join(out_dir, f"predictor_batch_size_r{rate}.png"), rate)
    print("✓")

    print("  score_progress...", end=" ", flush=True)
    plot_score_progress(trace_mm, trace_xx,
        os.path.join(out_dir, f"score_progress_r{rate}.png"), rate)
    print("✓")

    print("  score_progress_2...", end=" ", flush=True)
    plot_score_progress_2(trace_mm, trace_xx,
        os.path.join(out_dir, f"score_progress_2_r{rate}.png"), rate)
    print("✓")

    print("  batch_composition...", end=" ", flush=True)
    plot_batch_composition(trace_mm, trace_xx,
        os.path.join(out_dir, f"batch_composition_r{rate}.png"), rate)
    print("✓")

    print("  batch_composition_merged...", end=" ", flush=True)
    plot_batch_composition_merged(trace_mm,
        os.path.join(out_dir, f"batch_composition_merged_r{rate}.png"), rate)
    print("✓")

    print("  root_cause (+panels)...", end=" ", flush=True)
    if pt_mm and pt_xx:
        plot_root_cause(trace_mm, trace_xx, pt_mm, pt_xx, out_dir, rate)
        print("✓")
    else:
        print(f"skip (pt missing mm={pt_mm} xx={pt_xx})")

print(f"\nAll plots saved under: {OUT_ROOT}/")
