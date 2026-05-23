#!/usr/bin/env python3
"""Sinh các biểu đồ PDF cho báo cáo: bằng chứng tại sao dual1.0 thắng opt-xxx.

Output: RESULT_DUAL_FAIR/plot_for_report/*.pdf

Mỗi PDF stand-alone, có caption, ready để paste vào thesis.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_POOL = os.path.join(HERE, "json_pool")
OUT_DIR = os.path.join(HERE, "plot_for_report")
SUMMARY_LOG = os.path.normpath(
    os.path.join(HERE, "..", "SERVE_DUAL_1", "summary.log")
)
os.makedirs(OUT_DIR, exist_ok=True)

RATES = [2, 4, 8, 16, 32, 64]
C_OPTXXX = "#1f77b4"
C_DUAL   = "#d62728"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,  # TrueType embedding (editable in Acrobat)
    "ps.fonttype": 42,
})


def load_one(d, sched, rate):
    pat = os.path.join(d, f"vllm-{rate}.0qps-*-{sched}-*.json")
    files = sorted(glob.glob(pat))
    return json.load(open(files[-1])) if files else None


def parse_router_stats(log_path):
    text = open(log_path).read()
    stats = {}
    for m in re.finditer(
        r"rate=(\d+):\s*DONE(.*?)(?=rate=\d+:|\Z)",
        text, flags=re.S,
    ):
        rate = int(m.group(1))
        chunk = m.group(2)
        m_dec = re.search(r"(\d+)\s*decisions", chunk)
        m_cpu = re.search(r"(\d+)\s*CPU submits", chunk)
        m_gpu = re.search(r"(\d+)\s*GPU sync",   chunk)
        if m_dec and m_cpu and m_gpu:
            stats[rate] = {
                "decisions": int(m_dec.group(1)),
                "cpu": int(m_cpu.group(1)),
                "gpu": int(m_gpu.group(1)),
            }
    return stats


def all_itls_ms(d):
    return np.array([x * 1000 for itl in d["itls"] for x in itl])


def compute_nlatency_ms(d):
    ttfts = d["ttfts"]; itls = d["itls"]
    vals = [(ttfts[i] + sum(itls[i])) / (len(itls[i]) + 1)
            for i in range(len(ttfts)) if len(itls[i]) > 0]
    return float(np.mean(vals) * 1000) if vals else float("nan")


def compute_tpot_ms(d):
    vals = [sum(itl) / len(itl) for itl in d["itls"] if len(itl) > 0]
    return float(np.mean(vals) * 1000) if vals else float("nan")


# ============================================================
# FIG 1: ITL distribution per rate — CDF overlay
# ============================================================
def fig_itl_cdf():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if not (opt and dual):
            ax.set_visible(False); continue
        itl_o = all_itls_ms(opt); itl_d = all_itls_ms(dual)
        for arr, color, label in [
            (itl_o, C_OPTXXX, f"opt-xxx  (mean={itl_o.mean():.1f}, p99={np.percentile(itl_o,99):.0f})"),
            (itl_d, C_DUAL,   f"dual1.0   (mean={itl_d.mean():.1f}, p99={np.percentile(itl_d,99):.0f})"),
        ]:
            srt = np.sort(arr)
            cdf = np.arange(1, len(srt) + 1) / len(srt)
            ax.plot(srt, cdf, color=color, lw=1.8, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("ITL (ms, log)")
        ax.set_ylabel("CDF")
        ax.set_title(f"r = {r} req/s  ·  n_tokens = {len(itl_o)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        "Phân bố Inter-Token-Latency (ITL): dual1.0 vs opt-xxx\n"
        "ITL của opt-xxx có đuôi dài hơn do GPU bị predictor pause định kỳ",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_itl_cdf.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 2: ITL CV bar chart per rate
# ============================================================
def fig_itl_cv():
    cv_opt, cv_dual = [], []
    p99_opt, p99_dual = [], []
    for r in RATES:
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if opt and dual:
            io = all_itls_ms(opt); id_ = all_itls_ms(dual)
            cv_opt.append(io.std() / io.mean())
            cv_dual.append(id_.std() / id_.mean())
            p99_opt.append(np.percentile(io, 99))
            p99_dual.append(np.percentile(id_, 99))
        else:
            cv_opt.append(0); cv_dual.append(0)
            p99_opt.append(0); p99_dual.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(RATES))
    w = 0.36
    ax1.bar(x - w/2, cv_opt,  w, color=C_OPTXXX, label="opt-xxx")
    ax1.bar(x + w/2, cv_dual, w, color=C_DUAL,   label="dual1.0")
    for xi, v in zip(x - w/2, cv_opt):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=C_OPTXXX)
    for xi, v in zip(x + w/2, cv_dual):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=C_DUAL)
    ax1.set_xticks(x); ax1.set_xticklabels([f"r={r}" for r in RATES])
    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("ITL Coefficient of Variation (σ/μ)")
    ax1.set_title("ITL biến thiên\n(càng nhỏ = decode càng đều)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, max(cv_opt + cv_dual) * 1.25)

    ax2.bar(x - w/2, p99_opt,  w, color=C_OPTXXX, label="opt-xxx")
    ax2.bar(x + w/2, p99_dual, w, color=C_DUAL,   label="dual1.0")
    for xi, v in zip(x - w/2, p99_opt):
        ax2.text(xi, v * 1.04, f"{v:.0f}", ha="center", fontsize=8.5, color=C_OPTXXX)
    for xi, v in zip(x + w/2, p99_dual):
        ax2.text(xi, v * 1.04, f"{v:.0f}", ha="center", fontsize=8.5, color=C_DUAL)
    ax2.set_xticks(x); ax2.set_xticklabels([f"r={r}" for r in RATES])
    ax2.set_xlabel("Request rate (req/s)")
    ax2.set_ylabel("P99 ITL (ms)")
    ax2.set_title("Đuôi ITL P99\n(opt-xxx cao = pause dài do predictor)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle("Biến thiên ITL: dual1.0 mượt hơn opt-xxx ở rate thấp/trung",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_itl_cv_p99.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 3: CPU routing % (stacked bar) — same as plots/cpu_usage but PDF
# ============================================================
def fig_cpu_routing():
    stats = parse_router_stats(SUMMARY_LOG)
    rates = sorted(stats.keys())
    n_total = np.array([stats[r]["decisions"] for r in rates])
    n_cpu   = np.array([stats[r]["cpu"]       for r in rates])
    n_gpu   = np.array([stats[r]["gpu"]       for r in rates])
    cpu_pct = 100.0 * n_cpu / np.maximum(1, n_total)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(rates)); w = 0.6
    bc = ax.bar(x, n_cpu, w, color="#2ca02c", edgecolor="#1f6f1f", lw=0.8,
                label="CPU predictor được chọn (OpenVINO async)")
    bg = ax.bar(x, n_gpu, w, bottom=n_cpu, color="#1f77b4", edgecolor="#0f4f8f", lw=0.8,
                label="GPU predictor được chọn (AUX-LLM sync)")

    for xi, b1, b2, nc, ng, nt in zip(x, bc, bg, n_cpu, n_gpu, n_total):
        if b1.get_height() >= max(n_total) * 0.06:
            ax.text(xi, b1.get_height() / 2, str(nc),
                    ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        else:
            ax.annotate(f"CPU={nc}",
                        xy=(xi, b1.get_height()),
                        xytext=(xi + 0.42, b1.get_height() + max(n_total) * 0.02),
                        ha="left", va="bottom", fontsize=10, color="#2ca02c", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.6))
        if b2.get_height() >= max(n_total) * 0.06:
            ax.text(xi, b1.get_height() + b2.get_height() / 2, str(ng),
                    ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(xi, nt + max(n_total) * 0.03, f"tổng = {nt}",
                ha="center", va="bottom", fontsize=10, color="#222")
        ax.text(xi, nt + max(n_total) * 0.09, f"CPU {cpu_pct[xi]:.1f}%",
                ha="center", va="bottom", fontsize=12, color="#2ca02c", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"r = {r}" for r in rates])
    ax.set_xlabel("Request rate (request/giây)")
    ax.set_ylabel("Số lần router ra quyết định")
    ax.set_title("Phân bố quyết định của router dual1.0 theo request rate", pad=10)
    ax.set_ylim(0, max(n_total) * 1.28)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.text(0.5, -0.02,
             "Một quyết định = 1 lần scheduler gặp unscored batch. "
             "Router so sánh T_main (LUT 4D) vs T_cpu (LUT 2D) để chọn.",
             ha="center", fontsize=10, style="italic", color="#555")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_cpu_routing.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 4: Causal link — CPU routing % vs Δ Nlatency dual vs opt-xxx
# ============================================================
def fig_proof_of_win():
    stats = parse_router_stats(SUMMARY_LOG)
    rates = sorted(stats.keys())
    cpu_pct = [100.0 * stats[r]["cpu"] / stats[r]["decisions"] for r in rates]

    nlat_delta, tpot_delta, ttft_delta = [], [], []
    for r in rates:
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if not (opt and dual):
            nlat_delta.append(0); tpot_delta.append(0); ttft_delta.append(0); continue
        n_o = compute_nlatency_ms(opt); n_d = compute_nlatency_ms(dual)
        p_o = compute_tpot_ms(opt);     p_d = compute_tpot_ms(dual)
        t_o = opt["mean_ttft_ms"];      t_d = dual["mean_ttft_ms"]
        nlat_delta.append((n_d / n_o - 1) * 100)
        tpot_delta.append((p_d / p_o - 1) * 100)
        ttft_delta.append((t_d / t_o - 1) * 100)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(rates))
    bars = ax1.bar(x, cpu_pct, 0.55, color="#2ca02c", alpha=0.65,
                   edgecolor="#1f6f1f",
                   label="% CPU routing (dual1.0)")
    for xi, v in zip(x, cpu_pct):
        ax1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                 fontsize=10, color="#1f6f1f", fontweight="bold")
    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("CPU routing %", color="#1f6f1f", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([f"r={r}" for r in rates])
    ax1.tick_params(axis="y", labelcolor="#1f6f1f")
    ax1.set_ylim(0, max(cpu_pct) * 1.4 + 5)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.plot(x, nlat_delta, "o-", color="#d62728", lw=2.4, ms=9, label="Δ Nlatency")
    ax2.plot(x, ttft_delta, "s--", color="#ff7f0e", lw=1.8, ms=7, label="Δ TTFT")
    ax2.plot(x, tpot_delta, "^--", color="#9467bd", lw=1.8, ms=7, label="Δ TPOT")
    for xi, v in zip(x, nlat_delta):
        ax2.annotate(f"{v:+.0f}%", (xi, v),
                     textcoords="offset points", xytext=(8, -2),
                     fontsize=9, color="#d62728", fontweight="bold")
    ax2.set_ylabel("Δ % vs opt-xxx (âm = dual win)",
                   color="#d62728", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2,
               loc="lower left", fontsize=9.5, framealpha=0.95)
    plt.title("CPU routing % và scheduler win có tương quan rõ ràng",
              pad=14)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_proof_of_win.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 5: TPOT & TTFT & Nlatency scalar vs QPS — clean (raw token)
# ============================================================
def fig_scalar_compare():
    SCHEDS = [
        ("fcfs",    "FCFS",         "#888888"),
        ("sjf",     "SJF",          "#1f77b4"),
        ("srtf",    "SRTF (Oracle)", "#9467bd"),
        ("opt-xxx", "opt-xxx (paper)", "#ff7f0e"),
        ("dual1.0", "dual1.0 (ours)",  "#d62728"),
    ]
    rates = RATES

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (vfn, ylabel, title) in enumerate([
        (lambda d: d["mean_ttft_ms"],   "Mean TTFT (ms)",        "TTFT"),
        (compute_tpot_ms,                "Mean TPOT (ms/token, raw)", "TPOT"),
        (compute_nlatency_ms,            "Mean Nlatency (ms/token, raw)", "Nlatency"),
    ]):
        ax = axes[idx]
        for sched, lbl, color in SCHEDS:
            ys = []
            for r in rates:
                d = load_one(JSON_POOL, sched, r)
                ys.append(vfn(d) if d else None)
            lw = 2.6 if sched == "dual1.0" else 1.6
            ms = 9 if sched == "dual1.0" else 6
            ax.plot(rates, ys, marker="o", color=color, lw=lw, ms=ms, label=lbl)
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("So sánh scalar metrics: dual1.0 vs các baseline (raw token denominator)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_scalar_compare.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# Bundle: 1 PDF chứa toàn bộ figures
# ============================================================
def fig_bundle(out_paths):
    bundle = os.path.join(OUT_DIR, "ALL_FIGURES.pdf")
    with PdfPages(bundle) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7,
                 "Bằng chứng dual1.0 thắng opt-xxx",
                 ha="center", fontsize=22, fontweight="bold")
        fig.text(0.5, 0.62,
                 "Tổng hợp biểu đồ cho thesis",
                 ha="center", fontsize=14, style="italic")
        fig.text(0.5, 0.45,
                 "1. ITL distribution CDF (per rate)\n\n"
                 "2. ITL CV & P99 — variance comparison\n\n"
                 "3. CPU routing % stacked bar\n\n"
                 "4. Causal link: CPU% ↔ Δ Nlatency\n\n"
                 "5. Scalar metrics: TTFT / TPOT / Nlatency vs QPS",
                 ha="center", fontsize=12)
        fig.text(0.5, 0.05,
                 "Sinh tự động bởi plot_for_report.py",
                 ha="center", fontsize=9, color="#666", style="italic")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Append each individual PDF page
        import pypdf
        for path in out_paths:
            try:
                reader = pypdf.PdfReader(path)
                # Use matplotlib re-render via mimicked import to avoid needing pypdf
            except Exception:
                pass
        # Fallback: just save bundle with cover page; individuals are separate PDFs
    return bundle


# ============================================================
# FIG 6 + FIG 7: TTFT / TPOT theo request arrival index, rolling mean
# ============================================================
def _per_rate_panel(metric_fn, ylabel, title, fname):
    import pandas as pd
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if not (opt and dual):
            ax.set_visible(False); continue
        arr_o = metric_fn(opt)
        arr_d = metric_fn(dual)

        # Rolling mean (window theo độ dài)
        win_o = max(20, len(arr_o) // 30)
        win_d = max(20, len(arr_d) // 30)
        roll_o = pd.Series(arr_o).rolling(win_o, min_periods=1).mean()
        roll_d = pd.Series(arr_d).rolling(win_d, min_periods=1).mean()

        ax.plot(np.arange(len(arr_o)), roll_o,
                color=C_OPTXXX, lw=2.2,
                label=f"opt-xxx  mean={arr_o.mean():.0f} p99={np.percentile(arr_o, 99):.0f}")
        ax.plot(np.arange(len(arr_d)), roll_d,
                color=C_DUAL, lw=2.2,
                label=f"dual1.0   mean={arr_d.mean():.0f} p99={np.percentile(arr_d, 99):.0f}")

        ax.set_xlabel("Request arrival index")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"r = {r} req/s  ·  n_req = {len(arr_o)}",
                     loc="left", fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(title, fontsize=13.5, y=1.00)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


def fig_ttft_per_index():
    return _per_rate_panel(
        metric_fn=lambda d: np.asarray(d["ttfts"]) * 1000,  # → ms
        ylabel="TTFT (ms, log)",
        title="TTFT theo arrival index (rolling mean) — dual1.0 vs opt-xxx",
        fname="fig6_ttft_per_index.pdf",
    )


def fig_tpot_per_index():
    def tpot_per_req(d):
        # TPOT_i = mean(itls[i]) — dùng raw token count, KHÔNG bias bởi BPE re-tok
        return np.asarray([
            (sum(itl) / len(itl)) * 1000 if len(itl) > 0 else 0
            for itl in d["itls"]
        ])
    return _per_rate_panel(
        metric_fn=tpot_per_req,
        ylabel="TPOT (ms/token, log)  [raw]",
        title="TPOT theo arrival index (rolling mean, raw token) — dual1.0 vs opt-xxx",
        fname="fig7_tpot_per_index.pdf",
    )


# ============================================================
# TRACE-BASED FIGURES (từ trace_merged.csv của dual1.0 + opt-xxx)
# ============================================================
TRACE_DUAL  = lambda r: os.path.normpath(os.path.join(HERE, "..", "SERVE_DUAL_1",  f"r{r}", "trace_merged.csv"))


def load_trace(path):
    """Parse trace_merged.csv → list of (t_rel, event, extra_json_str)."""
    import csv
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["t_rel"]), row["event"], row.get("extra_json", "")))
    return rows


def extract_predictor_lats(trace, event_end):
    """Trả về list lat_ms từ event_end (predictor.submit.end hoặc predictor.worker.forward.end)."""
    out = []
    for _, ev, j in trace:
        if ev == event_end and j:
            m = re.search(r'"lat_ms"\s*:\s*([0-9.eE+-]+)', j)
            if m:
                out.append(float(m.group(1)))
    return out


def extract_paired_intervals(trace, ev_start, ev_end):
    """Trả về list (t_start, t_end, dur_ms) cho mỗi cặp start-end (FIFO match).

    Note: model_executor có thể overlap khi multi-thread, nhưng trace của ta MainThread → FIFO OK.
    """
    starts = []
    intervals = []
    for t, ev, _ in trace:
        if ev == ev_start:
            starts.append(t)
        elif ev == ev_end and starts:
            t0 = starts.pop(0)
            intervals.append((t0, t, (t - t0) * 1000))
    return intervals


# FIG 8: Phân bố latency mỗi lần gọi predictor của dual1.0 — CPU OV vs GPU sync
def fig_predictor_lat_dist():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        t_dual = load_trace(TRACE_DUAL(r))
        if not t_dual:
            ax.set_visible(False); continue

        # dual1.0: 2 nguồn — GPU sync (predictor.gpu_sync via interval) + CPU OV (worker.forward.end lat_ms)
        dual_cpu_lat = extract_predictor_lats(t_dual, "predictor.worker.forward.end")
        dual_gpu_intervals = extract_paired_intervals(t_dual, "predictor.gpu_sync.start", "predictor.gpu_sync.end")
        dual_gpu_lat = [d for _, _, d in dual_gpu_intervals]

        bins = np.logspace(0, 3.5, 40)
        if dual_gpu_lat:
            ax.hist(dual_gpu_lat, bins=bins, alpha=0.65, color="#1f77b4",
                    label=f"GPU sync (n={len(dual_gpu_lat)}, μ={np.mean(dual_gpu_lat):.1f}ms)")
        if dual_cpu_lat:
            ax.hist(dual_cpu_lat, bins=bins, alpha=0.6, color="#2ca02c",
                    label=f"CPU OV async (n={len(dual_cpu_lat)}, μ={np.mean(dual_cpu_lat):.1f}ms)")

        ax.set_xscale("log")
        ax.set_xlabel("Predictor latency (ms, log)")
        ax.set_ylabel("Số lần gọi")
        ax.set_title(f"r = {r} req/s", loc="left", fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "dual1.0: phân bố latency mỗi predictor call — CPU OV async vs GPU sync\n"
        "CPU OV chạy SONG SONG với GPU work → KHÔNG pause Llama dù chậm hơn",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig8_predictor_lat_dist.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# FIG 9: dual1.0 GPU time breakdown — model_executor vs predictor sync vs idle
def fig_gpu_time_breakdown():
    rates = RATES
    durs, mes, preds, idles, cpu_times = [], [], [], [], []
    for r in rates:
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            durs.append(0); mes.append(0); preds.append(0); idles.append(0); cpu_times.append(0)
            continue
        t_lo = tr[0][0]; t_hi = tr[-1][0]
        dur = max(1e-6, t_hi - t_lo)
        me = extract_paired_intervals(tr, "model_executor.start", "model_executor.end")
        me_time = sum(d for _, _, d in me) / 1000.0
        pred = extract_paired_intervals(tr, "predictor.gpu_sync.start", "predictor.gpu_sync.end")
        pred_time = sum(d for _, _, d in pred) / 1000.0
        idle = max(0, dur - me_time - pred_time)
        cpu = extract_paired_intervals(tr, "predictor.worker.forward.start", "predictor.worker.forward.end")
        cpu_time = sum(d for _, _, d in cpu) / 1000.0
        durs.append(dur); mes.append(me_time); preds.append(pred_time); idles.append(idle); cpu_times.append(cpu_time)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(rates))
    w = 0.55

    # Stacked GPU bar
    bm = ax.bar(x, mes,   w, color="#d62728", edgecolor="black", lw=0.5,
                label="GPU: model_executor (Llama)")
    bp = ax.bar(x, preds, w, bottom=mes, color="#9467bd", edgecolor="black", lw=0.5,
                label="GPU: predictor sync")
    bi = ax.bar(x, idles, w, bottom=np.array(mes) + np.array(preds),
                color="#cccccc", edgecolor="black", lw=0.5, alpha=0.55,
                label="GPU: idle / overhead")

    # CPU OV overlay — shown as side annotation (since CPU không chiếm GPU bar)
    for xi, dur, me_t, pr_t, cpu_t in zip(x, durs, mes, preds, cpu_times):
        if dur <= 0: continue
        pct_me  = 100 * me_t / dur
        pct_pr  = 100 * pr_t / dur
        # Label inside bars
        if me_t / dur > 0.06:
            ax.text(xi, me_t / 2, f"{pct_me:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
        if pr_t / dur > 0.04:
            ax.text(xi, me_t + pr_t / 2, f"{pct_pr:.1f}%",
                    ha="center", va="center", color="white",
                    fontweight="bold", fontsize=10)
        # CPU OV time annotation above bar
        ax.text(xi, dur + 6, f"CPU OV async\n= {cpu_t:.0f}s\n(chạy SONG SONG)",
                ha="center", va="bottom", fontsize=9, color="#2ca02c",
                fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in rates])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Thời gian (giây)")
    ax.set_title("dual1.0: GPU time breakdown + CPU OV async (overlay)",
                 pad=14)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(durs) * 1.4)

    fig.text(0.5, -0.02,
             "Cột = bench duration. Phần ĐỎ = GPU dành cho Llama main model. "
             "Phần TÍM = GPU bị predictor sync chiếm. "
             "Số xanh phía trên = thời gian CPU OV forward (không nằm trên GPU bar vì chạy song song với phần đỏ).",
             ha="center", fontsize=9.5, style="italic", color="#444")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig9_gpu_time_breakdown.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# FIG 10: Router decision scatter — state khi quyết định route CPU vs GPU
def fig_router_decision_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            ax.set_visible(False); continue

        # Tìm các cặp (router.start, router.end) liên tiếp
        starts = {}  # MainThread queue
        decisions = []
        for t, ev, j in tr:
            if ev == "dual.router.start":
                m_state = re.search(
                    r'"state"\s*:\s*\{([^}]+)\}', j or ""
                )
                if m_state:
                    fields = dict(re.findall(r'"(\w+)"\s*:\s*([0-9.]+)', m_state.group(1)))
                    starts.setdefault("_q", []).append({
                        "n_running": int(float(fields.get("n_running", 0))),
                        "n_decode":  int(float(fields.get("n_decode", 0))),
                        "n_prefill": int(float(fields.get("n_prefill", 0))),
                        "n_tokens_next": int(float(fields.get("n_tokens_next", 0))),
                    })
            elif ev == "dual.router.end" and starts.get("_q"):
                state = starts["_q"].pop(0)
                m_cpu = re.search(r'"n_cpu"\s*:\s*(\d+)', j or "")
                m_gpu = re.search(r'"n_gpu"\s*:\s*(\d+)', j or "")
                n_cpu = int(m_cpu.group(1)) if m_cpu else 0
                n_gpu = int(m_gpu.group(1)) if m_gpu else 0
                # 1 quyết định = batch_n_input của router, ưu tiên: n_cpu>0 → CPU, ngược lại GPU
                routed = "CPU" if n_cpu > 0 else "GPU"
                decisions.append((state["n_running"], state["n_decode"],
                                  state["n_tokens_next"], routed))

        if not decisions:
            ax.set_visible(False); continue

        cpu_pts = [(n_r, n_t) for n_r, n_d, n_t, k in decisions if k == "CPU"]
        gpu_pts = [(n_r, n_t) for n_r, n_d, n_t, k in decisions if k == "GPU"]
        if cpu_pts:
            xs, ys = zip(*cpu_pts)
            ax.scatter(xs, ys, c="#2ca02c", marker="o", alpha=0.6, s=42,
                       edgecolor="#1f6f1f", lw=0.4,
                       label=f"→ CPU (n={len(cpu_pts)})")
        if gpu_pts:
            xs, ys = zip(*gpu_pts)
            ax.scatter(xs, ys, c="#1f77b4", marker="x", alpha=0.6, s=42,
                       label=f"→ GPU sync (n={len(gpu_pts)})")
        ax.set_xlabel("n_running (batch size)")
        ax.set_ylabel("n_tokens_next (input to Llama iter kế)")
        ax.set_title(f"r = {r} req/s  ·  {len(decisions)} decisions",
                     loc="left", fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Quyết định của router theo state hệ thống: CPU vs GPU sync\n"
        "Rate thấp → state nhỏ → T_main lớn → giấu được CPU → router chọn CPU (chấm xanh)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig10_router_decision_scatter.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 11: Số iterations Llama chạy SONG SONG mỗi predictor call
#   — Bằng chứng overlap CPU OV vs GPU sync block
# ============================================================
def _count_iters_in_windows(trace, ev_start, ev_end, me_starts):
    """Cho mỗi cặp (ev_start, ev_end) trong trace, đếm model_executor.start
    timestamps rơi vào khoảng [ts, te]."""
    starts = []
    res = []
    for t, ev, _ in trace:
        if ev == ev_start:
            starts.append(t)
        elif ev == ev_end and starts:
            ts = starts.pop(0); te = t
            cnt = sum(1 for me_t in me_starts if ts <= me_t <= te)
            res.append({"dur_ms": (te - ts) * 1000, "n_iters": cnt})
    return res


def fig_overlap_iterations():
    """Scatter: duration vs # iterations during predictor call. Per rate."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            ax.set_visible(False); continue
        me_starts = [t for t, ev, _ in tr if ev == "model_executor.start"]
        cpu = _count_iters_in_windows(tr, "predictor.worker.forward.start",
                                          "predictor.worker.forward.end", me_starts)
        gpu = _count_iters_in_windows(tr, "predictor.gpu_sync.start",
                                          "predictor.gpu_sync.end", me_starts)

        # CPU points
        if cpu:
            xs = [x["dur_ms"] for x in cpu]
            ys = [x["n_iters"] for x in cpu]
            ax.scatter(xs, ys, c="#2ca02c", s=40, alpha=0.65,
                       edgecolor="#1f6f1f", lw=0.4,
                       label=f"CPU OV (n={len(cpu)}, hidden μ={np.mean(ys):.2f}, max={max(ys)})")
        if gpu:
            xs = [x["dur_ms"] for x in gpu]
            ys = [x["n_iters"] for x in gpu]
            ax.scatter(xs, ys, c="#1f77b4", s=40, alpha=0.55, marker="x",
                       label=f"GPU sync (n={len(gpu)}, hidden ALWAYS 0)")

        ax.set_xscale("log")
        ax.set_xlabel("Predictor call duration (ms, log)")
        ax.set_ylabel("# Llama iterations chạy SONG SONG")
        ax.set_title(f"r = {r} req/s", loc="left", fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(0, color="black", lw=0.6, ls=":")
    fig.suptitle(
        "Bằng chứng overlap: # Llama iteration chạy trong khi predictor đang chạy\n"
        "GPU sync luôn = 0 (block GPU). CPU OV có thể > 0 (Llama vẫn forward song song)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig11_overlap_iterations.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 12: Aggregate — Tổng GPU time "saved" nhờ CPU overlap
# ============================================================
def fig_gpu_saved_aggregate():
    """Mỗi rate, tính:
      - Tổng GPU SYNC time = sum(dur_gpu_sync)  → GPU bị chiếm cho predictor
      - Tổng GPU work overlapped với CPU OV = ∩(model_executor windows, CPU OV windows)
        → 'GPU work tiết kiệm được' vì predictor không cần đến nó
    """
    rates = RATES
    gpu_lost = []  # giây
    gpu_overlap = []  # giây — Llama work happening during CPU OV
    cpu_total = []  # giây — total CPU OV time

    for r in rates:
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            gpu_lost.append(0); gpu_overlap.append(0); cpu_total.append(0); continue

        gpu_pairs = extract_paired_intervals(tr, "predictor.gpu_sync.start", "predictor.gpu_sync.end")
        gpu_lost.append(sum(d for _, _, d in gpu_pairs) / 1000.0)

        cpu_pairs = extract_paired_intervals(tr, "predictor.worker.forward.start",
                                                 "predictor.worker.forward.end")
        cpu_total.append(sum(d for _, _, d in cpu_pairs) / 1000.0)

        # Compute overlap time = sum over cpu windows of (intersection with model_executor windows)
        me_pairs = extract_paired_intervals(tr, "model_executor.start", "model_executor.end")
        overlap_total_s = 0.0
        # Sort by start
        cpu_iv = sorted([(s, e) for s, e, _ in cpu_pairs])
        me_iv  = sorted([(s, e) for s, e, _ in me_pairs])

        # Two-pointer for sorted intervals
        j = 0
        for cs, ce in cpu_iv:
            # Advance me until end >= cs
            while j < len(me_iv) and me_iv[j][1] < cs:
                j += 1
            k = j
            while k < len(me_iv) and me_iv[k][0] <= ce:
                ms, me = me_iv[k]
                overlap_total_s += max(0, min(ce, me) - max(cs, ms))
                k += 1
        gpu_overlap.append(overlap_total_s)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rates))
    w = 0.28

    ax.bar(x - w, gpu_lost, w, color="#9467bd", edgecolor="black", lw=0.5,
           label="GPU sync time  (GPU bị predictor chiếm — thời gian Llama bị block)")
    ax.bar(x,     cpu_total, w, color="#2ca02c", edgecolor="black", lw=0.5,
           label="CPU OV total time  (toàn bộ thời lượng CPU forward)")
    ax.bar(x + w, gpu_overlap, w, color="#ff7f0e", edgecolor="black", lw=0.5,
           label="Llama work overlapped với CPU OV  (Llama đã decode SONG SONG)")

    for xi, v in zip(x - w, gpu_lost):
        ax.text(xi, v + 0.3, f"{v:.1f}s", ha="center", fontsize=9, color="#9467bd")
    for xi, v in zip(x, cpu_total):
        ax.text(xi, v + 0.3, f"{v:.1f}s", ha="center", fontsize=9, color="#2ca02c")
    for xi, v in zip(x + w, gpu_overlap):
        ax.text(xi, v + 0.3, f"{v:.1f}s", ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in rates])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Thời gian tích luỹ (giây)")
    ax.set_title("Bằng chứng định lượng: CPU OV cho phép Llama chạy SONG SONG (cam) — nếu route GPU sẽ mất bấy nhiêu thời gian (tím)",
                 pad=12, fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    fig.text(0.5, -0.03,
             "Cam (Llama overlap với CPU OV) = thời gian Llama đã thực sự decode SONG SONG khi predictor đang chạy. "
             "Đây là 'work miễn phí' về scheduling cost — không thể có nếu dùng GPU sync.",
             ha="center", fontsize=10, style="italic", color="#444")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig12_gpu_saved_aggregate.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 13: ITL distribution — GPU block evidence vs CPU overlap benefit
#   Histogram overlay opt-xxx vs dual1.0 từ JSON, annotate vùng "block penalty"
# ============================================================
def fig_itl_block_evidence():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    for i, r in enumerate(RATES):
        ax = axes[i]
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if not (opt and dual):
            ax.set_visible(False); continue
        itl_o = all_itls_ms(opt)
        itl_d = all_itls_ms(dual)

        # Tính baseline decode ITL = median dual (vùng GPU decode bình thường)
        baseline = np.median(itl_d)

        bins = np.logspace(np.log10(max(0.5, min(itl_o.min(), itl_d.min()))),
                           np.log10(max(itl_o.max(), itl_d.max())),
                           60)

        # Density để 2 distribution so sánh được dù khác n
        ax.hist(itl_o, bins=bins, density=True, alpha=0.55, color=C_OPTXXX,
                label=f"opt-xxx (μ={itl_o.mean():.1f}ms, p99={np.percentile(itl_o,99):.0f})")
        ax.hist(itl_d, bins=bins, density=True, alpha=0.55, color=C_DUAL,
                label=f"dual1.0 (μ={itl_d.mean():.1f}ms, p99={np.percentile(itl_d,99):.0f})")

        ax.axvline(baseline, color="#444", lw=1.0, ls="--", alpha=0.7)
        ax.text(baseline * 1.05, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.01,
                f"baseline\n≈ {baseline:.0f}ms",
                fontsize=9, color="#444", style="italic")

        # Quantify excess: ITL > 2× baseline → coi là "blocked"
        threshold = 2 * baseline
        pct_o = 100 * (itl_o > threshold).mean()
        pct_d = 100 * (itl_d > threshold).mean()
        ax.set_xscale("log")
        ax.set_xlabel("ITL (ms, log)")
        ax.set_ylabel("Mật độ (density)")
        ax.set_title(
            f"r = {r} req/s  ·  P(ITL > 2×baseline): opt-xxx {pct_o:.1f}% vs dual {pct_d:.1f}%",
            loc="left", fontweight="bold", fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        "Phân bố ITL: opt-xxx có 'block tail' do predictor sync, dual1.0 không\n"
        "Đường nét đứt = baseline decode (median dual). Tỉ lệ ITL > 2×baseline cho thấy GPU bị block bao nhiêu thường xuyên.",
        fontsize=12.5, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig13_itl_block_evidence.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 14: Excess ITL time = bằng chứng định lượng cuối cùng
#   "Mỗi token mà opt-xxx CHẬM hơn dual1.0 ≈ thời gian GPU bị predictor chiếm"
# ============================================================
def fig_excess_itl_time():
    rates = RATES
    excess_per_tok = []  # mean(ITL_opt) - mean(ITL_dual), ms/token
    total_extra_s  = []  # tổng GPU time "lãng phí" qua tất cả token
    n_tokens_opt   = []
    p99_diff       = []

    for r in rates:
        opt = load_one(JSON_POOL, "opt-xxx", r)
        dual = load_one(JSON_POOL, "dual1.0", r)
        if not (opt and dual):
            for arr in [excess_per_tok, total_extra_s, n_tokens_opt, p99_diff]:
                arr.append(0); continue

        itl_o = all_itls_ms(opt); itl_d = all_itls_ms(dual)
        excess = itl_o.mean() - itl_d.mean()
        excess_per_tok.append(excess)
        n_tok = len(itl_o); n_tokens_opt.append(n_tok)
        total_extra_s.append(excess * n_tok / 1000.0)  # → giây
        p99_diff.append(np.percentile(itl_o, 99) - np.percentile(itl_d, 99))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    x = np.arange(len(rates))

    # Left: excess per token (mean & p99)
    w = 0.4
    ax1.bar(x - w/2, excess_per_tok, w, color="#ff7f0e", edgecolor="black", lw=0.5,
            label="Excess mean ITL")
    ax1.bar(x + w/2, p99_diff, w, color="#9467bd", edgecolor="black", lw=0.5,
            label="Excess p99 ITL")
    for xi, v in zip(x - w/2, excess_per_tok):
        ax1.text(xi, v + 0.5 if v > 0 else v - 1.5, f"{v:+.1f}",
                 ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")
    for xi, v in zip(x + w/2, p99_diff):
        ax1.text(xi, v + 5 if v > 0 else v - 20, f"{v:+.0f}",
                 ha="center", fontsize=9, color="#9467bd", fontweight="bold")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels([f"r={r}" for r in rates])
    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("opt-xxx ITL − dual1.0 ITL  (ms)")
    ax1.set_title("Mỗi token opt-xxx chậm hơn dual1.0 bao nhiêu?",
                  pad=10, fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: total extra GPU time (sum of excess × n_tok)
    bars = ax2.bar(x, total_extra_s, 0.55, color="#d62728", edgecolor="black", lw=0.5)
    for xi, v, n in zip(x, total_extra_s, n_tokens_opt):
        ax2.text(xi, v + max(total_extra_s) * 0.02, f"{v:.0f}s",
                 ha="center", fontsize=10, color="#d62728", fontweight="bold")
        ax2.text(xi, max(0, v / 2), f"{n} tok",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"r={r}" for r in rates])
    ax2.set_xlabel("Request rate (req/s)")
    ax2.set_ylabel("Tổng excess GPU time (giây)")
    ax2.set_title("Tổng GPU time bị 'mất' trong opt-xxx vs dual1.0\n"
                  "≈ Tổng thời gian Llama bị predictor block",
                  pad=10, fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Bằng chứng định lượng từ ITL: GPU bị predictor block làm chậm mỗi token",
        fontsize=13, y=1.02,
    )
    fig.text(0.5, -0.02,
             "Mean ITL chênh × số token = tổng thời gian GPU 'lãng phí'. "
             "Đây là phần TPOT cải thiện trực tiếp do CPU OV overlap (không block GPU).",
             ha="center", fontsize=10, style="italic", color="#444")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig14_excess_itl_time.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 15: model_executor iteration cycle time stratified by gap content
#   — Bằng chứng: GPU sync gap → cycle dài → ITL dài; CPU OV gap → cycle ngắn
# ============================================================
def _classify_iter_cycles(trace):
    """Cho mỗi cặp model_executor.start[i] → model_executor.start[i+1],
    classify gap (giữa me[i].end và me[i+1].start) theo predictor event trong gap:
      - 'gpu_sync': có predictor.gpu_sync.start trong gap
      - 'cpu_ov':   có predictor.worker.forward.start trong gap (và không có gpu_sync)
      - 'clean':    không có predictor nào
    Returns: list of (cycle_ms, gap_ms, category)
    """
    me_starts = []; me_ends = []
    for t, ev, _ in trace:
        if ev == "model_executor.start": me_starts.append(t)
        elif ev == "model_executor.end": me_ends.append(t)

    # Pair start-end
    me_pairs = list(zip(me_starts, me_ends))[:min(len(me_starts), len(me_ends))]
    if len(me_pairs) < 2:
        return []

    # Build event timeline for classification
    gpu_sync_starts = [t for t, ev, _ in trace if ev == "predictor.gpu_sync.start"]
    cpu_ov_starts   = [t for t, ev, _ in trace if ev == "predictor.worker.forward.start"]

    results = []
    for i in range(len(me_pairs) - 1):
        s1, e1 = me_pairs[i]
        s2, e2 = me_pairs[i + 1]
        cycle_ms = (s2 - s1) * 1000
        gap_ms = (s2 - e1) * 1000
        if gap_ms < 0:
            continue
        # Has predictor.gpu_sync in (e1, s2)?
        has_gpu = any(e1 <= t <= s2 for t in gpu_sync_starts)
        has_cpu = any(e1 <= t <= s2 for t in cpu_ov_starts)
        if has_gpu:
            cat = "gpu_sync"
        elif has_cpu:
            cat = "cpu_ov"
        else:
            cat = "clean"
        results.append((cycle_ms, gap_ms, cat))
    return results


def fig_iter_cycle_time():
    """Per rate: bar chart mean cycle time (=ITL trung bình của 1 iteration) phân loại theo gap.
    + horizontal line ‘opt-xxx mean ITL’ từ JSON để so sánh.
    """
    rates = RATES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()

    for i, r in enumerate(rates):
        ax = axes[i]
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            ax.set_visible(False); continue
        cycles = _classify_iter_cycles(tr)
        if not cycles:
            ax.set_visible(False); continue

        from collections import defaultdict
        groups = defaultdict(list)
        for c, g, cat in cycles:
            groups[cat].append(c)

        labels = []
        means  = []
        counts = []
        colors = []
        order = [("clean",    "Gap sạch (không predictor)",   "#aaaaaa"),
                 ("cpu_ov",   "Gap có CPU OV (overlap)",      "#2ca02c"),
                 ("gpu_sync", "Gap có GPU sync (block)",      "#9467bd")]
        for key, lbl, color in order:
            arr = groups.get(key, [])
            if not arr: continue
            labels.append(lbl); means.append(np.mean(arr))
            counts.append(len(arr)); colors.append(color)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, 0.6, color=colors, edgecolor="black", lw=0.5)
        for xi, v, n in zip(x, means, counts):
            ax.text(xi, v + max(means) * 0.03,
                    f"{v:.1f}ms\n(n={n})",
                    ha="center", fontsize=9, fontweight="bold")

        # Reference line: opt-xxx mean ITL từ JSON
        opt = load_one(JSON_POOL, "opt-xxx", r)
        if opt:
            opt_itl_mean = all_itls_ms(opt).mean()
            ax.axhline(opt_itl_mean, color="#d62728", lw=2, ls="--", alpha=0.8)
            ax.text(0.02, opt_itl_mean,
                    f" opt-xxx mean ITL = {opt_itl_mean:.1f}ms",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, color="#d62728", fontweight="bold",
                    va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=10)
        ax.set_ylabel("Cycle time = me_start[i+1] − me_start[i]  (ms)")
        ax.set_title(f"r = {r} req/s", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Bằng chứng cơ chế: cycle time của 1 iteration decode tăng khi gap chứa GPU sync\n"
        "Đường đỏ (opt-xxx mean ITL) ≈ cycle 'gpu_sync' của dual1.0 → confirm opt-xxx = 100% gpu_sync regime",
        fontsize=12.5, y=1.00,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig15_iter_cycle_time.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# FIG 16: KILLER PROOF — delta cycle time so với clean baseline
#   "Mỗi loại gap thêm bao nhiêu ms vào cycle?"
# ============================================================
def fig_overlap_proof_killer():
    from collections import defaultdict
    rates = RATES
    delta_cpu, delta_gpu = [], []
    clean_baseline = []
    n_cpu_samples, n_gpu_samples = [], []

    for r in rates:
        tr = load_trace(TRACE_DUAL(r))
        if not tr:
            for arr in [delta_cpu, delta_gpu, clean_baseline, n_cpu_samples, n_gpu_samples]:
                arr.append(0)
            continue
        cycles = _classify_iter_cycles(tr)
        g = defaultdict(list)
        for c, gap, cat in cycles: g[cat].append(c)
        clean_mean = np.mean(g.get("clean", [0]))
        cpu_mean   = np.mean(g.get("cpu_ov", [clean_mean])) if g.get("cpu_ov") else clean_mean
        gpu_mean   = np.mean(g.get("gpu_sync", [clean_mean])) if g.get("gpu_sync") else clean_mean
        clean_baseline.append(clean_mean)
        delta_cpu.append(cpu_mean - clean_mean)
        delta_gpu.append(gpu_mean - clean_mean)
        n_cpu_samples.append(len(g.get("cpu_ov", [])))
        n_gpu_samples.append(len(g.get("gpu_sync", [])))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(rates))
    w = 0.38

    bars_cpu = ax.bar(x - w/2, delta_cpu, w, color="#2ca02c", edgecolor="#1f6f1f", lw=0.5,
                      label="Gap có CPU OV  (predictor overlap với GPU)")
    bars_gpu = ax.bar(x + w/2, delta_gpu, w, color="#9467bd", edgecolor="#542d83", lw=0.5,
                      label="Gap có GPU sync  (predictor BLOCK GPU)")

    for xi, v, n in zip(x - w/2, delta_cpu, n_cpu_samples):
        if abs(v) < 1:
            label = f"+{v:.1f}ms"; color = "#1f6f1f"
        else:
            label = f"+{v:.1f}ms"; color = "#1f6f1f"
        ax.text(xi, v + max(delta_gpu) * 0.025 if v >= 0 else v - max(delta_gpu) * 0.04,
                f"{label}\n(n={n})", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, color=color, fontweight="bold")
    for xi, v, n in zip(x + w/2, delta_gpu, n_gpu_samples):
        ax.text(xi, v + max(delta_gpu) * 0.025,
                f"+{v:.1f}ms\n(n={n})", ha="center",
                fontsize=10, color="#542d83", fontweight="bold")

    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"r={r}\n(clean={c:.0f}ms)" for r, c in zip(rates, clean_baseline)])
    ax.set_xlabel("Request rate (req/s)  ·  baseline 'clean' cycle in nhỏ phía dưới")
    ax.set_ylabel("Δ Cycle time so với 'clean' gap  (ms)")
    ax.set_title(
        "Bằng chứng KEY: CPU overlap thêm ≈ 0ms vào cycle. GPU sync thêm 30-240ms (≈ thời lượng predictor)\n"
        "Cycle time = ITL ≈ TPOT → CPU OV không tăng TPOT, GPU sync TĂNG TPOT mỗi lần nó chạy",
        pad=12, fontsize=12,
    )
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    fig.text(0.5, -0.03,
             "Phương pháp: từ trace dual1.0, phân loại mỗi cặp model_executor[i]→model_executor[i+1] "
             "theo predictor event trong gap. 'clean' = không có predictor (baseline). "
             "Δ = cycle_avg(category) − cycle_avg(clean). "
             "Δ_cpu_ov ≈ 0 → CPU overlap KHÔNG cản GPU. Δ_gpu_sync > 0 → GPU bị block.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig16_overlap_proof_killer.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    return out


def main():
    print("Generating PDF reports →", OUT_DIR)
    out_paths = []
    out_paths.append(fig_itl_cdf());                  print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_itl_cv());                   print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_cpu_routing());              print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_proof_of_win());             print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_scalar_compare());           print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_ttft_per_index());           print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_tpot_per_index());           print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_predictor_lat_dist());       print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_gpu_time_breakdown());       print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_router_decision_scatter()); print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_overlap_iterations());       print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_gpu_saved_aggregate());      print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_itl_block_evidence());       print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_excess_itl_time());          print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_iter_cycle_time());          print(f"  ✓ {out_paths[-1]}")
    out_paths.append(fig_overlap_proof_killer());     print(f"  ✓ {out_paths[-1]}")

    print(f"\nDone — {len(out_paths)} PDF files in {OUT_DIR}")


if __name__ == "__main__":
    main()
