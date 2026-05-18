"""plot_predictor_sweep.py — vẽ latency vs batch_size cho OV CPU và GPU.

Đọc CSV do bench_predictor_sweep.sh sinh ra:
  BENCH_PRED_RES/sweep_bs/sweep_summary.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    data = defaultdict(dict)  # data[backend][bs] = latency_ms
    tau = defaultdict(dict)   # tau[backend][bs] = kendall_tau
    with open(path) as f:
        for row in csv.DictReader(f):
            bs = int(row["batch_size"])
            data[row["backend"]][bs] = float(row["inference_time_ms"])
            tau[row["backend"]][bs] = float(row["kendall_tau"])
    return data, tau


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="BENCH_PRED_RES/sweep_bs/sweep_summary.csv")
    p.add_argument("--output", default="BENCH_PRED_RES/sweep_bs/latency_vs_bs.png")
    p.add_argument("--tau-output", default="BENCH_PRED_RES/sweep_bs/kendall_tau.png")
    p.add_argument("--tau-shift-output",
                   default="BENCH_PRED_RES/sweep_bs/kendall_tau_shift.png")
    p.add_argument("--drift-output",
                   default="BENCH_PRED_RES/sweep_bs/kendall_tau_drift_pct.png")
    p.add_argument("--latency-bar-output",
                   default="BENCH_PRED_RES/sweep_bs/latency_bar.png")
    p.add_argument("--tau-baseline", default="gpu",
                   help="backend dùng làm gốc cho biểu đồ shift (default: gpu = OPT-125M GPU)")
    p.add_argument("--ylog", action="store_true", help="dùng log scale cho trục Y")
    args = p.parse_args()

    data, tau = load_csv(Path(args.csv))

    # Tên backend trong CSV (do bench_predictor_sweep.sh sinh):
    #   ov_opt125m, ov_pythia70m, ov_pythia14m, gpu (= OPT-125M GPU)
    styles = {
        "ov_opt125m":   {"label": "OPT-125M CPU",    "color": "#1f77b4", "marker": "o"},
        "gpu":          {"label": "OPT-125M GPU",    "color": "#d62728", "marker": "s"},
        "ov_pythia70m": {"label": "Pythia-70M CPU",  "color": "#2ca02c", "marker": "^"},
        "ov_pythia14m": {"label": "Pythia-14M CPU",  "color": "#ff7f0e", "marker": "D"},
    }
    fallback_palette = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    fallback_markers = ["v", "<", ">", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_order = [b for b in styles if b in data] + [b for b in data if b not in styles]
    fb_i = 0
    for backend in plot_order:
        if backend in styles:
            style = styles[backend]
        else:
            style = {
                "label":  backend,
                "color":  fallback_palette[fb_i % len(fallback_palette)],
                "marker": fallback_markers[fb_i % len(fallback_markers)],
            }
            fb_i += 1
        bs_sorted = sorted(data[backend])
        lat = [data[backend][b] for b in bs_sorted]
        ax.plot(bs_sorted, lat, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    if args.ylog:
        ax.set_yscale("log")
    ax.set_xticks(sorted({b for bs in data.values() for b in bs}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Predictor latency vs batch size — CPU vs GPU")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] {out}")
    print(f"[plot] {out.with_suffix('.pdf')}")

    # ----- Grouped bar chart: latency per (batch_size, model) -----
    all_bs = sorted({b for bs in data.values() for b in bs})
    bar_models = [b for b in styles if b in data] + [b for b in data if b not in styles]
    n_models = len(bar_models)
    if all_bs and n_models:
        import numpy as np
        x = np.arange(len(all_bs))
        width = 0.8 / n_models
        fig_b, ax_b = plt.subplots(figsize=(max(8, 1.4 * len(all_bs)), 5))
        fb_i = 0
        for i, backend in enumerate(bar_models):
            if backend in styles:
                s = styles[backend]
            else:
                s = {"label": backend,
                     "color": fallback_palette[fb_i % len(fallback_palette)]}
                fb_i += 1
            vals = [data[backend].get(b, 0.0) for b in all_bs]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax_b.bar(x + offset, vals, width=width,
                            color=s["color"], edgecolor="black", linewidth=0.4,
                            label=s["label"])
            for bar, v in zip(bars, vals):
                if v <= 0:
                    continue
                ax_b.annotate(f"{v:.0f}",
                              (bar.get_x() + bar.get_width() / 2, v),
                              textcoords="offset points",
                              xytext=(0, 2), ha="center", fontsize=7, rotation=90)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([str(b) for b in all_bs])
        ax_b.set_xlabel("Batch size")
        ax_b.set_ylabel("Inference latency (ms)")
        ax_b.set_title("Predictor latency by batch size — grouped bars")
        ax_b.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_b.legend(loc="upper left", framealpha=0.9)

        fig_b.tight_layout()
        out_b = Path(args.latency_bar_output)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        fig_b.savefig(out_b, dpi=150)
        fig_b.savefig(out_b.with_suffix(".pdf"))
        print(f"[plot] {out_b}")
        print(f"[plot] {out_b.with_suffix('.pdf')}")

    # ----- Bar chart: Kendall tau per predictor -----
    # Tau gần như invariant theo batch_size → lấy trung bình mỗi backend.
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    bar_order = [b for b in styles if b in tau] + [b for b in tau if b not in styles]
    labels, values, colors = [], [], []
    fb_i = 0
    for backend in bar_order:
        if backend in styles:
            s = styles[backend]
        else:
            s = {"label": backend,
                 "color": fallback_palette[fb_i % len(fallback_palette)]}
            fb_i += 1
        vals = list(tau[backend].values())
        labels.append(s["label"])
        values.append(sum(vals) / len(vals))
        colors.append(s["color"])

    bars = ax2.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax2.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v),
                     textcoords="offset points",
                     xytext=(0, 4 if v >= 0 else -12),
                     ha="center", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Kendall's τ (predicted vs actual)")
    ax2.set_title("Predictor ranking quality — Kendall τ")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    fig2.tight_layout()
    out2 = Path(args.tau_output)
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out2, dpi=150)
    fig2.savefig(out2.with_suffix(".pdf"))
    print(f"[plot] {out2}")
    print(f"[plot] {out2.with_suffix('.pdf')}")

    # ----- Bar chart: ranking-quality shift (|τ|) relative to baseline -----
    # Dùng |τ| (correlation magnitude = ranking quality) để bar âm = degradation,
    # bar dương = improvement — trực quan hơn raw Δτ vốn dễ gây nhầm dấu.
    if args.tau_baseline not in tau:
        print(f"[warn] baseline backend '{args.tau_baseline}' không có trong CSV — bỏ qua biểu đồ shift")
    else:
        def mean_abs_tau(b):
            vals = [abs(v) for v in tau[b].values()]
            return sum(vals) / len(vals)

        baseline_abs = mean_abs_tau(args.tau_baseline)
        baseline_label = styles.get(args.tau_baseline, {}).get("label", args.tau_baseline)

        fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
        shift_labels, shift_values, shift_pct, shift_colors = [], [], [], []
        fb_i = 0
        for backend in bar_order:
            if backend == args.tau_baseline:
                continue
            if backend in styles:
                s = styles[backend]
            else:
                s = {"label": backend,
                     "color": fallback_palette[fb_i % len(fallback_palette)]}
                fb_i += 1
            d = mean_abs_tau(backend) - baseline_abs
            shift_labels.append(s["label"])
            shift_values.append(d)
            shift_pct.append(100.0 * d / baseline_abs if baseline_abs else 0.0)
            # đỏ = degradation (Δ<0), xanh lá = improvement (Δ>0), xám = ~0
            shift_colors.append("#d62728" if d < -1e-6 else ("#2ca02c" if d > 1e-6 else "#7f7f7f"))

        bars3 = ax3.bar(shift_labels, shift_values, color=shift_colors,
                        edgecolor="black", linewidth=0.6)
        ymax_abs = max((abs(v) for v in shift_values), default=0.0) or 1.0
        for bar, v, pct in zip(bars3, shift_values, shift_pct):
            inside = v < 0
            ax3.annotate(f"{v:+.4f}\n({pct:+.1f}%)",
                         (bar.get_x() + bar.get_width() / 2, v),
                         textcoords="offset points",
                         xytext=(0, -22 if inside else 6),
                         ha="center", fontsize=9,
                         color="white" if inside else "black",
                         fontweight="bold" if inside else "normal")
        ax3.axhline(0, color="black", linewidth=1.0)
        ax3.text(0.99, 0.97,
                 f"baseline {baseline_label}: |τ| = {baseline_abs:.4f}",
                 transform=ax3.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="gray"))
        ax3.set_ylim(-ymax_abs * 1.35, ymax_abs * 1.35)
        ax3.set_ylabel(f"Δ|τ|  (↑ better ranking,  ↓ worse)")
        ax3.set_title(f"Ranking-quality shift vs {baseline_label}")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.setp(ax3.get_xticklabels(), rotation=10, ha="right")

        fig3.tight_layout()
        out3 = Path(args.tau_shift_output)
        out3.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(out3, dpi=150)
        fig3.savefig(out3.with_suffix(".pdf"))
        print(f"[plot] {out3}")
        print(f"[plot] {out3.with_suffix('.pdf')}")

        # ----- Bar chart: drift % của các predictor CPU vs baseline -----
        cpu_backends = ["ov_opt125m", "ov_pythia70m", "ov_pythia14m"]
        cpu_backends = [b for b in cpu_backends if b in tau]
        if cpu_backends:
            fig4, ax4 = plt.subplots(figsize=(7.5, 4.8))
            d_labels, d_pct, d_colors, d_abs = [], [], [], []
            for backend in cpu_backends:
                s = styles.get(backend, {"label": backend, "color": "#7f7f7f"})
                abs_t = mean_abs_tau(backend)
                pct = 100.0 * (abs_t - baseline_abs) / baseline_abs if baseline_abs else 0.0
                d_labels.append(s["label"])
                d_pct.append(pct)
                d_abs.append(abs_t)
                d_colors.append("#d62728" if pct < -0.5 else ("#2ca02c" if pct > 0.5 else "#7f7f7f"))

            bars4 = ax4.bar(d_labels, d_pct, color=d_colors,
                            edgecolor="black", linewidth=0.6)
            ymax = max((abs(p) for p in d_pct), default=0.0) or 1.0
            for bar, pct, abs_t in zip(bars4, d_pct, d_abs):
                inside = pct < 0
                ax4.annotate(f"{pct:+.2f}%\n|τ|={abs_t:.4f}",
                             (bar.get_x() + bar.get_width() / 2, pct),
                             textcoords="offset points",
                             xytext=(0, -26 if inside else 6),
                             ha="center", fontsize=9,
                             color="white" if inside else "black",
                             fontweight="bold" if inside else "normal")
            ax4.axhline(0, color="black", linewidth=1.0)
            ax4.text(0.99, 0.97,
                     f"baseline {baseline_label}: |τ| = {baseline_abs:.4f}",
                     transform=ax4.transAxes, ha="right", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="gray"))
            ax4.set_ylim(-ymax * 1.4, ymax * 1.4)
            ax4.set_ylabel("Ranking-quality drift  (% so với baseline)")
            ax4.set_title(f"Drift % của predictor CPU vs {baseline_label}")
            ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
            plt.setp(ax4.get_xticklabels(), rotation=10, ha="right")

            fig4.tight_layout()
            out4 = Path(args.drift_output)
            out4.parent.mkdir(parents=True, exist_ok=True)
            fig4.savefig(out4, dpi=150)
            fig4.savefig(out4.with_suffix(".pdf"))
            print(f"[plot] {out4}")
            print(f"[plot] {out4.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
