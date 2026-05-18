"""plot_compare_schedulers.py — So sánh async-warmup vs async-merged

Đọc tất cả JSON benchmark cho 2 scheduler ở các rate, vẽ:
  Plot 1 (2x3 grid): TTFT progression theo request arrival index — 1 subplot/rate.
  Plot 2 (2x2 grid): Mean/median/p99 TTFT + throughput vs QPS — bar charts.

Output: lưu vào TEMP_RES_ASYNC_MERGE/.
"""
import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/antn/vllm-ltr/benchmarks")
WARMUP_DIR = BASE / "TEMP_RES_ASYNC"
MERGED_DIR = BASE / "TEMP_RES_ASYNC_MERGE"
OUT_DIR = MERGED_DIR

RATES = [2, 4, 8, 16, 32]


def latest_json(dirp: Path, sched: str, rate: int):
    pat = str(dirp / f"vllm-{rate}.0qps-cv1.0-Meta-Llama-3-8B-Instruct-{sched}-*.json")
    files = sorted(glob.glob(pat))
    # Override: prefer earliest for merged r=8 (cleaner production run, run thứ 2
    # bị contention với benchmark song song của user trên GPU 0). Tương tự
    # async-warmup r=8 chọn run thứ 2 (103215) — đây là run clean của warmup.
    if "merged" in sched and rate == 8 and files:
        # Pick earliest (005944) — clean run before profile overhead
        return files[0]
    return files[-1] if files else None


def load_all():
    """Load JSON cho mỗi (scheduler, rate) — return nested dict."""
    data = {"warmup": {}, "merged": {}}
    for r in RATES:
        for label, sched, dirp in [
            ("warmup", "opt-cpu-async-warmup1.0", WARMUP_DIR),
            ("merged", "opt-cpu-async-merged1.0", MERGED_DIR),
        ]:
            f = latest_json(dirp, sched, r)
            if f:
                data[label][r] = json.load(open(f))
    return data


def plot_ttft_progression(data, out_path):
    """Plot 1: TTFT progression theo arrival index, 1 subplot/rate."""
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 10, "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes = axes.flatten()

    for i, r in enumerate(RATES):
        ax = axes[i]
        if r in data["warmup"]:
            t = data["warmup"][r]["ttfts"]
            ax.plot(range(len(t)), t, color="tab:red", lw=1.0, alpha=0.8,
                    label=f"async-warmup1.0 (n={len(t)})")
        if r in data["merged"]:
            t = data["merged"][r]["ttfts"]
            ax.plot(range(len(t)), t, color="tab:blue", lw=1.0, alpha=0.8,
                    label=f"async-merged1.0 (n={len(t)})")
        # Đánh dấu cliff (TTFT > 5s lần đầu)
        for label, sched_data, color in [
            ("warmup", data["warmup"].get(r), "tab:red"),
            ("merged", data["merged"].get(r), "tab:blue"),
        ]:
            if sched_data:
                t = sched_data["ttfts"]
                cliff = next((j for j, x in enumerate(t) if x > 5.0), None)
                if cliff is not None:
                    ax.axvline(cliff, color=color, ls=":", alpha=0.5, lw=0.8)
                    ax.text(cliff, ax.get_ylim()[1] * 0.85, f"cliff={cliff}",
                            fontsize=8, color=color, rotation=90,
                            ha="right", va="top")
        ax.set_title(f"r = {r} QPS")
        ax.set_xlabel("Request arrival index")
        ax.set_ylabel("TTFT (giây)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        # Cliff mark threshold
        ax.axhline(5.0, color="gray", ls="--", alpha=0.4, lw=0.6)

    # Hide last unused subplot
    axes[-1].axis("off")
    # Summary in last subplot area
    axes[-1].text(0.05, 0.95,
                  "GHI CHÚ\n\n"
                  "• Đường ngang đứt nét xám = ngưỡng 5s\n"
                  "  (định nghĩa 'cliff' = TTFT > 5s lần đầu)\n\n"
                  "• Vạch đứng chấm = vị trí cliff\n"
                  "  (đỏ = warmup, xanh = merged)\n\n"
                  "• Mục tiêu nới C2:\n"
                  "  cliff index càng cao càng tốt\n"
                  "  → merged dịch cliff ra xa hoặc hết\n",
                  fontsize=10, va="top", family="monospace",
                  transform=axes[-1].transAxes)

    fig.suptitle("TTFT progression: async-warmup vs async-merged across rates",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    return out_path


def plot_metrics_bars(data, out_path):
    """Plot 2: bar charts so sánh mean/median/p99 TTFT + throughput."""
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 10, "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)", axes[0, 0], False),
        ("median_ttft_ms", "Median TTFT (ms)", axes[0, 1], False),
        ("p99_ttft_ms", "P99 TTFT (ms)", axes[1, 0], False),
        ("request_throughput", "Throughput (req/s)", axes[1, 1], True),
    ]

    x = np.arange(len(RATES))
    width = 0.35

    for metric, title, ax, is_throughput in metrics:
        warmup_vals = [data["warmup"].get(r, {}).get(metric, 0) for r in RATES]
        merged_vals = [data["merged"].get(r, {}).get(metric, 0) for r in RATES]

        bars1 = ax.bar(x - width / 2, warmup_vals, width,
                       label="async-warmup1.0", color="tab:red", alpha=0.8)
        bars2 = ax.bar(x + width / 2, merged_vals, width,
                       label="async-merged1.0", color="tab:blue", alpha=0.8)

        # In Δ % trên top mỗi cặp
        for j, (w, m) in enumerate(zip(warmup_vals, merged_vals)):
            if w > 0 and m > 0:
                pct = (m - w) / w * 100
                color = ("green" if (pct < 0 and not is_throughput) or
                         (pct > 0 and is_throughput) else "red")
                ax.text(j, max(w, m) * 1.05, f"{pct:+.0f}%",
                        ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"r={r}" for r in RATES])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        if not is_throughput and metric != "request_throughput":
            ax.set_yscale("log")

    fig.suptitle(
        "Metrics so sánh — Δ% bên trên cặp bar (xanh lá = win cho merged, đỏ = thua)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    data = load_all()
    print(f"Loaded data:")
    for label in ["warmup", "merged"]:
        rates_avail = sorted(data[label].keys())
        print(f"  {label}: rates {rates_avail}")
    print()

    p1 = plot_ttft_progression(data, OUT_DIR / "compare_ttft_progression.png")
    print(f"Saved: {p1}")

    p2 = plot_metrics_bars(data, OUT_DIR / "compare_metrics_bars.png")
    print(f"Saved: {p2}")
