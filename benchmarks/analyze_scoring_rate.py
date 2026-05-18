"""
Phân tích tỷ lệ requests được score bởi predictor — cho cả opt-xxx và opt-cpu.

Câu hỏi:
- Bao nhiêu % requests được score bởi predictor (aux_model_score not None)?
- Tỷ lệ này thay đổi thế nào theo rate?
- Requests nào KHÔNG được score? (output_len phân bố thế nào)

Nguồn data: SERVE_RES/latency-{sched}-...-r{rate}.0-...-o-1.pt
  → aux_model_scores ở index 8: float = scored, None = unscored.

Lưu ý: aux_model_score = None nghĩa là `set_aux_model_score()` chưa được gọi
cho request này TRƯỚC khi nó done. Có thể vì:
  1. Request admit trong FCFS warmup phase (2s đầu, opt-cpu)
  2. Predictor đang busy lúc request đến → submit_async skip
  3. Request admit và done trước khi Future kế tiếp bao gồm nó

Script này KHÔNG phân biệt được "scored before admit" vs "scored after admit"
— cần Level 2 (server-side timing log) để biết.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
SERVE = ROOT / "SERVE_RES"
MERGED = ROOT / "TEMP_RES_ASYNC_MERGE"
OUT = ROOT / "bench_analysis"
OUT.mkdir(exist_ok=True)

PT_KEYS = ["ttfts", "real_tpots", "latencies", "nlatencies", "actual_output_lens",
           "input_lens", "est_lens", "texts", "aux_model_scores", "pred_scores"]

SCHEDS = ["opt-xxx", "opt-cpu-warmup2.0", "opt-cpu-async-merged1.0"]
COLORS = {"opt-xxx": "#d62728", "opt-cpu-warmup2.0": "#1f77b4",
          "opt-cpu-async-merged1.0": "#2ca02c"}
LABELS = {"opt-xxx": "opt-xxx", "opt-cpu-warmup2.0": "opt-cpu",
          "opt-cpu-async-merged1.0": "opt-cpu-async-merged"}
RATES = [2, 4, 8, 16, 32, 64]
DIRS = {"opt-xxx": SERVE, "opt-cpu-warmup2.0": SERVE,
        "opt-cpu-async-merged1.0": MERGED}


def load(sched, rate):
    base = DIRS.get(sched, SERVE)
    p = base / f"latency-{sched}-Meta-Llama-3-8B-Instruct-p0-r{float(rate)}-c1.0-t60.0-o-1.pt"
    if not p.exists():
        return None
    raw = torch.load(p)
    return {k: raw[i] for i, k in enumerate(PT_KEYS) if i < len(raw)}


def is_scored(x):
    """True nếu aux_model_score là số hợp lệ."""
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def main():
    # Bảng tổng hợp
    print("=" * 75)
    print(f"{'sched':<22}{'rate':>5}{'n_total':>10}{'n_scored':>11}{'%scored':>10}")
    print("=" * 75)

    summary = {}   # (sched, rate) -> dict stats

    for sched in SCHEDS:
        for rate in RATES:
            r = load(sched, rate)
            if r is None:
                continue
            aux = r["aux_model_scores"]
            output_lens = np.asarray(r["actual_output_lens"], dtype=float)
            # Một số run có size mismatch — truncate về min
            n = min(len(aux), len(output_lens))
            aux = aux[:n]
            output_lens = output_lens[:n]
            scored_mask = np.array([is_scored(x) for x in aux])
            n_scored = scored_mask.sum()
            pct = 100 * n_scored / n
            print(f"{LABELS[sched]:<22}{rate:>5}{n:>10}{n_scored:>11}{pct:>9.1f}%")
            summary[(sched, rate)] = {
                "n": n, "n_scored": int(n_scored), "pct": pct,
                "scored_mask": scored_mask, "output_lens": output_lens,
            }
        print("-" * 75)

    # ───────────────── Plot 1: % scored vs rate ─────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for sched in SCHEDS:
        rates_x = []; pcts = []
        for rate in RATES:
            if (sched, rate) in summary:
                rates_x.append(rate)
                pcts.append(summary[(sched, rate)]["pct"])
        ax.plot(rates_x, pcts, "o-", label=LABELS[sched],
                color=COLORS[sched], linewidth=2, markersize=10)
        for x, y in zip(rates_x, pcts):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xlabel("request rate (qps)")
    ax.set_ylabel("% requests scored by predictor")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_ylim(-5, 110)
    ax.set_title("Scoring rate vs request rate\n"
                 "Cao = ranking effective, Thấp = phần lớn admit theo FCFS fallback")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "scoring_01_pct_vs_rate.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: scoring_01_pct_vs_rate.png")

    # ───────────────── Plot 2: scored vs unscored output_len distribution ─
    # Chỉ vẽ cho opt-cpu (opt-xxx 100% scored nên không có data unscored)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, rate in zip(axes, RATES):
        if ("opt-cpu-warmup2.0", rate) not in summary:
            continue
        s = summary[("opt-cpu-warmup2.0", rate)]
        scored_lens = s["output_lens"][s["scored_mask"]]
        unscored_lens = s["output_lens"][~s["scored_mask"]]
        bins = np.logspace(0, np.log10(max(s["output_lens"].max(), 1)), 30)
        ax.hist(scored_lens, bins=bins, alpha=0.6, label=f"scored (n={len(scored_lens)})",
                color="#2ca02c")
        ax.hist(unscored_lens, bins=bins, alpha=0.6, label=f"unscored (n={len(unscored_lens)})",
                color="#888888")
        ax.set_xscale("log")
        ax.set_xlabel("output_len (tokens)")
        ax.set_ylabel("count")
        ax.set_title(f"opt-cpu @ qps={rate}: scored vs unscored requests by output_len")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Output length distribution — scored vs unscored (opt-cpu only)\n"
                 "Pattern: scored requests có output dài hơn → admit sớm trong warmup → live lâu hơn để được score sau",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "scoring_02_output_len_split.png", dpi=130)
    plt.close(fig)
    print(f"Saved: scoring_02_output_len_split.png")

    # ───────────────── Plot 3: TPOT scored vs unscored (opt-cpu rate=16) ─
    # Hypothesis: unscored requests bị admit theo FCFS fallback → khả năng
    # rơi vào batch lớn (không có ranking ưu tiên ngắn) → TPOT cao hơn.
    s = summary.get(("opt-cpu-warmup2.0", 16))
    if s is not None:
        r = load("opt-cpu-warmup2.0", 16)
        tpot = np.asarray(r["real_tpots"], dtype=float)
        scored_mask = s["scored_mask"]
        # Truncate tpot về min với scored_mask
        n = min(len(tpot), len(scored_mask))
        tpot = tpot[:n]
        scored_mask = scored_mask[:n]
        scored_tpot = tpot[scored_mask]
        unscored_tpot = tpot[~scored_mask]

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot([scored_tpot, unscored_tpot],
                        labels=[f"scored (n={len(scored_tpot)})\nmean={scored_tpot.mean():.2f}s",
                                f"unscored (n={len(unscored_tpot)})\nmean={unscored_tpot.mean():.2f}s"],
                        patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor("#2ca02c"); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#888888"); bp["boxes"][1].set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_ylabel("TPOT (s/token)")
        ax.set_title("opt-cpu @ qps=16: TPOT của scored vs unscored requests\n"
                     "Nếu unscored TPOT cao hơn → ranking giúp giảm contention cho phần được score")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT / "scoring_03_tpot_scored_vs_unscored.png", dpi=130)
        plt.close(fig)
        print(f"Saved: scoring_03_tpot_scored_vs_unscored.png")

        print(f"\nopt-cpu @ qps=16 detail:")
        print(f"  scored:   n={len(scored_tpot)}, TPOT mean={scored_tpot.mean():.3f}s, "
              f"p99={np.percentile(scored_tpot, 99):.3f}s, "
              f"output_len mean={s['output_lens'][scored_mask].mean():.0f}")
        print(f"  unscored: n={len(unscored_tpot)}, TPOT mean={unscored_tpot.mean():.3f}s, "
              f"p99={np.percentile(unscored_tpot, 99):.3f}s, "
              f"output_len mean={s['output_lens'][~scored_mask].mean():.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
