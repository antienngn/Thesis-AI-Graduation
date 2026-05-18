"""plot_priority_verify.py — Verify warmup-era priority invariant của merged scheduler.

Đọc tick profile CSV của opt-cpu-async-merged, tạo plot tập trung trả lời 1 câu hỏi:
    "Trong drain phase, warmup-era requests có LUÔN được ưu tiên admit
     trước post-warmup scored không?"

Invariant kỳ vọng (từ composite sort key):
    sort_key = (0 if warmup_era else 1, ...)
    → Mọi tick: warmup-era waiting phải gần 0 (admit ASAP).
    → Post-warmup scored chỉ admit vào slot KV CÒN LẠI sau warmup-era.

Plot 3 panel:
    Panel 1 (CHÍNH): n_running stacked — warmup-era ở dưới, post-warmup ở trên.
        Trong drain phase, lớp dưới (warmup-era) phải là FOUNDATION ổn định.
        Lớp trên (post-warmup) admit BÊN TRÊN — không bao giờ thay thế lớp dưới.
    Panel 2: n_waiting stacked — warmup-era waiting, post-warmup scored,
        post-warmup unscored. Warmup-era waiting phải gần 0 (priority absolute).
    Panel 3: Decision verification — vẽ tỉ số
        n_warmup_era_waiting / (n_warmup_era_waiting + n_post_scored_waiting)
        Trong drain: tỉ số này == 0 (không có warmup-era nào kẹt trong waiting
        khi có slot trống) là PASS. Nếu > 0 = warmup-era bị bỏ lại = invariant FAIL.

Usage:
    python plot_priority_verify.py <path_to_csv>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: str) -> str:
    csv = Path(csv_path)
    df = pd.read_csv(csv)

    # Derive: post-warmup running = total running - warmup-era running
    df["n_post_running"] = df.n_running - df.n_warmup_era_running

    # Stage transitions
    s2 = df[df.stage == 2]
    t_warmup_end = s2.t_rel.min() if len(s2) else df.t_rel.min()

    # "Drain phase" = khoảng warmup-era còn ở running (post warmup_end mà
    # n_warmup_era_running > 0). Khác với "Stage 2 dwell" ở async-warmup cũ —
    # ở merged, drain đồng thời với post-warmup admission.
    drain_mask = (df.t_rel >= t_warmup_end) & (df.n_warmup_era_running > 0)
    drain_end = df[drain_mask].t_rel.max() if drain_mask.any() else t_warmup_end
    drain_dwell = drain_end - t_warmup_end

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 10, "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"Priority verification — {csv.name}\n"
        f"Warmup end: t={t_warmup_end:.2f}s  |  "
        f"Drain phase (warmup-era running > 0): {drain_dwell:.2f}s  |  "
        f"Câu hỏi: warmup-era LUÔN admit trước post-warmup scored?",
        fontsize=12, y=0.995,
    )

    # Tô background drain phase (vùng cần verify invariant)
    def shade(ax):
        ax.axvspan(0, t_warmup_end, alpha=0.10, color="tab:blue", zorder=0,
                   label="warmup phase" if ax is axes[0] else None)
        ax.axvspan(t_warmup_end, drain_end, alpha=0.20, color="tab:orange",
                   zorder=0, label="DRAIN (verify here)" if ax is axes[0]
                   else None)
        ax.axvspan(drain_end, df.t_rel.max(), alpha=0.10, color="tab:green",
                   zorder=0, label="steady state" if ax is axes[0] else None)

    # =====================================================================
    # PANEL 1 — n_running stacked: warmup-era (bottom) + post-warmup (top)
    # =====================================================================
    # KỲ VỌNG: Trong vùng cam (drain), warmup-era LUÔN là lớp dưới chiếm
    # đầy trước. Post-warmup chỉ "đắp" lên trên. Nếu warmup-era xuống 0
    # trong khi post-warmup còn cao → là sau drain, OK.
    # SAI nếu: post-warmup có giá trị NHƯNG warmup-era WAITING > 0 cùng
    # lúc (panel 2 sẽ catch).
    ax1 = axes[0]
    shade(ax1)
    ax1.fill_between(df.t_rel, 0, df.n_warmup_era_running,
                     color="tab:red", alpha=0.7,
                     label="warmup-era running (foundation, drain dần)")
    ax1.fill_between(df.t_rel, df.n_warmup_era_running, df.n_running,
                     color="tab:cyan", alpha=0.6,
                     label="post-warmup running (đắp lên trên slot còn lại)")
    ax1.set_ylabel("n_running (stacked)")
    ax1.set_title(
        "PANEL 1: n_running phân theo class — warmup-era (đỏ) phải là "
        "foundation suốt drain"
    )
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # =====================================================================
    # PANEL 2 — n_waiting stacked: warmup-era waiting (PRIORITY CHECK) +
    # post-warmup scored + post-warmup unscored
    # =====================================================================
    # KỲ VỌNG: warmup-era waiting (lớp DƯỚI) phải GẦN 0 trong drain — vì
    # composite key (0, arrival_time) ưu tiên cứng → admit ngay khi có slot.
    # Nếu warmup-era waiting > 0 đáng kể TRONG khi post-warmup scored
    # đang được admit → INVARIANT FAIL.
    ax2 = axes[1]
    shade(ax2)
    cum1 = df.n_warmup_era_waiting
    cum2 = cum1 + df.n_postwarmup_waiting_scored
    cum3 = cum2 + df.n_postwarmup_waiting_unscored

    ax2.fill_between(df.t_rel, 0, cum1, color="tab:red", alpha=0.8,
                     label="warmup-era WAITING (PRIORITY CHECK — phải ~ 0)")
    ax2.fill_between(df.t_rel, cum1, cum2, color="tab:orange", alpha=0.6,
                     label="post-warmup SCORED waiting (admit theo SJF)")
    ax2.fill_between(df.t_rel, cum2, cum3, color="tab:gray", alpha=0.5,
                     label="post-warmup UNSCORED (gated by predictor)")
    ax2.set_ylabel("n_waiting (stacked)")
    ax2.set_title(
        "PANEL 2: n_waiting phân theo class — warmup-era waiting (đỏ) phải "
        "luôn ~ 0 nếu priority đúng"
    )
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # =====================================================================
    # PANEL 3 — Invariant check: số lượng tick "vi phạm priority"
    # =====================================================================
    # Tick "vi phạm" = drain phase tick mà n_warmup_era_waiting > 0 đồng thời
    # n_post_running > 0. Nghĩa là CÓ post-warmup admit khi CÒN warmup-era
    # đứng đợi — đảo priority.
    # Plot dạng scatter/event để dễ đếm.
    ax3 = axes[2]
    shade(ax3)
    drain_df = df[drain_mask].copy()
    drain_df["violate"] = ((drain_df.n_warmup_era_waiting > 0)
                          & (drain_df.n_post_running > 0))

    ax3.plot(df.t_rel, df.n_warmup_era_waiting, color="tab:red", lw=1.5,
             label="warmup-era waiting count")
    ax3.plot(df.t_rel, df.n_post_running, color="tab:cyan", lw=1.0,
             alpha=0.7, label="post-warmup running count")

    # Highlight các tick vi phạm
    if drain_df.violate.any():
        violations = drain_df[drain_df.violate]
        ax3.scatter(violations.t_rel,
                    [violations.n_warmup_era_waiting.max() * 1.1] * len(violations),
                    marker="x", color="black", s=30, zorder=5,
                    label=f"VIOLATION ticks: {len(violations)} "
                    f"(warmup-era waiting > 0 + post-warmup running > 0)")

    ax3.set_ylabel("count")
    ax3.set_xlabel("t_rel (giây từ serve_start)")
    ax3.set_title(
        "PANEL 3: Invariant check — đếm tick warmup-era kẹt waiting trong "
        "khi post-warmup đang chạy"
    )
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = csv.with_name(csv.stem + "_priority.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()

    # In summary verification result
    n_drain = drain_mask.sum()
    n_violate = drain_df.violate.sum() if len(drain_df) else 0
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Drain phase ticks: {n_drain}")
    print(f"Violation ticks (warmup-era waiting > 0 AND "
          f"post-warmup running > 0): {n_violate}")
    print(f"  → {'PASS — warmup-era priority maintained' if n_violate == 0 else 'FAIL — priority inversion detected'}")
    if n_drain > 0:
        max_warmup_wait = drain_df.n_warmup_era_waiting.max()
        max_post_run = drain_df.n_post_running.max()
        print(f"Max n_warmup_era_waiting in drain: {max_warmup_wait}")
        print(f"Max n_post_running in drain: {max_post_run}")

    return str(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_priority_verify.py <path_to_csv>")
        sys.exit(1)
    out = plot(sys.argv[1])
    print(f"Saved: {out}")
