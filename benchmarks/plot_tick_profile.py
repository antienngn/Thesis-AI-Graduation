"""plot_tick_profile.py — Visualize tick profile CSV.

Đọc 1 file CSV produce bởi _profile_tick_async_warmup, tạo timeline 4-panel
PNG cùng directory. Các stage được tô màu nền để dễ đọc:
  Stage 1 (warmup): xanh dương nhạt
  Stage 2 (drain):  cam nhạt   ← KHOẢNG QUAN TRỌNG
  Stage 3 (post):   xanh lá nhạt

Usage:
    python plot_tick_profile.py <path_to_csv>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def shade_stages(ax, t_warmup_end, t_drain_end, t_max):
    """Tô background cho 3 stage để người đọc nhìn 1 phát biết đang ở stage nào."""
    ax.axvspan(0, t_warmup_end, alpha=0.12, color="tab:blue", zorder=0)
    ax.axvspan(t_warmup_end, t_drain_end, alpha=0.18, color="tab:orange",
               zorder=0)
    ax.axvspan(t_drain_end, t_max, alpha=0.10, color="tab:green", zorder=0)


def plot(csv_path: str) -> str:
    csv = Path(csv_path)
    df = pd.read_csv(csv)

    # Stage transitions
    s2 = df[df.stage == 2]
    s3 = df[df.stage == 3]
    t_warmup_end = s2.t_rel.min() if len(s2) else df.t_rel.min()
    t_drain_end = s3.t_rel.min() if len(s3) else df.t_rel.max()
    t_max = df.t_rel.max()
    s2_dwell = t_drain_end - t_warmup_end

    # Settings global cho readability
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(
        f"Tick profile: {csv.name}\n"
        f"Background tô màu — Xanh dương=Stage 1 (warmup, 0→{t_warmup_end:.2f}s)  |  "
        f"Cam=Stage 2 (drain, dwell={s2_dwell:.2f}s)  |  "
        f"Xanh lá=Stage 3 (post-drain)",
        fontsize=13, y=0.995,
    )

    # =========================================================================
    # PANEL 1 — KV cache & warmup-era drain progress
    # =========================================================================
    # Câu hỏi chính của panel: "Trong Stage 2 (vùng cam), GPU có rảnh không?"
    # - Đường XANH DƯƠNG đậm (% KV trống): gần 100% trong Stage 2 = GPU rảnh
    # - Đường ĐỎ (warmup-era đang decode): cap ở 8 rồi tụt = drain progress
    ax1 = axes[0]
    shade_stages(ax1, t_warmup_end, t_drain_end, t_max)
    ax1b = ax1.twinx()

    free_pct = df.n_free_gpu_blocks / df.n_total_gpu_blocks * 100
    line1 = ax1.plot(df.t_rel, free_pct, color="tab:blue", lw=1.8,
                     label="% KV cache trống (trục TRÁI)")
    ax1.set_ylabel("% KV cache trống", color="tab:blue", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(20, color="tab:blue", ls=":", alpha=0.4, lw=0.8)
    ax1.text(t_max * 0.98, 22, "ngưỡng 20% (decision threshold)",
             ha="right", fontsize=8, color="tab:blue", alpha=0.7)

    line2 = ax1b.plot(df.t_rel, df.n_warmup_era_running, color="tab:red",
                      lw=1.8, label="số warmup-era đang decode (trục PHẢI)")
    ax1b.set_ylabel("số warmup-era đang decode", color="tab:red",
                    fontsize=11)
    ax1b.tick_params(axis="y", labelcolor="tab:red")
    ax1b.set_ylim(bottom=0)

    ax1.set_title(
        "PANEL 1: KV cache headroom & warmup-era drain  "
        "(GPU có chỗ trống không?)"
    )
    # Combined legend
    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right",
               framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # PANEL 2 — Waiting queue composition (METRIC CHÍNH)
    # =========================================================================
    # Câu hỏi: "C2 đang block bao nhiêu request đáng ra có thể chạy?"
    # - Vùng CAM tô đặc: post-warmup ĐÃ scored, đang bị Stage 2 quarantine
    # - Vùng XÁM phía trên: post-warmup chưa scored (predictor đang chạy)
    # - Đường ĐEN: tổng waiting (= warmup-era waiting + 2 vùng trên)
    ax2 = axes[1]
    shade_stages(ax2, t_warmup_end, t_drain_end, t_max)

    ax2.fill_between(df.t_rel, 0, df.n_postwarmup_waiting_scored,
                     color="tab:orange", alpha=0.65,
                     label="post-warmup đã SCORED (bị C2 block) — đáng lẽ chạy được")
    cum2 = df.n_postwarmup_waiting_scored + df.n_postwarmup_waiting_unscored
    ax2.fill_between(df.t_rel, df.n_postwarmup_waiting_scored, cum2,
                     color="tab:gray", alpha=0.55,
                     label="post-warmup CHƯA scored (predictor đang chạy)")
    ax2.plot(df.t_rel, df.n_waiting, color="black", lw=1.3,
             label="tổng waiting (đường viền trên)")

    ax2.set_ylabel("số request trong waiting queue")
    ax2.set_title(
        "PANEL 2: Waiting queue — vùng CAM = mất mát do C2 quarantine "
        "(METRIC CHÍNH)"
    )
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Annotate peak scored backlog
    peak_idx = df.n_postwarmup_waiting_scored.idxmax()
    peak_t = df.t_rel.iloc[peak_idx]
    peak_v = df.n_postwarmup_waiting_scored.iloc[peak_idx]
    ax2.annotate(
        f"đỉnh: {int(peak_v)} request\nbị block tại t={peak_t:.1f}s",
        xy=(peak_t, peak_v), xytext=(peak_t + t_max * 0.05, peak_v + 20),
        fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1),
    )

    # =========================================================================
    # PANEL 3 — Predictor pipeline + tick interval
    # =========================================================================
    # Câu hỏi: "Predictor có theo kịp không?"
    # - TÍM (queue): item đợi worker pull
    # - NÂU (in_flight): item đang trong worker pipeline (queue + đang compute)
    # - XANH LÁ mảnh (trục phải): tick interval — scheduler có chậm không
    ax3 = axes[2]
    shade_stages(ax3, t_warmup_end, t_drain_end, t_max)
    ax3b = ax3.twinx()

    line5 = ax3.plot(df.t_rel, df.stream_queue_depth, color="tab:purple",
                     lw=1.5, label="stream_queue_depth (chờ worker pull)")
    line6 = ax3.plot(df.t_rel, df.stream_in_flight, color="tab:brown",
                     lw=1.5, alpha=0.7,
                     label="stream_in_flight (đang trong pipeline)")
    ax3.set_ylabel("số item trong predictor pipeline")
    ax3.axhline(50, color="red", ls=":", alpha=0.4, lw=0.8)
    ax3.text(t_max * 0.98, 51, "ngưỡng 50 = predictor là bottleneck",
             ha="right", fontsize=8, color="red", alpha=0.7)

    tick_ms = df.t_rel.diff() * 1000
    line7 = ax3b.plot(df.t_rel, tick_ms, color="tab:green", lw=0.6,
                      alpha=0.5,
                      label="scheduler tick interval — ms (trục PHẢI)")
    ax3b.set_ylabel("tick interval (ms)", color="tab:green")
    ax3b.tick_params(axis="y", labelcolor="tab:green")

    ax3.set_title(
        "PANEL 3: Predictor pipeline + scheduler tick interval  "
        "(predictor có catch-up không?)"
    )
    lines3 = line5 + line6 + line7
    ax3.legend(lines3, [l.get_label() for l in lines3], loc="upper right",
               framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # PANEL 4 — Stage indicator + concurrent execution (n_running, n_swapped)
    # =========================================================================
    # Câu hỏi: "Khi Stage 3 mở cổng, hệ thống phản ứng thế nào?"
    # - VÀNG step (trục trái): stage 1/2/3
    # - CYAN: n_running — số request đang được serve. Sẽ thấy nó nhảy mạnh
    #   ngay khi qua Stage 2→3 boundary (cổng C2 mở).
    # - HỒNG: n_swapped — preemption (KV pressure)
    ax4 = axes[3]
    shade_stages(ax4, t_warmup_end, t_drain_end, t_max)
    ax4b = ax4.twinx()

    line8 = ax4.plot(df.t_rel, df.stage, color="tab:olive", lw=2.0,
                     drawstyle="steps-post",
                     label="stage hiện tại (trục TRÁI)")
    ax4.set_ylabel("stage (1=warmup, 2=drain, 3=post-drain)",
                   color="tab:olive")
    ax4.set_yticks([1, 2, 3])
    ax4.set_ylim(0.5, 3.5)
    ax4.tick_params(axis="y", labelcolor="tab:olive")

    line9 = ax4b.plot(df.t_rel, df.n_running, color="tab:cyan", lw=1.5,
                      label="n_running — request đang serve (trục PHẢI)")
    line10 = ax4b.plot(df.t_rel, df.n_swapped, color="tab:pink", lw=1.5,
                       label="n_swapped — bị preempt (trục PHẢI)")
    ax4b.set_ylabel("count")
    ax4b.set_ylim(bottom=0)

    ax4.set_title(
        "PANEL 4: Stage transitions + concurrent execution  "
        "(burst khi Stage 3 mở cổng?)"
    )
    lines4 = line8 + line9 + line10
    ax4.legend(lines4, [l.get_label() for l in lines4], loc="upper right",
               framealpha=0.9)
    ax4.set_xlabel("t_rel (giây từ serve_start_time)", fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Annotate burst
    if len(s3):
        burst_t = s3.t_rel.iloc[0:50].iloc[-1] if len(s3) > 50 else s3.t_rel.iloc[-1]
        burst_n = df[df.t_rel <= burst_t].n_running.max()
        ax4b.annotate(
            f"n_running peak ≈ {int(burst_n)}\n(burst sau khi C2 mở cổng)",
            xy=(burst_t, burst_n),
            xytext=(burst_t + t_max * 0.05, burst_n * 0.7),
            fontsize=9, color="darkblue",
            arrowprops=dict(arrowstyle="->", color="darkblue", lw=1),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = csv.with_suffix(".png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_tick_profile.py <path_to_csv>")
        sys.exit(1)
    out = plot(sys.argv[1])
    print(f"Saved: {out}")
