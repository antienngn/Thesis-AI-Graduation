"""plot_predictor_dynamics.py — Visualize predictor behavior thực tế.

Đọc tick profile CSV, vẽ 3 panel cho thấy:
  Panel 1: queue_depth + in_flight (predictor pipeline state) over time
  Panel 2: scored backlog vs unscored backlog (gate effect)
  Panel 3: throughput đo được (delta scored count / delta t)

Mục đích: hiểu predictor work pattern trước khi tune chunk_size + T.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(csv_path: str) -> str:
    csv = Path(csv_path)
    df = pd.read_csv(csv)

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # === Panel 1: Predictor pipeline state ===
    # queue_depth = items waiting for worker pull
    # in_flight = items submitted but not yet completed (queue + currently computing)
    ax1 = axes[0]
    ax1.fill_between(df.t_rel, 0, df.stream_queue_depth,
                     color="tab:purple", alpha=0.6,
                     label="stream_queue_depth (chờ worker pull)")
    ax1.fill_between(df.t_rel, df.stream_queue_depth, df.stream_in_flight,
                     color="tab:brown", alpha=0.5,
                     label="stream_in_flight - queue (đang compute trong worker)")
    ax1.axhline(50, color="red", ls=":", alpha=0.5, lw=0.8,
                label="ngưỡng 50 (predictor bottleneck)")
    ax1.set_ylabel("count")
    ax1.set_title("PANEL 1: Predictor pipeline — queue depth + items in compute")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # === Panel 2: Scored vs unscored backlog ===
    # Trong waiting queue, post-warmup tách 2 loại:
    #   scored (ready, sẽ vào running) vs unscored (đang chờ predictor)
    ax2 = axes[1]
    ax2.fill_between(df.t_rel, 0, df.n_postwarmup_waiting_scored,
                     color="tab:orange", alpha=0.6,
                     label="post-warmup SCORED (sẵn sàng admit)")
    cum = df.n_postwarmup_waiting_scored + df.n_postwarmup_waiting_unscored
    ax2.fill_between(df.t_rel,
                     df.n_postwarmup_waiting_scored, cum,
                     color="tab:gray", alpha=0.5,
                     label="post-warmup UNSCORED (đang chờ predictor)")
    ax2.set_ylabel("count")
    ax2.set_title("PANEL 2: Waiting queue — scored vs unscored (gate effect)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Predictor instantaneous throughput ===
    # Đo bằng cách: nếu có N item rời in_flight giữa 2 tick, throughput
    # ≈ N / Δt. Dùng delta của in_flight nhưng cẩn thận: in_flight tăng
    # khi submit, giảm khi worker xong.
    # Approx throughput bằng: scored_added_per_sec
    # (n_scored tăng khi predictor xong 1 batch)
    # Đơn giản hơn: dùng cumulative score count derived
    df_sorted = df.sort_values("t_rel").reset_index(drop=True)
    # Approx: rate of in_flight decrease = scoring rate
    df_sorted["d_inflight"] = df_sorted.stream_in_flight.diff()
    df_sorted["d_t"] = df_sorted.t_rel.diff()
    # When in_flight DECREASES that's worker completing (rate >0)
    df_sorted["rate"] = -df_sorted.d_inflight / df_sorted.d_t
    df_sorted["rate"] = df_sorted["rate"].clip(lower=0)
    # Smooth với rolling window 30 ticks
    rate_smooth = df_sorted["rate"].rolling(30, min_periods=1).mean()
    ax3 = axes[2]
    ax3.plot(df_sorted.t_rel, rate_smooth, color="tab:green", lw=1.0,
             label="estimated predictor throughput (smoothed, req/s)")
    ax3.axhline(80, color="orange", ls=":", alpha=0.5, lw=0.8,
                label="theoretical chunk=4 throughput (~80 req/s)")
    ax3.axhline(267, color="red", ls=":", alpha=0.5, lw=0.8,
                label="theoretical chunk=32 throughput (~267 req/s)")
    ax3.set_ylabel("req/s")
    ax3.set_xlabel("t_rel (giây)")
    ax3.set_title("PANEL 3: Predictor throughput (smoothed) vs theoretical chunk_size limits")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 200)

    fig.suptitle(
        f"Predictor dynamics — {csv.name}\n"
        f"Hiểu pattern trước khi tune chunk_size + T",
        fontsize=12, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = csv.with_name(csv.stem + "_predictor.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()

    # Print summary stats
    print(f"\n=== Predictor stats ===")
    print(f"Max queue depth ever:    {df.stream_queue_depth.max()}")
    print(f"Max in_flight ever:      {df.stream_in_flight.max()}")
    print(f"Mean queue depth (post warmup phase): "
          f"{df[df.stage>=2].stream_queue_depth.mean():.1f}")
    print(f"Tick where queue first exceeded 30: "
          f"{df[df.stream_queue_depth > 30].t_rel.min() if (df.stream_queue_depth > 30).any() else 'never'}")
    print(f"Estimated max instant throughput: "
          f"{rate_smooth.max():.0f} req/s")
    print(f"Estimated avg throughput (during stage 2): "
          f"{df_sorted[df_sorted.stage>=2]['rate'].rolling(30, min_periods=1).mean().mean():.1f} req/s")
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_predictor_dynamics.py <path_to_csv>")
        sys.exit(1)
    out = plot(sys.argv[1])
    print(f"\nSaved: {out}")
