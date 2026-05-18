#!/usr/bin/env python3
"""plot_scheduler_predictor_flow.py — Visualize flow nhận request của
scheduler opt-cpu-async-merged + tương tác với OpenVINOPredictor.

Vẽ 1 diagram tổng + sequence diagram cho 1 tick (Stage 2 merged phase).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(20, 14))

# ============================================================
# TOP: Architecture overview
# ============================================================
ax = fig.add_subplot(2, 1, 1)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")
ax.set_title("opt-cpu-async-merged: Architecture overview",
             fontsize=14, fontweight="bold", pad=10)


def box(ax, x, y, w, h, text, color, fontsize=9, fontweight="normal"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                           edgecolor="black", facecolor=color, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, wrap=True)


def arrow(ax, x1, y1, x2, y2, label="", color="black", style="->",
          rad=0.0, fontsize=8):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, color=color, mutation_scale=15,
                         linewidth=1.3, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.5, label, ha="center",
                color=color, fontsize=fontsize)


# Client → Engine
box(ax, 2, 42, 14, 5, "vLLM async-engine\n(receive HTTP)", "#FFE0B2",
    fontsize=9, fontweight="bold")
box(ax, 2, 34, 14, 5, "AsyncLLMEngine\n.add_request()", "#FFE0B2")

# Scheduler
box(ax, 22, 28, 56, 19,
    "Scheduler (singleton)\n"
    "self.waiting / self.running / self.swapped\n"
    "self.serve_start_time (lazy init)\n"
    "self.warmup_seconds = T (parse từ schedule_type)\n"
    "self.aux_model = OpenVINOPredictor\n"
    "self._get_ordered_requests = "
    "_get_opt_cpu_async_merged_ordered_requests",
    "#E1F5FE", fontsize=10, fontweight="bold")

# Predictor
box(ax, 84, 28, 14, 19,
    "OpenVINOPredictor\n"
    "(on CPU)\n\n"
    "_stream_queue\n"
    "_stream_in_flight\n"
    "_stream_lock\n"
    "_stream_chunk_size",
    "#E8F5E9", fontsize=9, fontweight="bold")

# Worker thread
box(ax, 84, 14, 14, 10,
    "Worker thread\n"
    "(ThreadPoolExecutor\n"
    " max_workers=1)\n\n"
    "_stream_worker_loop()",
    "#FFF9C4", fontsize=9, fontweight="bold")

# GPU model
box(ax, 22, 8, 30, 14,
    "GPU (main Llama-3-8B)\n"
    "model.execute_model()\n"
    "→ paged attention + KV cache\n"
    "→ token output (streaming)",
    "#FFCCBC", fontsize=10, fontweight="bold")

# Arrows
arrow(ax, 9, 41.5, 9, 39, "1. POST /v1/completions", fontsize=8)
arrow(ax, 16, 36, 22, 36, "2. add_seq_group(sg)\n   → self.waiting", fontsize=8)
arrow(ax, 78, 36, 84, 36, "3a. submit_streaming(post-warmup)", fontsize=8)
arrow(ax, 84, 24, 84, 24.5, "")
arrow(ax, 91, 28, 91, 24, "spawn worker\nif idle", color="darkred",
      fontsize=7)
arrow(ax, 84, 19, 78, 26, "3b. set_aux_model_score(s)\n   (cross-thread)",
      color="darkred", fontsize=7, rad=-0.2)
arrow(ax, 50, 28, 50, 22, "4. execute_model(batch)", fontsize=8)
arrow(ax, 52, 15, 78, 26, "5. token output → stream → client",
      color="#FF6F00", fontsize=8, rad=0.3)


# ============================================================
# BOTTOM: Sequence diagram for 1 scheduler tick (Stage 2)
# ============================================================
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 50)
ax2.axis("off")
ax2.set_title("Flow 1 tick scheduler ở Stage 2 (merged post-warmup phase)",
              fontsize=14, fontweight="bold", pad=10)

# Lifelines (vertical lines)
LANES = [
    ("Scheduler\n_get_opt_cpu_async_\nmerged_ordered_requests()",
     10, "#E1F5FE"),
    ("aux_model\n(OpenVINOPredictor)", 32, "#E8F5E9"),
    ("_stream_queue\n_stream_in_flight", 54, "#FFF9C4"),
    ("Worker thread\n_stream_worker_loop", 76, "#FFCCBC"),
    ("SequenceGroup\n(in waiting/running)", 95, "#F8BBD0"),
]
for label, x, color in LANES:
    ax2.add_patch(Rectangle((x - 8, 44), 16, 4,
                             edgecolor="black", facecolor=color))
    ax2.text(x, 46, label, ha="center", va="center", fontsize=8.5,
             fontweight="bold")
    ax2.plot([x, x], [3, 44], "k--", linewidth=0.6, alpha=0.5)


def step(ax, y, label, x_from, x_to, color="black", note_x=None, note=""):
    arrow(ax, x_from, y, x_to, y, color=color, fontsize=7.5)
    ax.text((x_from + x_to) / 2, y + 0.8, label, ha="center",
            color=color, fontsize=7.5)
    if note:
        ax.text(note_x if note_x else (x_from + x_to) / 2, y - 1.2,
                note, ha="center", color="dimgray", fontsize=7,
                style="italic")


# Steps (top to bottom = time)
y = 40
step(ax2, y, "elapsed = time() - serve_start_time",
     10, 10, color="navy")
ax2.text(10, y - 1.2, "→ elapsed ≥ T_warmup → Stage 2", ha="center",
         color="dimgray", fontsize=7, style="italic")

y -= 4
step(ax2, y, "1. poll_streaming()", 10, 32, color="purple")
ax2.text(32, y - 1.2, "→ no-op (return 0)", ha="center", color="dimgray",
         fontsize=7, style="italic")

y -= 4
ax2.text(10, y + 0.6,
         "2. Tìm need_aux_scores trong self.waiting\n   "
         "[r for r in self.waiting if not is_warmup_era(r) and "
         "r.need_aux_model_score()]",
         ha="left", color="darkgreen", fontsize=7.5)

y -= 5
step(ax2, y, "3. submit_streaming(need_aux_scores)", 10, 32, color="purple")

y -= 4
step(ax2, y, "4a. acquire _stream_lock", 32, 54, color="darkred")

y -= 3
step(ax2, y, "4b. for sg in seq_groups: if not in in_flight\n"
     "    → queue.append((sg, prompt)) + in_flight.add(rid)",
     32, 54, color="darkred")

y -= 4
step(ax2, y, "4c. if queue non-empty + worker idle\n"
     "    → spawn worker (executor.submit)",
     32, 76, color="darkred", note_x=54,
     note="(async, không block scheduler tick)")

y -= 5
step(ax2, y,
     "5. Schedulable build = running + swapped + warmup-era waiting\n"
     "                       + post-warmup SCORED waiting (gate filter)",
     10, 10, color="navy")
ax2.text(10, y - 1.5,
         "→ post-warmup unscored bị GATE khỏi schedulable",
         ha="center", color="darkred", fontsize=7, style="italic")

y -= 4
step(ax2, y,
     "6. assert: warmup-era OR aux_model_score is not None",
     10, 10, color="darkred")

y -= 4
step(ax2, y,
     "7. sort by composite key (class_priority, in_class_key)\n"
     "   warmup-era: (0, arrival_time)  |  post-warmup: (1, -score)",
     10, 10, color="navy")

y -= 4
step(ax2, y, "8. return sorted list → GPU executor",
     10, 10, color="navy")

# Worker thread parallel timeline (right side)
ax2.text(76, 3, "Worker thread (parallel, không block tick):",
         ha="center", fontsize=8, color="darkblue", fontweight="bold")
ax2.text(76, 1.5,
         "1. lock → pop ≤ chunk_size từ queue → unlock\n"
         "2. tokenize batch (HF Rust, GIL released)\n"
         "3. compiled_model([ids, mask]) — OV C++\n"
         "4. for each (sg, score): set_aux_model_score(score)  ← cross-thread\n"
         "5. discard request_id từ in_flight\n"
         "6. loop until queue empty → exit (sẽ spawn lại nếu cần)",
         ha="center", fontsize=7, color="darkblue")

# Cross-thread arrow from Worker to SequenceGroup
arrow(ax2, 76, 16, 95, 16,
      "set_aux_model_score(s)\n(GIL atomic single-attr)",
      color="darkred", rad=0.3, fontsize=7)

# Legend
ax2.text(50, 0.3,
         "Color: navy=scheduler logic | purple=predictor public API | "
         "darkred=cross-thread/sync | darkgreen=filter logic",
         ha="center", fontsize=7, color="black", style="italic")

fig.tight_layout()
out = "/home/antn/vllm-ltr/benchmarks/scheduler_predictor_flow.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Wrote {out}")
