"""
Plot the distribution of requests actually sent by benchmark_serving_real.py
for each (request_rate, request_time) scenario from bench_ser_ov_async.sh.

Replicates the sampling/arrival logic in benchmark_serving_real.py:
  - num_prompts = rate * time
  - Load first num_prompts * 1.2 lines from the jsonl (order preserved)
  - Filter: 4 <= prompt_len <= 1024, output_len >= 4, prompt+output <= 20480
  - Take first num_prompts post-filter
  - Inter-arrival ~ Gamma(shape=1/cv^2, scale=cv^2/rate), default cv=1 -> Exp(1/rate)
  - seed = 0 (matches benchmark default)
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


# bench_ser_ov_async.sh scenarios (all commented except r=64, but the file
# represents the full sweep so we plot them all)
DEFAULT_RATES = [2, 4, 8, 16, 32, 64]
DEFAULT_TIME = 60
DEFAULT_CV = 1.0
DEFAULT_SEED = 0


def sample_for_scenario(prompts, comp_token_ids, prompt_token_ids,
                        num_prompts, ignore_limit=False):
    """Mimics sample_requests() in benchmark_serving_real.py."""
    take = int(num_prompts * 1.2)
    assert len(prompts) >= take, f"need {take} rows, have {len(prompts)}"

    sampled = []
    for i in range(take):
        p_len = len(prompt_token_ids[i])
        o_len = len(comp_token_ids[i])
        if not ignore_limit:
            if p_len < 4 or o_len < 4:
                continue
            if p_len > 1024 or p_len + o_len > 20480:
                continue
        sampled.append((p_len, o_len))
        if len(sampled) >= num_prompts:
            break

    assert len(sampled) >= num_prompts, (
        f"after filter only {len(sampled)} < {num_prompts}; "
        "increase the 1.2x headroom or check filter limits"
    )
    return np.array(sampled[:num_prompts])  # shape (N, 2): (in, out)


def arrival_times(num_prompts, rate, cv, rng):
    """Cumulative arrival times under Gamma inter-arrival."""
    shape = 1.0 / (cv * cv)
    scale = (cv * cv) / rate
    # The benchmark yields request[0] immediately, then sleeps before request[1]..N-1.
    # So there are num_prompts-1 intervals.
    intervals = rng.gamma(shape, scale, size=num_prompts - 1)
    arrivals = np.concatenate([[0.0], np.cumsum(intervals)])
    return arrivals, intervals


def plot_per_scenario(arr_in, arr_out, arrivals, intervals, rate, time_s,
                      save_dir, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n = len(arr_in)
    total_dur = arrivals[-1] if len(arrivals) else 0.0

    # Input lengths
    ax = axes[0, 0]
    ax.hist(arr_in, bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(arr_in.mean(),   color="red",   ls="--", lw=1.2,
               label=f"mean={arr_in.mean():.0f}")
    ax.axvline(np.median(arr_in), color="black", ls="--", lw=1.2,
               label=f"median={np.median(arr_in):.0f}")
    ax.axvline(np.percentile(arr_in, 95), color="purple", ls=":", lw=1.2,
               label=f"p95={np.percentile(arr_in,95):.0f}")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Input length  (n={n})")
    ax.grid(alpha=0.3); ax.legend()

    # Output lengths
    ax = axes[0, 1]
    ax.hist(arr_out, bins=40, color="darkorange", edgecolor="black", alpha=0.85)
    ax.axvline(arr_out.mean(),   color="red",   ls="--", lw=1.2,
               label=f"mean={arr_out.mean():.0f}")
    ax.axvline(np.median(arr_out), color="black", ls="--", lw=1.2,
               label=f"median={np.median(arr_out):.0f}")
    ax.axvline(np.percentile(arr_out, 95), color="purple", ls=":", lw=1.2,
               label=f"p95={np.percentile(arr_out,95):.0f}")
    ax.set_xlabel("Output tokens")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Output length  (n={n})")
    ax.grid(alpha=0.3); ax.legend()

    # Arrival timeline (cumulative requests over time)
    ax = axes[1, 0]
    ax.plot(arrivals, np.arange(1, n + 1), lw=1.2, color="seagreen",
            label="actual arrivals")
    ideal_t = np.linspace(0, total_dur, 200) if total_dur > 0 else np.array([0])
    ax.plot(ideal_t, ideal_t * rate, ls="--", color="grey",
            label=f"ideal rate={rate}/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative requests")
    ax.set_title(f"Arrival timeline  (duration={total_dur:.1f}s, "
                 f"target={time_s}s)")
    ax.grid(alpha=0.3); ax.legend()

    # Inter-arrival distribution
    ax = axes[1, 1]
    if len(intervals):
        ax.hist(intervals * 1000, bins=40, color="indianred",
                edgecolor="black", alpha=0.85)
        ax.axvline(intervals.mean() * 1000, color="black", ls="--", lw=1.2,
                   label=f"mean={intervals.mean()*1000:.1f}ms  "
                         f"(ideal={1000/rate:.1f}ms)")
    ax.set_xlabel("Inter-arrival (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Inter-arrival times  Gamma(shape=1/cv², scale=cv²/rate)")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(
        f"{dataset_name} | rate={rate} qps  time={time_s}s  -> num_prompts={n}",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(save_dir, f"scenario_r{rate}_t{time_s}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overlay(scenarios, save_dir, dataset_name):
    """Overlay input/output CDFs across scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap("viridis")
    for idx, s in enumerate(scenarios):
        c = cmap(idx / max(1, len(scenarios) - 1))
        for ax, key, label in ((axes[0], "in", "input"),
                               (axes[1], "out", "output")):
            xs = np.sort(s[key])
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=c, lw=1.3,
                    label=f"r={s['rate']} (n={len(xs)})")
    for ax, title in ((axes[0], "Input length CDF"),
                      (axes[1], "Output length CDF")):
        ax.set_xlabel("Tokens"); ax.set_ylabel("CDF")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"{dataset_name} | request length CDFs per scenario")
    fig.tight_layout()
    out = os.path.join(save_dir, "scenarios_length_cdf.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="llama3-8b-sharegpt-test-t1-s0-8192.jsonl")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--rates", type=int, nargs="+", default=DEFAULT_RATES)
    ap.add_argument("--time", type=int, default=DEFAULT_TIME,
                    help="--request-time used in the .sh")
    ap.add_argument("--cv", type=float, default=DEFAULT_CV)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--save-dir", default="plots_distribution/scenarios_async")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    # Tokenize once over the largest prefix we'll need.
    max_take = int(max(args.rates) * args.time * 1.2)
    print(f"Loading first {max_take} rows from {args.dataset}")
    prompts, completions = [], []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_take:
                break
            d = json.loads(line)
            prompts.append(d["prompt"])
            completions.append(d.get("generated", ""))

    print(f"Loading tokenizer {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    print("Tokenizing prompts + completions...")
    prompt_ids = tok(prompts).input_ids
    comp_ids = tok(completions).input_ids

    summary_rows = []
    scenarios = []
    for rate in args.rates:
        num_prompts = rate * args.time
        print(f"\n=== rate={rate} qps  num_prompts={num_prompts} ===")
        lens = sample_for_scenario(prompts, comp_ids, prompt_ids, num_prompts)
        arr_in, arr_out = lens[:, 0], lens[:, 1]
        total = arr_in + arr_out

        rng = np.random.RandomState(args.seed)
        arrivals, intervals = arrival_times(num_prompts, rate, args.cv, rng)

        out_png = plot_per_scenario(arr_in, arr_out, arrivals, intervals,
                                    rate, args.time, args.save_dir, dataset_name)
        print(f"  saved {out_png}")
        print(f"  input  mean={arr_in.mean():.1f}  median={np.median(arr_in):.0f}  "
              f"p95={np.percentile(arr_in,95):.0f}  max={arr_in.max()}")
        print(f"  output mean={arr_out.mean():.1f}  median={np.median(arr_out):.0f}  "
              f"p95={np.percentile(arr_out,95):.0f}  max={arr_out.max()}")
        print(f"  duration actual={arrivals[-1]:.1f}s  target={args.time}s  "
              f"observed_rate={num_prompts/arrivals[-1]:.2f}/s")

        summary_rows.append({
            "rate": rate, "num_prompts": num_prompts,
            "in_mean": arr_in.mean(), "in_p95": np.percentile(arr_in, 95),
            "in_max": int(arr_in.max()),
            "out_mean": arr_out.mean(), "out_p95": np.percentile(arr_out, 95),
            "out_max": int(arr_out.max()),
            "tot_mean": total.mean(), "tot_p95": np.percentile(total, 95),
            "duration_s": float(arrivals[-1]),
        })
        scenarios.append({"rate": rate, "in": arr_in, "out": arr_out})

    overlay = plot_overlay(scenarios, args.save_dir, dataset_name)
    print(f"\nSaved overlay {overlay}")

    csv_path = os.path.join(args.save_dir, "scenarios_summary.csv")
    keys = list(summary_rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(f"{r[k]:.2f}" if isinstance(r[k], float)
                             else str(r[k]) for k in keys) + "\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
