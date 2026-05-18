import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def load_jsonl(path):
    prompts, completions = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d["prompt"])
            completions.append(d.get("generated", ""))
    return prompts, completions


def token_lengths(texts, tokenizer, batch=512):
    lengths = []
    for i in range(0, len(texts), batch):
        ids = tokenizer(texts[i : i + batch], add_special_tokens=False).input_ids
        lengths.extend(len(x) for x in ids)
    return np.array(lengths)


def stats(arr):
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="llama3-8b-sharegpt-test-t1-s0-8192.jsonl")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--save-dir", default="plots_distribution/single")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.file))[0]

    print(f"Loading {args.file}")
    prompts, completions = load_jsonl(args.file)
    print(f"  {len(prompts)} samples")

    print(f"Loading tokenizer {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    print("Tokenizing prompts...")
    in_lens = token_lengths(prompts, tok)
    print("Tokenizing completions...")
    out_lens = token_lengths(completions, tok)
    tot_lens = in_lens + out_lens

    s_in, s_out, s_tot = stats(in_lens), stats(out_lens), stats(tot_lens)
    print("\nInput :", s_in)
    print("Output:", s_out)
    print("Total :", s_tot)

    # ---- 1x3 histograms ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    series = [
        ("Input length",  in_lens,  "steelblue", s_in),
        ("Output length", out_lens, "darkorange", s_out),
        ("Total length",  tot_lens, "seagreen",  s_tot),
    ]
    for ax, (title, arr, color, st) in zip(axes, series):
        ax.hist(arr, bins=args.bins, color=color, edgecolor="black", alpha=0.8)
        ax.axvline(st["mean"],   color="red",    ls="--", lw=1.2, label=f"mean={st['mean']:.0f}")
        ax.axvline(st["median"], color="black",  ls="--", lw=1.2, label=f"median={st['median']:.0f}")
        ax.axvline(st["p95"],    color="purple", ls=":",  lw=1.2, label=f"p95={st['p95']:.0f}")
        ax.set_title(f"{title} (n={st['count']})")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(name, fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(args.save_dir, f"{name}_hist.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- 2D joint scatter + marginals ----
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(in_lens, out_lens, s=4, alpha=0.25, color="steelblue")
    ax_main.set_xlabel("Input tokens")
    ax_main.set_ylabel("Output tokens")
    ax_main.grid(alpha=0.3)

    ax_top.hist(in_lens, bins=args.bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Freq")

    ax_right.hist(out_lens, bins=args.bins, orientation="horizontal",
                  color="darkorange", edgecolor="black", alpha=0.8)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Freq")

    fig.suptitle(f"{name}\ninput vs output token length", fontsize=12)
    out2 = os.path.join(args.save_dir, f"{name}_joint.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # ---- CSV summary ----
    csv_path = os.path.join(args.save_dir, f"{name}_stats.csv")
    with open(csv_path, "w") as f:
        f.write("series,count,mean,median,p90,p95,p99,min,max\n")
        for label, st in (("input", s_in), ("output", s_out), ("total", s_tot)):
            f.write(f"{label},{st['count']},{st['mean']:.2f},{st['median']:.2f},"
                    f"{st['p90']:.2f},{st['p95']:.2f},{st['p99']:.2f},"
                    f"{st['min']},{st['max']}\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
