#!/bin/bash
# bench_predictor_sweep.sh — sweep batch_size cho OV CPU và GPU.
# Mỗi (backend, model, batch_size) chạy 1 process riêng để cô lập (giống bench_predictor.sh).
set -e

PYTHIA14M_DIR=MODEL/results/pythia-14m-llama3-8b-sharegpt-score-trainbucket10-b32
DATASET=llama3-8b-sharegpt-test-t1-s0-8192.jsonl
LLAMA_TOK=meta-llama/Meta-Llama-3-8B-Instruct
OUT_DIR=BENCH_PRED_RES/sweep_bs
GPU_ID=1
NUM_PROMPTS=500
N_ITERS=10
WARMUP=6
OV_PRECISION=bf16

BATCH_SIZES=(1 2 4 8 16 32)

# (tag, model_dir) pairs — Pythia 70M đã có kết quả; lần này chỉ chạy Pythia 14M trên OV CPU.
OV_MODELS=(
  "pythia14m:$PYTHIA14M_DIR"
)

mkdir -p "$OUT_DIR"

for BS in "${BATCH_SIZES[@]}"; do
  echo
  echo "############################################"
  echo "###  batch_size = $BS"
  echo "############################################"

  for entry in "${OV_MODELS[@]}"; do
    TAG="${entry%%:*}"
    MDIR="${entry#*:}"
    echo "--- [OV CPU $TAG, bs=$BS, precision=$OV_PRECISION] ---"
    python bench_predictor.py \
      --backend openvino \
      --config $MDIR/usage_config_ov.json \
      --llama-tokenizer $LLAMA_TOK \
      --dataset $DATASET \
      --num-prompts $NUM_PROMPTS \
      --batch-size $BS \
      --n-iters $N_ITERS \
      --warmup $WARMUP \
      --ov-precision $OV_PRECISION \
      --output "$OUT_DIR/openvino_${TAG}_bs${BS}.json"
  done
done

echo
echo "=========================================================="
echo "=== Summary (sweep batch_size: OpenVINO CPU vs GPU)"
echo "=========================================================="
python - <<PY
import json, glob, os, csv, re

out_dir = "$OUT_DIR"
rows = []
# openvino_<tag>_bs<N>.json, openvino_bs<N>.json (legacy = OPT-125M), hoặc gpu_bs<N>.json
pat = re.compile(r"(openvino(?:_(?P<tag>[a-zA-Z0-9]+))?|gpu)_bs(?P<bs>\d+)\.json")
for path in sorted(glob.glob(os.path.join(out_dir, "*_bs*.json"))):
    m = pat.match(os.path.basename(path))
    if not m: continue
    is_gpu = m.group(0).startswith("gpu")
    if is_gpu:
        backend = "gpu"
    else:
        tag = m.group("tag") or "opt125m"  # untagged legacy file = OPT-125M
        backend = f"ov_{tag}"
    bs = int(m.group("bs"))
    r = json.load(open(path))
    rows.append({
        "backend": backend,
        "batch_size": bs,
        "inference_time_ms": r["inference_time_ms"],
        "kendall_tau": r["kendall_tau"],
    })

rows.sort(key=lambda x: (x["batch_size"], x["backend"]))

# CSV
csv_path = os.path.join(out_dir, "sweep_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["backend", "batch_size", "inference_time_ms", "kendall_tau"])
    w.writeheader()
    w.writerows(rows)
print(f"[csv] {csv_path}")
print()

# Bảng so sánh: per-bs, mỗi backend 1 dòng + speedup so với GPU
by_bs = {}
for r in rows:
    by_bs.setdefault(r["batch_size"], {})[r["backend"]] = r

print(f"{'bs':>4}  {'backend':<14} {'ms':>10} {'tau':>8}  {'vs_gpu':>8}")
print("-" * 56)
for bs in sorted(by_bs):
    gpu_ms = by_bs[bs].get("gpu", {}).get("inference_time_ms")
    for backend in sorted(by_bs[bs]):
        r = by_bs[bs][backend]
        ms = r["inference_time_ms"]; tau = r["kendall_tau"]
        sp = (ms / gpu_ms) if (gpu_ms and backend != "gpu") else float("nan")
        sp_str = f"{sp:>7.2f}x" if sp == sp else "       -"
        print(f"{bs:>4}  {backend:<14} {ms:>10.2f} {tau:>8.4f}  {sp_str}")
    print()
PY
