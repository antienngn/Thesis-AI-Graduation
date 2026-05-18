#!/usr/bin/env python3
"""parse_predictor_latency.py — Parse server logs trong PRE_LAT_E2E/,
extract predictor timing, output CSV cho builder.

Filename convention: server_<sched>_<predictor>_r<rate>.log
  - sched:     async-merged | opt-xxx
  - predictor: pythia | opt
  - rate:      số nguyên (2,4,8,16,32,64)

Regex (theo backend):
  - opt-xxx + AUXLLM (GPU):     "OPT-TIME: n=<int> t=<float>s"
  - opt-xxx + OpenVINO (CPU):   "OV-PRED-TIME (sync): n=<int> t=<float>s"
  - async-merged + OpenVINO:    "OV-STREAM-TIME: n=<int> t=<float>s"

Parser scan TẤT CẢ 3 regex trên mỗi file opt-xxx (auto-detect backend); với
file async-merged chỉ scan OV-STREAM-TIME. Bỏ qua các log không match.

Output: PRE_LAT_E2E/predictor_latency_raw.csv (1 row = 1 forward call)
Cột: scheduler, predictor, rate, n_per_chunk, latency_ms, per_req_ms, source
"""

import csv
import re
from glob import glob
from pathlib import Path

import os as _os

ROOT = Path(__file__).parent
# Override qua env LOG_DIR để parse cho thư mục khác (vd TEMP_RES_ASYNC_BURST):
#   LOG_DIR=TEMP_RES_ASYNC_BURST python parse_predictor_latency.py
LOG_DIR = ROOT / _os.environ.get("LOG_DIR", "PRE_LAT_E2E")
OUT_CSV = LOG_DIR / "predictor_latency_raw.csv"

PAT_STREAM = re.compile(r"OV-STREAM-TIME: n=(\d+) t=([\d.]+)s")
PAT_OV_SYNC = re.compile(r"OV-PRED-TIME \(sync\): n=(\d+) t=([\d.]+)s")
PAT_OPT = re.compile(r"OPT-TIME: n=(\d+) t=([\d.]+)s")
PAT_FN = re.compile(r"server_(.+)_(opt|pythia)_r(\d+)\.log$")


def main():
    log_files = sorted(glob(str(LOG_DIR / "server_*_r*.log")))
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return

    rows = []
    summary = {}  # (sched, predictor, rate, source) -> n_calls

    for log_path in log_files:
        fn = Path(log_path).name
        m = PAT_FN.match(fn)
        if not m:
            print(f"SKIP (filename không match schema): {fn}")
            continue
        sched, predictor, rate = m.group(1), m.group(2), int(m.group(3))

        # opt-xxx: thử cả 3 regex (auto-detect AUXLLM vs OpenVINO);
        # async-merged/warmup: chỉ stream regex.
        if sched == "opt-xxx":
            patterns = [
                (PAT_OPT, "OPT-TIME (AUXLLM)"),
                (PAT_OV_SYNC, "OV-PRED-TIME (OpenVINO sync)"),
            ]
        else:
            patterns = [(PAT_STREAM, "OV-STREAM-TIME (OpenVINO streaming)")]

        with open(log_path) as f:
            content = f.read()

        per_source_count = {}
        for pat, source in patterns:
            for mm in pat.finditer(content):
                n, t = int(mm.group(1)), float(mm.group(2))
                if n == 0:
                    continue
                rows.append({
                    "scheduler": sched,
                    "predictor": predictor,
                    "rate": rate,
                    "n_per_chunk": n,
                    "latency_ms": t * 1000.0,
                    "per_req_ms": t * 1000.0 / n,
                    "source": source,
                })
                per_source_count[source] = per_source_count.get(source, 0) + 1

        n_match = sum(per_source_count.values())
        summary[(sched, predictor, rate)] = per_source_count
        sources_str = ", ".join(f"{k}={v}" for k, v in per_source_count.items()) or "0"
        print(f"  {fn}: {n_match} calls  [{sources_str}]")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scheduler", "predictor", "rate",
            "n_per_chunk", "latency_ms", "per_req_ms", "source",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV} ({len(rows)} total forward calls)")
    print(f"\nSummary by (scheduler, predictor, rate):")
    for k in sorted(summary.keys()):
        sources = summary[k]
        sources_str = ", ".join(f"{src}={n}" for src, n in sources.items())
        print(f"  {k}: {sources_str}")


if __name__ == "__main__":
    main()
