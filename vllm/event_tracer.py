"""event_tracer.py — Light-weight CSV event logger cho server-side profiling.

Mục đích: trace events trong scheduler / predictor / model_executor để phân
tích CPU/GPU overlap, predictor backpressure, scheduler tick blocking.

Usage:
  Bật bằng env: TRACE_EVENTS=1 TRACE_EVENTS_PATH=/path/to/trace.csv
  Off mặc định → zero overhead.

Reference time: t_rel = time.time() - serve_start_time
  serve_start_time set ở Scheduler.add_seq_group() khi nhận request đầu tiên
  → t=0 = "server bắt đầu xử lý workload".

Thread safety:
  - File write trong lock (main thread + ov-predictor worker cùng truy cập)
  - Buffered I/O (8KB) — tradeoff: nếu server crash, vài line cuối có thể mất

Schema CSV:
  t_rel,event,thread,extra_json
    t_rel: giây từ serve_start (6 decimal precision = µs)
    event: tên event (vd "scheduler.tick.start")
    thread: tên thread đang log (MainThread / ov-predictor_0)
    extra_json: JSON dict serialized, vd {"rid":"cmpl-001","prompt_len":94}
"""
import csv
import json
import os
import threading
import time
from typing import Optional

_fh = None
_writer = None
_lock = threading.Lock()
_serve_start: Optional[float] = None
_enabled: bool = False


def init() -> bool:
    """Lazy init từ env. Return True nếu tracer được bật.

    Idempotent — gọi nhiều lần OK (first call init, subsequent no-op).
    Dùng csv.writer để escape `"` trong JSON (extra_json) đúng cách.
    """
    global _fh, _writer, _enabled
    if _fh is not None:
        return _enabled
    if not int(os.environ.get("TRACE_EVENTS", 0)):
        _enabled = False
        return False
    path = os.environ.get("TRACE_EVENTS_PATH")
    if not path:
        _enabled = False
        return False
    # Tạo dir nếu chưa có
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # buffering=8192 cân bằng giữa I/O throughput và flush timeliness
    _fh = open(path, "w", buffering=8192, newline="")
    _writer = csv.writer(_fh, quoting=csv.QUOTE_MINIMAL)
    _writer.writerow(["t_rel", "event", "thread", "extra_json"])
    _enabled = True
    return True


def set_serve_start(t: float) -> None:
    """Set mốc t=0. Gọi 1 lần ở Scheduler.add_seq_group khi
    serve_start_time vừa được set."""
    global _serve_start
    if _serve_start is None:
        _serve_start = t


def is_enabled() -> bool:
    """Check nhanh nếu tracing đang active. Dùng để skip overhead
    construct extra dict khi off."""
    return _enabled and _serve_start is not None


def log(event: str, extra: Optional[dict] = None) -> None:
    """Append 1 event vào CSV. Thread-safe.

    Args:
        event: tên event (vd "scheduler.tick.start")
        extra: dict serializable bằng json (vd {"tick": 42})
    """
    if not _enabled or _serve_start is None or _writer is None:
        return
    t_rel = time.time() - _serve_start
    thread = threading.current_thread().name
    if extra:
        # Compact JSON — không space
        extra_str = json.dumps(extra, separators=(",", ":"))
    else:
        extra_str = "{}"
    with _lock:
        _writer.writerow([f"{t_rel:.6f}", event, thread, extra_str])


def close() -> None:
    """Đóng file. Gọi ở shutdown (nếu cần)."""
    global _fh
    if _fh is not None:
        _fh.flush()
        _fh.close()
        _fh = None
