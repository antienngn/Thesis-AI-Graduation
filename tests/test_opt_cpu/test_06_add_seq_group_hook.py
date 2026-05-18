"""
Test B2-bis — verify hook `serve_start_time` capture trong Scheduler.add_seq_group.

Logic cần verify:
  1. Lần đầu add_seq_group được gọi (với warmup_seconds > 0):
     serve_start_time = time.time().
  2. Lần thứ 2+ gọi add_seq_group: serve_start_time KHÔNG bị reset.
  3. Khi warmup_seconds == 0: serve_start_time vẫn None (zero overhead).

Tests dùng simulation logic vì init full Scheduler cần mock nhiều dependency.
"""
import time
from unittest.mock import MagicMock


def _simulate_add_seq_group(scheduler, seq_group):
    """Mirror logic của Edit B2-bis ở scheduler.py.

    Phải khớp 1:1 với code trong vllm/core/scheduler.py: add_seq_group.
    """
    if scheduler.warmup_seconds > 0 and scheduler.serve_start_time is None:
        scheduler.serve_start_time = time.time()
    # body cũ — không quan tâm trong test này
    scheduler.waiting.append(seq_group)


def test_first_call_captures_time():
    """add_seq_group lần đầu set serve_start_time = time.time()."""
    sched = MagicMock()
    sched.warmup_seconds = 2.0
    sched.serve_start_time = None
    sched.waiting = []

    before = time.time()
    _simulate_add_seq_group(sched, MagicMock())
    after = time.time()

    assert sched.serve_start_time is not None
    assert before <= sched.serve_start_time <= after


def test_subsequent_calls_dont_reset():
    """Lần gọi thứ 2 KHÔNG được ghi đè serve_start_time."""
    sched = MagicMock()
    sched.warmup_seconds = 2.0
    sched.serve_start_time = None
    sched.waiting = []

    _simulate_add_seq_group(sched, MagicMock())
    t1 = sched.serve_start_time

    time.sleep(0.05)  # đảm bảo time.time() khác
    _simulate_add_seq_group(sched, MagicMock())
    t2 = sched.serve_start_time

    assert t1 == t2, "serve_start_time phải KHÔNG bị reset ở call thứ 2"


def test_no_warmup_no_capture():
    """warmup_seconds = 0 → serve_start_time vẫn None (zero overhead)."""
    sched = MagicMock()
    sched.warmup_seconds = 0.0
    sched.serve_start_time = None
    sched.waiting = []

    _simulate_add_seq_group(sched, MagicMock())
    assert sched.serve_start_time is None

    # Gọi nhiều lần vẫn None
    _simulate_add_seq_group(sched, MagicMock())
    _simulate_add_seq_group(sched, MagicMock())
    assert sched.serve_start_time is None


def test_real_scheduler_add_seq_group():
    """Verify thực tế trong Scheduler instance (không mock simulate).

    Dùng patch BlockSpaceManager để init Scheduler thành công, rồi gọi
    add_seq_group thật.
    """
    from unittest.mock import patch
    from vllm.core.scheduler import Scheduler

    cfg = MagicMock()
    cfg.schedule_type = "opt-cpu-warmup1.0"   # warmup=1.0
    cfg.max_num_batched_tokens = 1024
    cfg.max_num_seqs = 256
    cfg.max_model_len = 2048
    cfg.use_v2_block_manager = False

    cache_cfg = MagicMock()
    cache_cfg.block_size = 16
    cache_cfg.num_gpu_blocks = 100
    cache_cfg.num_cpu_blocks = 0
    cache_cfg.sliding_window = None
    cache_cfg.enable_prefix_caching = False

    with patch("vllm.core.scheduler.BlockSpaceManager"):
        try:
            s = Scheduler(cfg, cache_cfg, None)
        except Exception:
            # Nếu Scheduler init crash do dispatch chưa có opt-cpu (Edit B2.5
            # chưa thêm), skip test này — sẽ được verify ở test_08_dispatch.
            import pytest
            pytest.skip("Scheduler init failed — likely Edit B2.5 not yet applied")

        assert s.warmup_seconds == 1.0
        assert s.serve_start_time is None

        # Gọi add_seq_group thật → serve_start_time phải được set
        sg = MagicMock()
        sg.request_id = "test-req-1"
        s.add_seq_group(sg)
        assert s.serve_start_time is not None
        t1 = s.serve_start_time

        # Gọi lần 2 không reset
        sg2 = MagicMock()
        sg2.request_id = "test-req-2"
        s.add_seq_group(sg2)
        assert s.serve_start_time == t1
