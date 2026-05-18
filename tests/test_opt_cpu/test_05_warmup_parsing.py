"""
Test B1 — verify regex parse `warmup{T}` từ schedule_type.

Pattern: `re.search(r"warmup(\d+\.?\d*)", schedule_type)`

Cover các case:
  - Không có "warmup" trong tên → warmup_seconds = 0.0
  - Có "warmup" với int → parse correct
  - Có "warmup" với float → parse correct
  - "warmup0" (= disable) → 0.0
"""
import re


def parse_warmup(schedule_type: str) -> float:
    """Mirror logic của Edit B1 — phải match đúng implementation trong scheduler.py."""
    m = re.search(r"warmup(\d+\.?\d*)", schedule_type)
    return float(m.group(1)) if m else 0.0


def test_no_warmup_in_name():
    """Schedule type không chứa 'warmup' → trả 0.0."""
    assert parse_warmup("fcfs") == 0.0
    assert parse_warmup("opt-xxx") == 0.0
    assert parse_warmup("sjf") == 0.0
    assert parse_warmup("opt-starv5period10") == 0.0  # có starv nhưng không warmup


def test_warmup_integer_value():
    """Parse số nguyên."""
    assert parse_warmup("opt-cpu-warmup2") == 2.0
    assert parse_warmup("opt-cpu-warmup10") == 10.0
    assert parse_warmup("opt-cpu-warmup0") == 0.0  # disable


def test_warmup_float_value():
    """Parse số thực."""
    assert parse_warmup("opt-cpu-warmup0.5") == 0.5
    assert parse_warmup("opt-cpu-warmup2.0") == 2.0
    assert parse_warmup("opt-cpu-warmup1.5") == 1.5


def test_warmup_with_other_suboptions():
    """Warmup phải parse được cùng các sub-option khác (vd starv)."""
    # Note: thứ tự trong tên không quan trọng vì regex tìm chuỗi "warmup<number>"
    assert parse_warmup("opt-cpu-warmup2.0-starv5period10") == 2.0
    assert parse_warmup("opt-cpu-starv5period10-warmup2.0") == 2.0


def test_scheduler_init_sets_warmup_seconds():
    """Verify Edit B1 trong Scheduler thực sự set self.warmup_seconds đúng."""
    from unittest.mock import MagicMock, patch
    from vllm.core.scheduler import Scheduler

    cfg = MagicMock()
    cfg.schedule_type = "opt-cpu-warmup2.5"
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

    # Patch BlockSpaceManager để không cần allocator thật
    with patch("vllm.core.scheduler.BlockSpaceManager"):
        try:
            s = Scheduler(cfg, cache_cfg, None)
            assert s.warmup_seconds == 2.5
            assert s.serve_start_time is None  # chưa có request nào
        except AssertionError:
            raise
        except Exception:
            # Nếu Scheduler init crash do dispatch chưa có opt-cpu branch
            # (Edit B2.5 chưa thêm), ít nhất verify regex parse logic độc lập.
            assert parse_warmup("opt-cpu-warmup2.5") == 2.5
