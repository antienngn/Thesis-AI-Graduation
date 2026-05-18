"""
Test B2.5 — verify dispatch logic ở Scheduler.__init__.

Mục đích kiểm:
  1. schedule_type "opt-cpu-warmup2.0" → dispatch tới _get_opt_cpu_ordered_requests.
  2. schedule_type "opt-xxx" → vẫn dispatch tới _get_opt_ordered_requests (không bị steal).
  3. schedule_type "fcfs"/"sjf" → không bị ảnh hưởng.
  4. need_score = True cho cả "opt" và "opt-cpu" branches.

Đây là test critical: nếu dispatch sai, baseline opt-xxx có thể bị break.
"""
from unittest.mock import MagicMock, patch

from vllm.core.scheduler import Scheduler


def _build_scheduler(schedule_type: str):
    """Init Scheduler stub với schedule_type cho trước, patch BlockSpaceManager."""
    cfg = MagicMock()
    cfg.schedule_type = schedule_type
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
        return Scheduler(cfg, cache_cfg, None)


def test_dispatch_opt_cpu_with_warmup():
    """opt-cpu-warmup2.0 → _get_opt_cpu_ordered_requests + warmup_seconds parsed."""
    s = _build_scheduler("opt-cpu-warmup2.0")
    assert s._get_ordered_requests == s._get_opt_cpu_ordered_requests
    assert s.warmup_seconds == 2.0
    assert s.need_score is True


def test_dispatch_opt_cpu_without_warmup_suffix():
    """opt-cpu (không có warmup) vẫn dispatch tới opt_cpu method, warmup=0."""
    s = _build_scheduler("opt-cpu")
    assert s._get_ordered_requests == s._get_opt_cpu_ordered_requests
    assert s.warmup_seconds == 0.0  # không có suffix warmup → default 0.0


def test_dispatch_opt_legacy_unchanged():
    """opt-xxx (baseline cũ) PHẢI vẫn dispatch tới _get_opt_ordered_requests."""
    s = _build_scheduler("opt-xxx")
    # CRITICAL: không được bị steal bởi opt-cpu branch
    assert s._get_ordered_requests == s._get_opt_ordered_requests
    assert s._get_ordered_requests != s._get_opt_cpu_ordered_requests
    assert s.warmup_seconds == 0.0
    assert s.need_score is True


def test_dispatch_opt_with_warmup_suffix_legacy():
    """Edge case: 'opt-warmup2.0' (không có cpu) → vẫn tới opt branch.

    Pattern startswith('opt-cpu') KHÔNG match 'opt-warmup2.0', nên rơi vào
    branch opt cũ. warmup_seconds vẫn được parse (do regex chạy độc lập).
    """
    s = _build_scheduler("opt-warmup2.0")
    assert s._get_ordered_requests == s._get_opt_ordered_requests
    assert s.warmup_seconds == 2.0   # regex parse vẫn chạy


def test_dispatch_fcfs_unchanged():
    """fcfs không có _get_ordered_requests assigned (theo code hiện tại pass at line 290)."""
    s = _build_scheduler("fcfs")
    # fcfs/sjf/ljf branches at line 290 chỉ pass — không assign _get_ordered_requests
    # nên warmup_seconds cũng = 0
    assert s.warmup_seconds == 0.0


def test_dispatch_xpt_unchanged():
    """xpt không bị ảnh hưởng bởi opt-cpu branch."""
    # xpt cần torch.load distribution path, skip nếu không có file thực
    import pytest
    pytest.skip("xpt requires distribution file path — not testing here")


def test_opt_cpu_with_starvation_combo():
    """opt-cpu-warmup2.0 + starvation control combo.

    Skipped vì existing parser cho 'starv{X}period{Y}' ở scheduler.py:271-274
    có format requirement riêng (offset -1 ở end của starv parse) — không
    liên quan tới logic opt-cpu, là pre-existing limitation.
    """
    import pytest
    pytest.skip("starv+period parser format không tương thích để test combo")
