"""
Test B2/D2 — verify _get_opt_cpu_ordered_requests:
  Warmup phase:
    - elapsed < warmup_seconds: trả FCFS theo arrival_time, KHÔNG gọi predictor.
    - serve_start_time=None: vẫn FCFS (elapsed=0).
  Ranking phase (D2 — async):
    - poll_results() được gọi mỗi tick (non-blocking).
    - Nếu need_aux_scores và predictor không busy: submit_async được gọi.
    - Nếu predictor busy: submit_async KHÔNG được gọi (skip).
    - Sort dùng _safe_neg_score (None → 0.0) — không crash với running unscored.

Test gọi method trực tiếp với mock Scheduler để không cần init full vLLM.
"""
import time
from collections import deque
from unittest.mock import MagicMock

from vllm.core.scheduler import Scheduler


def _make_seq_group(arrival_time: float, score=None, request_id: str = "x"):
    """Tạo SequenceGroup mock với fields scheduler cần."""
    sg = MagicMock()
    sg.metrics = MagicMock(arrival_time=arrival_time)
    sg.aux_model_score = score
    # need_aux_model_score() returns True khi score is None (xem sequence.py:461)
    sg.need_aux_model_score = MagicMock(return_value=score is None)
    sg.request_id = request_id
    return sg


def _make_scheduler_stub(warmup_seconds, serve_start_time, waiting,
                         running=None, swapped=None, predictor_busy=False):
    """Tạo Scheduler stub với just enough state để gọi _get_opt_cpu_ordered_requests.

    Args:
        predictor_busy: nếu True → mock aux_model.is_busy() trả True (predictor
            đang xử lý batch trước, scheduler nên skip submit mới).
    """
    s = MagicMock()
    s.warmup_seconds = warmup_seconds
    s.serve_start_time = serve_start_time
    s.starv = -1                              # disable starvation logic
    s.aux_model = MagicMock()
    # Mock async API (D1) — mặc định không busy, poll/submit return 0/True
    s.aux_model.is_busy = MagicMock(return_value=predictor_busy)
    s.aux_model.poll_results = MagicMock(return_value=0)
    s.aux_model.submit_async = MagicMock(return_value=True)
    s.waiting = deque(waiting or [])
    s.running = deque(running or [])
    s.swapped = deque(swapped or [])
    return s


# =============================================================================
# Warmup phase tests
# =============================================================================

def test_warmup_branch_returns_fcfs_order():
    """Trong warmup window, sort theo arrival_time, KHÔNG gọi predictor."""
    s = _make_scheduler_stub(
        warmup_seconds=2.0,
        serve_start_time=time.time() - 0.5,   # elapsed = 0.5 < 2.0
        waiting=[
            _make_seq_group(arrival_time=2.0, request_id="req-late"),
            _make_seq_group(arrival_time=1.0, request_id="req-early"),
        ],
    )

    ordered = Scheduler._get_opt_cpu_ordered_requests(s)

    # Earliest arrival first
    assert ordered[0].metrics.arrival_time == 1.0
    assert ordered[1].metrics.arrival_time == 2.0

    # KHÔNG gọi predictor trong warmup phase (cả sync và async API)
    s.aux_model.poll_results.assert_not_called()
    s.aux_model.submit_async.assert_not_called()


def test_serve_start_time_none_treated_as_in_warmup():
    """serve_start_time = None (chưa có request đầu) → coi như đang trong warmup → FCFS."""
    s = _make_scheduler_stub(
        warmup_seconds=2.0,
        serve_start_time=None,                 # CHƯA có request đầu
        waiting=[_make_seq_group(arrival_time=1.0)],
    )

    ordered = Scheduler._get_opt_cpu_ordered_requests(s)
    assert len(ordered) == 1
    s.aux_model.submit_async.assert_not_called()


def test_warmup_boundary_in_window_uses_fcfs():
    """Trong warmup window (elapsed < warmup_seconds): FCFS branch."""
    s = _make_scheduler_stub(
        warmup_seconds=10.0,                  # margin rộng để tránh timing flakiness
        serve_start_time=time.time() - 1.0,   # elapsed ≈ 1.0 < 10.0
        waiting=[_make_seq_group(arrival_time=1.0)],
    )
    Scheduler._get_opt_cpu_ordered_requests(s)
    s.aux_model.submit_async.assert_not_called()


# =============================================================================
# Ranking phase tests — async predictor pattern (D2)
# =============================================================================

def test_post_warmup_polls_and_submits_when_idle():
    """Sau warmup, predictor không busy → poll_results + submit_async."""
    sg_unscored = _make_seq_group(arrival_time=1.0, score=None, request_id="r1")

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,    # past warmup
        waiting=[sg_unscored],
        predictor_busy=False,
    )

    Scheduler._get_opt_cpu_ordered_requests(s)

    # poll_results phải được gọi mỗi tick
    s.aux_model.poll_results.assert_called_once()
    # submit_async phải được gọi với batch chưa scored
    s.aux_model.submit_async.assert_called_once()
    submitted_batch = s.aux_model.submit_async.call_args[0][0]
    assert sg_unscored in submitted_batch


def test_post_warmup_skips_submit_when_predictor_busy():
    """Predictor đang busy → poll vẫn gọi nhưng submit_async KHÔNG gọi.

    Đây là cốt lõi của async pattern: scheduler tick không block đợi
    predictor — nếu predictor đang chạy batch trước, skip tick này.
    """
    sg_unscored = _make_seq_group(arrival_time=1.0, score=None, request_id="r1")

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,
        waiting=[sg_unscored],
        predictor_busy=True,                   # ← predictor đang xử lý batch trước
    )

    Scheduler._get_opt_cpu_ordered_requests(s)

    # poll vẫn gọi (rẻ, ~µs)
    s.aux_model.poll_results.assert_called_once()
    # submit KHÔNG gọi vì predictor busy
    s.aux_model.submit_async.assert_not_called()


def test_post_warmup_no_submit_when_no_unscored_in_waiting():
    """Tất cả waiting đã có score → submit_async KHÔNG được gọi."""
    sg_scored1 = _make_seq_group(arrival_time=1.0, score=0.3, request_id="r1")
    sg_scored2 = _make_seq_group(arrival_time=2.0, score=0.8, request_id="r2")
    sg_scored1.need_aux_model_score = MagicMock(return_value=False)
    sg_scored2.need_aux_model_score = MagicMock(return_value=False)

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,
        waiting=[sg_scored1, sg_scored2],
    )

    ordered = Scheduler._get_opt_cpu_ordered_requests(s)

    # poll vẫn gọi
    s.aux_model.poll_results.assert_called_once()
    # submit KHÔNG gọi vì không có request cần score
    s.aux_model.submit_async.assert_not_called()
    # Sort theo -score (cao nhất trước)
    assert ordered[0].aux_model_score == 0.8
    assert ordered[1].aux_model_score == 0.3


def test_post_warmup_only_submits_unscored_from_waiting():
    """submit_async chỉ chứa các request CHƯA scored (không gồm scored ones)."""
    sg_scored = _make_seq_group(arrival_time=1.0, score=0.5, request_id="scored")
    sg_unscored = _make_seq_group(arrival_time=2.0, score=None, request_id="unscored")
    sg_scored.need_aux_model_score = MagicMock(return_value=False)

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,
        waiting=[sg_scored, sg_unscored],
    )

    Scheduler._get_opt_cpu_ordered_requests(s)

    s.aux_model.submit_async.assert_called_once()
    submitted = s.aux_model.submit_async.call_args[0][0]
    assert sg_unscored in submitted
    assert sg_scored not in submitted


def test_post_warmup_does_not_submit_running_or_swapped():
    """submit_async CHỈ chứa waiting, KHÔNG chứa running/swapped.

    Lý do: running đã đang serve, score chúng vô ích; submit batch lớn
    sẽ làm predictor busy lâu hơn → các tick sau không submit được.
    """
    sg_w = _make_seq_group(arrival_time=2.0, score=None, request_id="waiting")
    sg_r = _make_seq_group(arrival_time=0.5, score=None, request_id="running")
    sg_s = _make_seq_group(arrival_time=0.7, score=None, request_id="swapped")

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,
        waiting=[sg_w],
        running=[sg_r],
        swapped=[sg_s],
    )

    Scheduler._get_opt_cpu_ordered_requests(s)

    s.aux_model.submit_async.assert_called_once()
    submitted = s.aux_model.submit_async.call_args[0][0]
    assert sg_w in submitted
    assert sg_r not in submitted
    assert sg_s not in submitted


def test_sort_with_none_safe_score():
    """Sort key None-safe: running unscored treated như 0.0, không crash với -None."""
    sg_high = _make_seq_group(arrival_time=2.0, score=0.8, request_id="high")
    sg_low = _make_seq_group(arrival_time=3.0, score=0.2, request_id="low")
    sg_none = _make_seq_group(arrival_time=0.5, score=None, request_id="none")
    for sg in (sg_high, sg_low):
        sg.need_aux_model_score = MagicMock(return_value=False)

    s = _make_scheduler_stub(
        warmup_seconds=1.0,
        serve_start_time=time.time() - 5.0,
        waiting=[sg_high, sg_low],
        running=[sg_none],                      # unscored → key=0.0
    )

    # Phải không crash với None
    result = Scheduler._get_opt_cpu_ordered_requests(s)
    assert len(result) == 3

    # Order: -score = -0.8, -0.2, 0.0 (None) → sg_high < sg_low < sg_none
    request_ids = [sg.request_id for sg in result]
    assert request_ids == ["high", "low", "none"]


def test_warmup_zero_skips_warmup_branch():
    """warmup_seconds = 0 → ranking ngay từ request đầu (không có FCFS phase)."""
    sg = _make_seq_group(arrival_time=1.0, score=0.5)
    sg.need_aux_model_score = MagicMock(return_value=False)

    s = _make_scheduler_stub(
        warmup_seconds=0.0,
        serve_start_time=time.time(),    # elapsed ~= 0, nhưng warmup=0 nên skip
        waiting=[sg],
    )

    Scheduler._get_opt_cpu_ordered_requests(s)
    # Vào ranking branch — poll_results được gọi
    s.aux_model.poll_results.assert_called_once()


def test_warmup_boundary_past_window_uses_ranking():
    """Sau warmup window: ranking branch (poll được gọi)."""
    s = _make_scheduler_stub(
        warmup_seconds=0.1,                    # warmup cực ngắn
        serve_start_time=time.time() - 5.0,    # elapsed ≈ 5.0 >> 0.1
        waiting=[],                            # empty
    )
    result = Scheduler._get_opt_cpu_ordered_requests(s)
    assert result == []
    # Ngay cả khi waiting empty, vẫn vào ranking branch và gọi poll_results
    s.aux_model.poll_results.assert_called_once()
