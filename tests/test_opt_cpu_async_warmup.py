"""Test cho opt-cpu-async-warmup scheduler + streaming predictor API.

Coverage:
  Phần 1 — OpenVINOPredictor streaming API:
    - submit_streaming dedup
    - Queue grow + worker spawn
    - Worker apply scores incremental
    - has_pending_stream tracking
    - Worker restart sau khi queue drain
    - Cleanup khi shutdown
    - State separation: legacy submit_async không bị ảnh hưởng

  Phần 2 — Scheduler 3-phase logic:
    - Stage 1: arrival_time sort, không gọi predictor
    - Stage 2: chỉ pick warmup-era, post-warmup BLOCKED
    - Stage 3: score sort + gate unscored
    - Stage 2 → 3 transition tự động
    - Predictor eager trong Stage 2
    - Assert invariant Stage 3 catch bug

Test dùng MOCK predictor để không cần load OPT-125M model thực sự (nhanh +
deterministic). Cũng có 1 smoke test với real predictor (skip nếu không có
model file).
"""
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add vllm-ltr root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Helpers: mock SequenceGroup, mock predictor
# =============================================================================

class _FakeSequence:
    """Fake Sequence tối thiểu để mock SequenceGroup.seqs_dict[0].prompt."""
    def __init__(self, prompt: str):
        self.prompt = prompt


class _FakeMetrics:
    """Fake metrics để mock req.metrics.arrival_time."""
    def __init__(self, arrival_time: float):
        self.arrival_time = arrival_time


class FakeSeqGroup:
    """Mock SequenceGroup for testing.

    Implement đủ interface mà predictor + scheduler cần:
      - request_id
      - seqs_dict (cho predictor lấy prompt)
      - aux_model_score (read/write)
      - need_aux_model_score()
      - set_aux_model_score(s)
      - metrics.arrival_time
      - idle, runs, pri (cho starvation logic)
    """
    def __init__(self, request_id: str, prompt: str, arrival_time: float):
        self.request_id = request_id
        self.seqs_dict = {0: _FakeSequence(prompt)}
        self.aux_model_score = None
        self.metrics = _FakeMetrics(arrival_time)
        # For starvation logic
        self.idle = 0
        self.runs = 0
        self.pri = 0

    def need_aux_model_score(self) -> bool:
        return self.aux_model_score is None

    def set_aux_model_score(self, score) -> None:
        self.aux_model_score = score


# =============================================================================
# Phần 1 — OpenVINOPredictor streaming API tests
# =============================================================================
# Chạy với mock compiled_model + tokenizer để không cần OPT-125M weights.

class _MockTokenizerOutput:
    """Mock HF tokenizer output (BatchEncoding-like)."""
    def __init__(self, batch_size: int, seq_len: int):
        import numpy as np
        # Tạo numpy arrays giả lập input_ids + attention_mask
        self._ids = np.zeros((batch_size, seq_len), dtype=np.int64)
        self._mask = np.ones((batch_size, seq_len), dtype=np.int64)

    def __getitem__(self, key):
        # Return torch-like object có .numpy() method
        class _T:
            def __init__(self, arr):
                self.arr = arr
            def numpy(self):
                return self.arr
        if key == "input_ids":
            return _T(self._ids)
        elif key == "attention_mask":
            return _T(self._mask)
        raise KeyError(key)


def _build_mock_predictor(score_value=0.5, latency_per_chunk=0.05):
    """Build OpenVINOPredictor với mocks thay cho compiled_model + tokenizer.

    Args:
        score_value: giá trị score giả lập trả ra cho mỗi prompt.
        latency_per_chunk: simulated compute latency per mini-batch (giây).

    Returns:
        OpenVINOPredictor instance đã mocked, sẵn sàng submit_streaming.
    """
    from vllm.model_executor.openvino_predictor import OpenVINOPredictor

    # Bypass __init__ để không load model thật
    p = OpenVINOPredictor.__new__(OpenVINOPredictor)
    p.max_length = 2048
    p.max_batch_size = 20

    # Mock tokenizer (callable trả MockTokenizerOutput)
    def mock_tokenizer(prompts, **kwargs):
        return _MockTokenizerOutput(len(prompts), 16)
    p.tokenizer = mock_tokenizer

    # Mock compiled_model: trả logits có shape [batch, 1] với value=score_value
    def mock_compiled_model(inputs):
        import numpy as np
        batch = inputs[0].shape[0]
        time.sleep(latency_per_chunk)  # simulate compute time
        logits = np.full((batch, 1), score_value, dtype=np.float32)
        return {"logits": logits}
    p.compiled_model = mock_compiled_model

    # Init streaming + legacy state (giả lập init flow)
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    p.async_mode = True
    p._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ov-test")
    p._pending_future = None
    p._pending_seq_groups = None
    p._stream_queue = deque()
    p._stream_in_flight = set()
    p._stream_lock = threading.Lock()
    p._stream_worker_active = False

    return p


# =============================================================================
# Test cases — Phần 1: Streaming predictor API
# =============================================================================

def test_submit_streaming_empty_returns_false():
    """submit_streaming([]) phải return False, không crash."""
    p = _build_mock_predictor()
    try:
        assert p.submit_streaming([]) is False
        assert not p.has_pending_stream()
    finally:
        p.shutdown()


def test_submit_streaming_dedup():
    """Submit cùng 1 request 2 lần — chỉ thêm vào queue 1 lần."""
    p = _build_mock_predictor(latency_per_chunk=0.5)  # latency dài để dedup test
    try:
        sg = FakeSeqGroup("req1", "prompt 1", 0.0)

        # Lần đầu submit — should add
        assert p.submit_streaming([sg]) is True
        assert sg.request_id in p._stream_in_flight

        # Lần 2 submit cùng request — should dedup (return False vì 0 new added)
        assert p.submit_streaming([sg]) is False

        # Wait worker xong
        time.sleep(1.0)
        assert sg.aux_model_score is not None  # đã được score
    finally:
        p.shutdown()


def test_submit_streaming_apply_scores():
    """Submit 5 requests, verify tất cả đều được apply score (incremental)."""
    p = _build_mock_predictor(score_value=0.7, latency_per_chunk=0.02)
    try:
        sgs = [FakeSeqGroup(f"req{i}", f"prompt {i}", float(i))
               for i in range(5)]
        assert p.submit_streaming(sgs) is True

        # Wait worker
        for _ in range(50):
            time.sleep(0.05)
            if all(s.aux_model_score is not None for s in sgs):
                break

        for sg in sgs:
            assert sg.aux_model_score == pytest.approx(0.7), \
                f"sg {sg.request_id} score={sg.aux_model_score}"
        assert not p.has_pending_stream()
    finally:
        p.shutdown()


def test_submit_streaming_large_batch_chunks():
    """Submit 50 requests với max_batch_size=20 → 3 mini-batches.

    Verify:
    - Worker chia thành 3 chunks (20+20+10)
    - Score apply incremental (mini-batch xong → set ngay)
    - Tất cả requests cuối cùng đều có score
    """
    p = _build_mock_predictor(score_value=1.5, latency_per_chunk=0.05)
    try:
        sgs = [FakeSeqGroup(f"req{i}", f"prompt {i}", float(i))
               for i in range(50)]
        assert p.submit_streaming(sgs) is True

        # Capture incremental progress
        scored_over_time = []
        for _ in range(40):
            time.sleep(0.025)
            scored_count = sum(1 for s in sgs if s.aux_model_score is not None)
            scored_over_time.append(scored_count)
            if scored_count == 50:
                break

        # Verify final state: all scored
        assert all(s.aux_model_score == pytest.approx(1.5) for s in sgs)

        # Verify incremental: phải có ít nhất 1 sample THỜI ĐIỂM mà
        # 0 < scored < 50 (chứng tỏ apply incremental, không atomic cuối)
        partial_progress = [c for c in scored_over_time if 0 < c < 50]
        assert len(partial_progress) > 0, \
            f"Expected incremental progress, got: {scored_over_time}"
    finally:
        p.shutdown()


def test_submit_streaming_concurrent_during_worker():
    """Submit batch mới TRONG khi worker đang chạy batch trước.

    Verify:
    - submit_streaming không block
    - Request mới được add vào queue
    - Worker pick up request mới sau khi xong batch hiện tại (continuous)
    """
    p = _build_mock_predictor(latency_per_chunk=0.1)
    try:
        # Batch 1: 30 requests
        batch1 = [FakeSeqGroup(f"a{i}", f"p {i}", 0.0) for i in range(30)]
        assert p.submit_streaming(batch1) is True

        # Đợi worker bắt đầu mini-batch đầu (30ms < latency_per_chunk=100ms)
        time.sleep(0.03)

        # Submit batch 2 — worker đang busy nhưng vẫn accept
        batch2 = [FakeSeqGroup(f"b{i}", f"p {i}", 1.0) for i in range(10)]
        t0 = time.time()
        assert p.submit_streaming(batch2) is True
        # submit_streaming phải nhanh (không block)
        assert time.time() - t0 < 0.05, "submit_streaming blocked"

        # Wait final apply
        for _ in range(30):
            time.sleep(0.05)
            scored = sum(1 for s in batch1 + batch2
                         if s.aux_model_score is not None)
            if scored == 40:
                break

        assert all(s.aux_model_score is not None for s in batch1)
        assert all(s.aux_model_score is not None for s in batch2)
    finally:
        p.shutdown()


def test_has_pending_stream_lifecycle():
    """has_pending_stream tracks state đúng từ submit → worker exit."""
    p = _build_mock_predictor(latency_per_chunk=0.1)
    try:
        # Initial: no work
        assert p.has_pending_stream() is False

        # After submit: pending = True
        sgs = [FakeSeqGroup(f"r{i}", f"p {i}", 0.0) for i in range(5)]
        p.submit_streaming(sgs)
        assert p.has_pending_stream() is True

        # Wait worker xong
        for _ in range(20):
            time.sleep(0.05)
            if not p.has_pending_stream():
                break

        # After worker xong: pending = False
        assert p.has_pending_stream() is False
        assert all(s.aux_model_score is not None for s in sgs)
    finally:
        p.shutdown()


def test_worker_restart_after_drain():
    """Worker exit khi queue empty, submit mới → spawn worker lại."""
    p = _build_mock_predictor(latency_per_chunk=0.02)
    try:
        # Submit batch 1, đợi xong
        batch1 = [FakeSeqGroup(f"a{i}", f"p {i}", 0.0) for i in range(3)]
        p.submit_streaming(batch1)
        for _ in range(20):
            time.sleep(0.05)
            if not p.has_pending_stream():
                break
        assert all(s.aux_model_score is not None for s in batch1)

        # Worker đã exit
        with p._stream_lock:
            assert p._stream_worker_active is False

        # Submit batch 2 — worker phải spawn lại
        batch2 = [FakeSeqGroup(f"b{i}", f"p {i}", 1.0) for i in range(3)]
        p.submit_streaming(batch2)
        for _ in range(20):
            time.sleep(0.05)
            if not p.has_pending_stream():
                break
        assert all(s.aux_model_score is not None for s in batch2)
    finally:
        p.shutdown()


def test_poll_streaming_is_noop():
    """poll_streaming luôn return 0 (no-op)."""
    p = _build_mock_predictor()
    try:
        assert p.poll_streaming() == 0
        sgs = [FakeSeqGroup("r1", "p", 0.0)]
        p.submit_streaming(sgs)
        time.sleep(0.2)  # wait worker
        # Sau khi worker apply, poll vẫn return 0
        assert p.poll_streaming() == 0
    finally:
        p.shutdown()


def test_legacy_state_unchanged_by_streaming():
    """Streaming API KHÔNG đụng vào _pending_future / _pending_seq_groups.

    Verify state isolation: submit_streaming không set _pending_future.
    """
    p = _build_mock_predictor(latency_per_chunk=0.1)
    try:
        # Snapshot legacy state
        assert p._pending_future is None
        assert p._pending_seq_groups is None

        # Submit qua streaming
        sgs = [FakeSeqGroup(f"r{i}", "p", 0.0) for i in range(3)]
        p.submit_streaming(sgs)

        # Legacy state vẫn untouched
        assert p._pending_future is None
        assert p._pending_seq_groups is None

        # Wait + verify scores applied (qua streaming path)
        for _ in range(20):
            time.sleep(0.05)
            if not p.has_pending_stream():
                break
        assert all(s.aux_model_score is not None for s in sgs)

        # Legacy state STILL untouched sau khi worker xong
        assert p._pending_future is None
        assert p._pending_seq_groups is None
    finally:
        p.shutdown()


def test_shutdown_clears_streaming_state():
    """shutdown() phải clear streaming queue + in_flight + worker_active."""
    p = _build_mock_predictor(latency_per_chunk=0.5)  # long latency
    sgs = [FakeSeqGroup(f"r{i}", "p", 0.0) for i in range(50)]
    p.submit_streaming(sgs)

    # Có pending state
    assert p.has_pending_stream() is True

    # Shutdown
    p.shutdown()

    # State cleared
    with p._stream_lock:
        assert len(p._stream_queue) == 0
        assert len(p._stream_in_flight) == 0
        assert p._stream_worker_active is False


# =============================================================================
# Phần 2 — Scheduler 3-phase logic tests
# =============================================================================
# Mock predictor + scheduler internals để test logic phân stage không cần
# load model hay full vLLM engine.

class _MockAuxModel:
    """Mock aux_model implement streaming API tối thiểu cho scheduler test.

    Track calls để verify scheduler gọi đúng API ở Stage nào.
    """
    def __init__(self):
        self.submit_streaming_calls = []  # list[list[seq_groups]]
        self.poll_streaming_count = 0

    def submit_streaming(self, seq_groups):
        self.submit_streaming_calls.append(list(seq_groups))
        # Simulate immediate scoring để test Stage 3 invariant
        # (trong test thật, worker thread sẽ apply async)
        return True

    def poll_streaming(self):
        self.poll_streaming_count += 1
        return 0

    # Legacy methods (không nên gọi từ async scheduler)
    def submit_async(self, sgs):
        raise AssertionError("Async scheduler should not call submit_async")

    def poll_results(self):
        raise AssertionError("Async scheduler should not call poll_results")

    def is_busy(self):
        raise AssertionError("Async scheduler should not call is_busy")


class _MockScheduler:
    """Wrapper minimal để gọi _get_opt_cpu_async_warmup_ordered_requests
    không cần init full Scheduler (cần block_manager, executor, etc.)."""

    def __init__(self, warmup_seconds=2.0, starv=-1):
        # Required attributes for the method
        self.warmup_seconds = warmup_seconds
        self.serve_start_time = None
        self.starv = starv
        self.period = 100
        self.waiting = []
        self.running = []
        self.swapped = []
        self.aux_model = _MockAuxModel()


def _import_scheduler_method():
    """Import method từ Scheduler class (không init Scheduler thật)."""
    from vllm.core.scheduler import Scheduler
    return Scheduler._get_opt_cpu_async_warmup_ordered_requests


def test_stage1_warmup_fcfs_no_predictor():
    """Stage 1: t < T → sort by arrival_time, KHÔNG gọi predictor."""
    method = _import_scheduler_method()

    sched = _MockScheduler(warmup_seconds=10.0)  # warmup dài để chắc đang Stage 1
    sched.serve_start_time = time.time() - 1.0  # elapsed = 1s < 10s

    # 3 requests với arrival_time khác nhau
    r1 = FakeSeqGroup("a", "p", 0.0)
    r2 = FakeSeqGroup("b", "p", 1.0)
    r3 = FakeSeqGroup("c", "p", 0.5)
    sched.waiting = [r1, r2, r3]

    result = method(sched)

    # Sort theo arrival_time: r1 (0.0), r3 (0.5), r2 (1.0)
    assert [r.request_id for r in result] == ["a", "c", "b"]
    # Predictor KHÔNG được gọi ở Stage 1
    assert sched.aux_model.submit_streaming_calls == []
    assert sched.aux_model.poll_streaming_count == 0


def test_stage2_drain_picks_only_warmup_era():
    """Stage 2: t >= T, còn warmup-era → schedulable chỉ chứa warmup-era.

    Post-warmup waiting BLOCKED (kể cả đã có score).
    """
    method = _import_scheduler_method()

    serve_start = time.time() - 5.0  # elapsed = 5s
    sched = _MockScheduler(warmup_seconds=2.0)
    sched.serve_start_time = serve_start
    warmup_end = serve_start + 2.0

    # warmup-era waiting (arrival < warmup_end), unscored
    we_w1 = FakeSeqGroup("we1", "p", warmup_end - 0.5)
    we_w2 = FakeSeqGroup("we2", "p", warmup_end - 0.1)
    # warmup-era running (đã pick up từ Stage 1, vẫn ở running)
    we_r1 = FakeSeqGroup("wer1", "p", warmup_end - 1.0)
    # post-warmup waiting (arrival > warmup_end), không score
    pw_unscored = FakeSeqGroup("pw1", "p", warmup_end + 1.0)
    # post-warmup waiting đã có score
    pw_scored = FakeSeqGroup("pw2", "p", warmup_end + 2.0)
    pw_scored.aux_model_score = -2.5

    sched.waiting = [we_w1, we_w2, pw_unscored, pw_scored]
    sched.running = [we_r1]
    sched.swapped = []

    result = method(sched)

    # Schedulable phải chỉ chứa warmup-era (we_w1, we_w2, we_r1)
    # Post-warmup pw_unscored + pw_scored đều BLOCKED (kể cả pw_scored
    # đã có score thật)
    result_ids = [r.request_id for r in result]
    assert set(result_ids) == {"we1", "we2", "wer1"}
    assert "pw1" not in result_ids, "Unscored post-warmup leak vào Stage 2"
    assert "pw2" not in result_ids, "Scored post-warmup phải BLOCK Stage 2"

    # Sort theo arrival_time: wer1 (-1.0) < we1 (-0.5) < we2 (-0.1)
    expected_order = ["wer1", "we1", "we2"]
    assert result_ids == expected_order

    # Predictor được gọi (eager scoring post-warmup unscored = pw_unscored)
    assert len(sched.aux_model.submit_streaming_calls) == 1
    submitted_ids = [s.request_id
                     for s in sched.aux_model.submit_streaming_calls[0]]
    assert submitted_ids == ["pw1"]
    # pw_scored đã có score → không submit lại


def test_stage3_post_drain_score_sort_with_gate():
    """Stage 3: warmup-era hết, post-warmup sort by score, gate unscored."""
    method = _import_scheduler_method()

    serve_start = time.time() - 30.0  # elapsed = 30s
    sched = _MockScheduler(warmup_seconds=2.0)
    sched.serve_start_time = serve_start
    warmup_end = serve_start + 2.0

    # KHÔNG có warmup-era nào → Stage 3
    # Post-warmup running (đã scored từ Stage 3 trước)
    pw_r1 = FakeSeqGroup("r1", "p", warmup_end + 1.0)
    pw_r1.aux_model_score = -1.0
    # Post-warmup waiting scored
    pw_w_scored1 = FakeSeqGroup("w1", "p", warmup_end + 5.0)
    pw_w_scored1.aux_model_score = -3.5  # short job
    pw_w_scored2 = FakeSeqGroup("w2", "p", warmup_end + 6.0)
    pw_w_scored2.aux_model_score = -0.5  # medium
    # Post-warmup waiting unscored (predictor chưa kịp)
    pw_w_unscored = FakeSeqGroup("w3", "p", warmup_end + 7.0)

    sched.waiting = [pw_w_scored1, pw_w_scored2, pw_w_unscored]
    sched.running = [pw_r1]
    sched.swapped = []

    result = method(sched)
    result_ids = [r.request_id for r in result]

    # Schedulable = running + swapped + scored waiting
    # Unscored waiting BLOCKED bởi gate
    assert set(result_ids) == {"r1", "w1", "w2"}
    assert "w3" not in result_ids, "Unscored waiting leak vào Stage 3"

    # Sort by -aux_model_score, ascending (smaller -score first = larger score first):
    # Per baseline _get_opt_ordered_requests semantic: HIGHER aux_model_score
    # means SHORTER predicted job. Sort key = -score, ascending →
    # smaller -score (= larger score) goes first → SJF correct.
    #
    # All test scores are negative (all predicted as long jobs):
    #   w2 (score=-0.5, key=0.5)  ← least long, "shortest" → first
    #   r1 (score=-1.0, key=1.0)
    #   w1 (score=-3.5, key=3.5)  ← longest → last
    expected_order = ["w2", "r1", "w1"]
    assert result_ids == expected_order

    # Predictor được gọi để score w3
    assert len(sched.aux_model.submit_streaming_calls) == 1
    submitted_ids = [s.request_id
                     for s in sched.aux_model.submit_streaming_calls[0]]
    assert submitted_ids == ["w3"]


def test_stage_transition_2_to_3():
    """Stage 2 → 3 transition tự động khi warmup-era hết."""
    method = _import_scheduler_method()

    serve_start = time.time() - 10.0
    sched = _MockScheduler(warmup_seconds=2.0)
    sched.serve_start_time = serve_start
    warmup_end = serve_start + 2.0

    # Round 1: còn warmup-era → Stage 2
    we = FakeSeqGroup("we", "p", warmup_end - 0.5)
    pw = FakeSeqGroup("pw", "p", warmup_end + 1.0)
    pw.aux_model_score = -1.0  # đã scored

    sched.waiting = [we, pw]
    result_stage2 = method(sched)
    assert [r.request_id for r in result_stage2] == ["we"]
    assert "pw" not in [r.request_id for r in result_stage2]  # blocked

    # Round 2: warmup-era đã rời (giả lập decode xong)
    sched.waiting = [pw]  # we drained
    sched.running = []
    result_stage3 = method(sched)
    assert [r.request_id for r in result_stage3] == ["pw"]


def test_stage3_invariant_assertion_triggers_on_unscored_in_running():
    """Defensive assert phải trigger nếu running có unscored sg ở Stage 3.

    Đây là test bug detection — invariant phải catch nếu logic break.
    """
    method = _import_scheduler_method()

    serve_start = time.time() - 30.0
    sched = _MockScheduler(warmup_seconds=2.0)
    sched.serve_start_time = serve_start
    warmup_end = serve_start + 2.0

    # Post-warmup running với score=None (vi phạm invariant)
    bad_running = FakeSeqGroup("bad", "p", warmup_end + 1.0)
    # bad_running.aux_model_score = None (default)

    sched.running = [bad_running]
    sched.waiting = []
    sched.swapped = []

    # Phải raise AssertionError do invariant violated
    with pytest.raises(AssertionError, match="Stage 3 invariant violated"):
        method(sched)


def test_stage1_no_serve_start_time():
    """Edge case: serve_start_time chưa set (request đầu chưa qua add_seq_group).

    elapsed = 0.0 → vẫn ở Stage 1 (warmup phase, FCFS).
    """
    method = _import_scheduler_method()
    sched = _MockScheduler(warmup_seconds=2.0)
    sched.serve_start_time = None  # chưa có request nào

    r1 = FakeSeqGroup("a", "p", 0.0)
    sched.waiting = [r1]

    result = method(sched)
    # Stage 1 → sort by arrival_time
    assert [r.request_id for r in result] == ["a"]
    # Predictor không được gọi
    assert sched.aux_model.submit_streaming_calls == []


def test_warmup0_skips_to_stage3_immediately():
    """warmup_seconds=0 → bỏ qua Stage 1 + 2, vào Stage 3 ngay.

    Mọi request đều "post-warmup" (arrival_time >= serve_start_time).
    """
    method = _import_scheduler_method()
    sched = _MockScheduler(warmup_seconds=0.0)
    serve_start = time.time() - 1.0
    sched.serve_start_time = serve_start

    # arrival_time >= warmup_end (serve_start + 0.0 = serve_start)
    r1 = FakeSeqGroup("a", "p", serve_start + 0.5)
    r1.aux_model_score = -1.5  # đã scored

    sched.waiting = [r1]
    result = method(sched)
    assert [r.request_id for r in result] == ["a"]


def test_starvation_logic_in_stage3():
    """Starvation control áp dụng đúng trong Stage 3 (giống baseline)."""
    method = _import_scheduler_method()

    serve_start = time.time() - 30.0
    sched = _MockScheduler(warmup_seconds=2.0, starv=5)
    sched.period = 10
    sched.serve_start_time = serve_start
    warmup_end = serve_start + 2.0

    # 2 requests scored, 1 idle quá threshold
    starved = FakeSeqGroup("starved", "p", warmup_end + 1.0)
    starved.aux_model_score = -0.1  # score thấp (job dài)
    starved.idle = 6  # idle > starv=5 → promote pri=-1

    fresh = FakeSeqGroup("fresh", "p", warmup_end + 2.0)
    fresh.aux_model_score = -3.0  # score cao (job ngắn)
    fresh.idle = 1

    sched.waiting = [fresh, starved]
    result = method(sched)

    # Sau starvation: starved.pri = -1, fresh.pri = 0
    # Sort by (pri, -score): (-1, 0.1) < (0, 3.0) → starved trước
    assert [r.request_id for r in result] == ["starved", "fresh"]
    assert starved.pri == -1
    assert starved.idle == 0


# =============================================================================
# Phần 3 — Smoke test với real predictor (skip nếu không có model)
# =============================================================================

CONFIG_PATH = (Path(__file__).resolve().parent.parent / "benchmarks"
               / "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32"
               / "usage_config_ov.json")


@pytest.mark.skipif(not CONFIG_PATH.exists(),
                    reason="OPT-125M predictor checkpoint not available")
def test_smoke_real_predictor_streaming():
    """End-to-end smoke test với real OPT-125M predictor + streaming API.

    Skip nếu model file không có. Verify:
    - submit_streaming hoạt động với model thật
    - Scores được apply (giá trị real, không phải mock)
    - has_pending_stream lifecycle đúng
    """
    from vllm.config_predictor import PrefillPredictorConfig
    from vllm.model_executor.openvino_predictor import OpenVINOPredictor

    cfg = PrefillPredictorConfig.from_json(str(CONFIG_PATH)).model
    # Resolve cfg.path tương đối với benchmarks dir (path trong config là
    # relative khi user chạy từ benchmarks/). Fallback: tìm finetuned dir
    # gần CONFIG_PATH.
    model_path = cfg.path
    if not Path(model_path).exists():
        # Try resolve relative to CONFIG_PATH parent
        candidate = CONFIG_PATH.parent / "finetuned"
        if candidate.exists():
            model_path = str(candidate)

    p = OpenVINOPredictor(
        model_path=model_path,
        tokenizer_name=cfg.pred_model,
        num_labels=cfg.num_labels,
        max_length=cfg.max_length,
        max_batch_size=cfg.max_batch_size,
        num_threads=cfg.num_threads,
        inference_precision=cfg.inference_precision,
        async_mode=True,
    )
    try:
        sgs = [FakeSeqGroup(f"req{i}",
                            f"This is a test prompt number {i} " * 10,
                            float(i))
               for i in range(5)]

        assert p.submit_streaming(sgs) is True
        assert p.has_pending_stream() is True

        # Wait worker complete
        for _ in range(60):
            time.sleep(0.5)
            if not p.has_pending_stream():
                break

        # Verify scores applied + giá trị reasonable (không phải None,
        # không phải hằng số mock)
        scores = [s.aux_model_score for s in sgs]
        assert all(s is not None for s in scores), \
            f"Some scores still None: {scores}"
        # Scores phải khác nhau (5 prompts khác nhau, không tất cả cùng giá trị)
        assert len(set(scores)) > 1, f"All scores identical: {scores}"
    finally:
        p.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
