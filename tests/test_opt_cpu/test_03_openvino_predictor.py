"""
Test A2/D1 — verify OpenVINOPredictor:
  Sync API (legacy):
    - Init thành công (load checkpoint + compile OV).
    - obtain_aux_scores trả đúng số score, gọi set_aux_model_score.
    - Empty input không crash. Mini-batch split.
  Async API (D1):
    - submit_async non-blocking, trả True nếu OK / False nếu busy.
    - poll_results: trả 0 khi chưa done, > 0 khi done.
    - is_busy() phản ánh đúng trạng thái pending Future.
    - Lifecycle: submit → poll cho đến khi scores được set.

Test dùng SequenceGroup mock để không phụ thuộc vào full vLLM init.
"""
import time

import pytest
from unittest.mock import MagicMock

from vllm.model_executor.openvino_predictor import OpenVINOPredictor


# Cache predictor instance giữa các test — load + compile mất ~30s
@pytest.fixture(scope="module")
def predictor(checkpoint_path, tokenizer_name):
    """Init OpenVINOPredictor một lần, share giữa các test."""
    p = OpenVINOPredictor(
        model_path=checkpoint_path,
        tokenizer_name=tokenizer_name,
        num_labels=1,
        max_length=512,
        max_batch_size=32,
        num_threads=2,            # ít thread cho test môi trường
        inference_precision="f16",
        async_mode=True,          # bật async API
    )
    yield p
    p.shutdown()


def _make_mock_seq_group(prompt: str):
    """Tạo SequenceGroup mock chỉ với fields predictor cần.

    Predictor truy cập:
      sg.seqs_dict[<key>].prompt
      sg.set_aux_model_score(score)
    """
    sg = MagicMock()
    seq = MagicMock()
    seq.prompt = prompt
    sg.seqs_dict = {"k0": seq}
    return sg


def _wait_until_idle(predictor, timeout=30.0):
    """Helper: đợi pending future done, drain qua poll_results.

    Dùng để đưa predictor về trạng thái idle giữa các test (vì fixture
    scope=module, state pending có thể leak từ test trước).
    """
    start = time.time()
    while predictor.is_busy():
        if time.time() - start > timeout:
            raise TimeoutError("Predictor future did not complete within timeout")
        time.sleep(0.01)
    predictor.poll_results()  # drain bất kỳ result còn lại


# =============================================================================
# Sync API tests (legacy — phải pass nguyên sau refactor)
# =============================================================================

def test_predictor_init(predictor):
    """Init không lỗi, các attribute config đúng (gồm async infra)."""
    assert predictor.max_length == 512
    assert predictor.max_batch_size == 32
    assert predictor.compiled_model is not None
    assert predictor.tokenizer is not None
    # Async infra đã được tạo
    assert predictor.async_mode is True
    assert predictor._executor is not None
    assert predictor._pending_future is None


def test_obtain_aux_scores_returns_correct_count(predictor):
    """obtain_aux_scores trả đúng số score = số seq_groups input."""
    _wait_until_idle(predictor)
    sgs = [
        _make_mock_seq_group("hello world"),
        _make_mock_seq_group("another prompt"),
        _make_mock_seq_group("third one is here"),
    ]
    scores = predictor.obtain_aux_scores(sgs)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_obtain_aux_scores_calls_set_aux_model_score(predictor):
    """Sau khi compute score, mỗi seq_group phải được gọi set_aux_model_score(score)."""
    _wait_until_idle(predictor)
    sgs = [
        _make_mock_seq_group("prompt one"),
        _make_mock_seq_group("prompt two"),
    ]
    scores = predictor.obtain_aux_scores(sgs)
    for sg, expected_score in zip(sgs, scores):
        sg.set_aux_model_score.assert_called_once_with(expected_score)


def test_obtain_aux_scores_empty_input(predictor):
    """Empty input → trả [] không crash, không gọi predictor."""
    _wait_until_idle(predictor)
    scores = predictor.obtain_aux_scores([])
    assert scores == []


def test_obtain_aux_scores_minibatch_split(predictor):
    """Nếu len(seq_groups) > max_batch_size, predictor phải split mini-batch."""
    _wait_until_idle(predictor)
    sgs = [_make_mock_seq_group(f"prompt {i}") for i in range(50)]
    scores = predictor.obtain_aux_scores(sgs)
    assert len(scores) == 50
    for sg in sgs:
        sg.set_aux_model_score.assert_called_once()


def test_scores_are_deterministic(predictor):
    """Cùng input → cùng output (predictor là pure inference, no dropout)."""
    _wait_until_idle(predictor)
    sg1 = _make_mock_seq_group("test prompt for determinism")
    sg2 = _make_mock_seq_group("test prompt for determinism")
    s1 = predictor.obtain_aux_scores([sg1])[0]
    s2 = predictor.obtain_aux_scores([sg2])[0]
    assert abs(s1 - s2) < 1e-5


# =============================================================================
# Async API tests (D1 mới)
# =============================================================================

def test_async_initial_state_not_busy(predictor):
    """Trước khi submit, predictor không busy."""
    _wait_until_idle(predictor)
    assert predictor.is_busy() is False


def test_submit_async_returns_true_when_idle(predictor):
    """submit_async trả True khi predictor đang rảnh."""
    _wait_until_idle(predictor)

    sg = _make_mock_seq_group("async test prompt")
    ok = predictor.submit_async([sg])
    assert ok is True
    assert predictor.is_busy() is True   # vừa submit, đang chạy

    _wait_until_idle(predictor)


def test_submit_async_returns_false_when_busy(predictor):
    """submit_async trả False khi đã có future pending."""
    _wait_until_idle(predictor)

    sg1 = _make_mock_seq_group("first batch")
    ok1 = predictor.submit_async([sg1])
    assert ok1 is True

    # Trong khi future_1 chưa done, submit_async lần 2 phải bị skip
    sg2 = _make_mock_seq_group("second batch")
    ok2 = predictor.submit_async([sg2])
    assert ok2 is False
    # sg2 KHÔNG được score (predictor đang busy với sg1)

    _wait_until_idle(predictor)
    # Verify chỉ sg1 được set score, sg2 không được
    sg1.set_aux_model_score.assert_called_once()
    sg2.set_aux_model_score.assert_not_called()


def test_submit_async_empty_input(predictor):
    """Submit empty list → trả False, không tạo future."""
    _wait_until_idle(predictor)
    assert predictor.submit_async([]) is False
    assert predictor.is_busy() is False


def test_poll_results_zero_when_not_busy(predictor):
    """poll_results trả 0 khi không có future pending."""
    _wait_until_idle(predictor)
    assert predictor.poll_results() == 0


def test_poll_results_returns_count_when_done(predictor):
    """Sau khi future done, poll_results trả đúng số seq_group được set score."""
    _wait_until_idle(predictor)

    sgs = [
        _make_mock_seq_group("alpha"),
        _make_mock_seq_group("beta"),
        _make_mock_seq_group("gamma"),
    ]
    predictor.submit_async(sgs)
    # Đợi worker thread compute xong
    while predictor.is_busy():
        time.sleep(0.01)

    n = predictor.poll_results()
    assert n == 3
    # Sau poll, không còn pending
    assert predictor.is_busy() is False
    # Mỗi sg đã được set score đúng 1 lần
    for sg in sgs:
        sg.set_aux_model_score.assert_called_once()


def test_async_full_lifecycle(predictor):
    """End-to-end: submit → busy → wait → poll cập nhật score → idle."""
    _wait_until_idle(predictor)

    sgs = [_make_mock_seq_group(f"lifecycle prompt {i}") for i in range(5)]
    assert predictor.submit_async(sgs) is True
    assert predictor.is_busy() is True

    # Đợi worker xong
    while predictor.is_busy():
        time.sleep(0.01)

    n = predictor.poll_results()
    assert n == 5
    # State reset sau poll
    assert predictor.is_busy() is False
    # Mỗi sg được set score đúng 1 lần
    for sg in sgs:
        sg.set_aux_model_score.assert_called_once()


def test_async_scores_match_sync(predictor):
    """Score từ async path phải khớp với sync path (cùng input, cùng model)."""
    _wait_until_idle(predictor)

    prompt = "consistency check between sync and async"

    # Sync
    sg_sync = _make_mock_seq_group(prompt)
    sync_score = predictor.obtain_aux_scores([sg_sync])[0]

    # Async
    sg_async = _make_mock_seq_group(prompt)
    predictor.submit_async([sg_async])
    while predictor.is_busy():
        time.sleep(0.01)
    predictor.poll_results()

    # Lấy score từ mock call_args (set_aux_model_score(score))
    async_score = sg_async.set_aux_model_score.call_args[0][0]

    assert abs(sync_score - async_score) < 1e-5


def test_submit_async_disabled_when_async_mode_false(checkpoint_path, tokenizer_name):
    """Init với async_mode=False → submit_async raise RuntimeError."""
    p = OpenVINOPredictor(
        model_path=checkpoint_path,
        tokenizer_name=tokenizer_name,
        num_labels=1,
        max_length=128,
        max_batch_size=8,
        num_threads=2,
        inference_precision="f16",
        async_mode=False,
    )
    try:
        sg = _make_mock_seq_group("should fail")
        with pytest.raises(RuntimeError, match="Async mode disabled"):
            p.submit_async([sg])
    finally:
        p.shutdown()
