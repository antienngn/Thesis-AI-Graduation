"""
Test A4 — verify PrefillPredictorConfig.from_json load đúng JSON OV mới.

Chỉ kiểm tra:
  1. File JSON parse được không lỗi.
  2. 3 field mới (device, num_threads, inference_precision) được populate đúng.
  3. Field cũ vẫn parse đúng.
"""
import os

from vllm.config_predictor import PrefillPredictorConfig


def test_load_ov_json_config(json_config_path):
    """JSON config OV phải load được và 3 field mới có giá trị hợp lệ.

    num_threads value cụ thể là user-configurable (có thể thay đổi 4/8/16
    cho bench sweep), nên test chỉ assert kiểu và range hợp lý thay vì
    cố định một giá trị.
    """
    assert os.path.exists(json_config_path), f"Missing {json_config_path}"

    cfg = PrefillPredictorConfig.from_json(json_config_path)

    # Field mới (Edit A1)
    assert cfg.model.device == "openvino"
    assert isinstance(cfg.model.num_threads, int) and cfg.model.num_threads >= 1
    assert cfg.model.inference_precision in ("f32", "f16", "bf16")

    # Field cũ vẫn đúng
    assert cfg.model.pred_model == "facebook/opt-125m"
    assert cfg.model.num_labels == 1
    assert cfg.model.mtype == "rank"
    assert cfg.model.max_length == 2048
    assert cfg.model.max_batch_size == 1000


def test_legacy_json_still_works():
    """File JSON cũ (không có 3 field mới) vẫn load được — defaults được áp dụng."""
    legacy_path = (
        "/home/antn/vllm-ltr/benchmarks/MODEL/results/"
        "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json"
    )
    if not os.path.exists(legacy_path):
        # Nếu file gốc không tồn tại, skip
        import pytest
        pytest.skip(f"Legacy {legacy_path} not present")

    cfg = PrefillPredictorConfig.from_json(legacy_path)
    # 3 field mới phải fallback về default
    assert cfg.model.device == "auto"
    assert cfg.model.num_threads == 4
    assert cfg.model.inference_precision == "f16"
