"""
Test A1 — verify PrefillModelConfig đã accept 3 field mới với đúng default.

Mục đích: đảm bảo schema đã mở rộng đúng và default values khớp với plan.
"""
from vllm.config_predictor import PrefillModelConfig


def test_default_values_for_new_fields():
    """3 field mới phải có default value đúng theo plan."""
    cfg = PrefillModelConfig(
        pred_model="facebook/opt-125m",
        num_labels=1,
        mtype="rank",
        activation=None,
    )
    # Default phải nguyên vẹn để không phá flow cũ (auto = AUXLLM GPU)
    assert cfg.device == "auto"
    assert cfg.num_threads == 4
    assert cfg.inference_precision == "f16"


def test_accept_openvino_device_with_overrides():
    """User có thể override num_threads, precision khi device=openvino."""
    cfg = PrefillModelConfig(
        pred_model="facebook/opt-125m",
        num_labels=1,
        mtype="rank",
        activation=None,
        device="openvino",
        num_threads=8,
        inference_precision="f32",
    )
    assert cfg.device == "openvino"
    assert cfg.num_threads == 8
    assert cfg.inference_precision == "f32"


def test_existing_fields_still_work():
    """3 field mới không phá field cũ — đảm bảo backwards compat."""
    cfg = PrefillModelConfig(
        pred_model="facebook/opt-125m",
        num_labels=10,
        mtype="class",
        activation="Sigmoid",
        path="/some/path",
        max_length=2048,
        max_batch_size=1000,
    )
    assert cfg.pred_model == "facebook/opt-125m"
    assert cfg.num_labels == 10
    assert cfg.mtype == "class"
    assert cfg.activation == "Sigmoid"
    assert cfg.path == "/some/path"
    assert cfg.max_length == 2048
    assert cfg.max_batch_size == 1000
    # Default vẫn áp dụng cho field mới
    assert cfg.device == "auto"
