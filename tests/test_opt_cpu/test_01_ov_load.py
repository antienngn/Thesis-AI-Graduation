"""
Test 01 — Phase 1 GATE.

Kiểm tra:
  1. Checkpoint OPT-125m load được qua OpenVINO API trực tiếp
     (bypass optimum-intel vì conflict với torch 2.2 trong env này).
  2. Logits của OV và PyTorch khớp nhau (max abs diff < 1e-3).

Nếu test này FAIL → DỪNG implement, không đáng đi tiếp.
"""
import pytest
import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _LogitsOnlyWrapper(torch.nn.Module):
    """Wrap HuggingFace SequenceClassification model để forward chỉ trả logits.

    HF model trả `SequenceClassifierOutputWithPast` (dataclass có cả Tensor lẫn
    Tuple). TorchScript tracing không xử lý được mixed dataclass output, nên cần
    wrap để OV có thể trace được.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _build_ov_model(checkpoint_path, tokenizer_name, precision="f16", num_threads=4):
    """Convert PyTorch checkpoint → OpenVINO IR và compile cho CPU.

    Pattern này thay thế cho optimum.intel.openvino.OVModelForSequenceClassification
    vì optimum-intel không tương thích với torch 2.2 (nncf cần torch.uint16).
    """
    import openvino as ov

    # Load PyTorch model (eval mode để disable dropout)
    pt_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).eval()
    # Wrap để chỉ expose logits — bắt buộc cho OV tracing
    wrapped = _LogitsOnlyWrapper(pt_model).eval()

    # OpenVINO cần "example_input" để trace shape của graph
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    sample = tok(["dummy"], padding=True, truncation=True,
                 max_length=16, return_tensors="pt")
    example_input = (sample["input_ids"], sample["attention_mask"])

    # Convert sang OV IR (in-memory). Convert có thể mất 5-30s lần đầu.
    ov_model = ov.convert_model(wrapped, example_input=example_input)

    # Compile cho CPU với hint precision và thread count
    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU", {
        "INFERENCE_NUM_THREADS": str(num_threads),
        "INFERENCE_PRECISION_HINT": precision,
        "PERFORMANCE_HINT": "LATENCY",
    })
    return compiled, tok, pt_model


def test_ov_load_and_parity(checkpoint_path, tokenizer_name):
    """OV phải load checkpoint và sinh logits gần bằng PyTorch (sai số < 1e-3)."""
    compiled, tok, pt_model = _build_ov_model(checkpoint_path, tokenizer_name,
                                              precision="f32")  # f32 để parity check

    # Input giống nhau cho cả 2 model
    inp = tok(["hello world", "another test prompt for verification"],
              padding=True, truncation=True, max_length=128, return_tensors="pt")

    # PyTorch reference
    with torch.no_grad():
        pt_logits = pt_model(**inp).logits  # shape: [batch, num_labels]

    # OpenVINO inference. compiled() nhận positional dict-like input theo order
    # đã trace ở convert_model: (input_ids, attention_mask).
    ov_outputs = compiled([inp["input_ids"].numpy(), inp["attention_mask"].numpy()])
    # ov_outputs là dict keyed bởi output port. Dùng output đầu tiên.
    ov_logits = list(ov_outputs.values())[0]  # numpy array

    # Numeric parity
    diff = np.abs(pt_logits.numpy() - ov_logits).max()
    print(f"max abs diff (PT vs OV-f32): {diff:.6f}")
    assert diff < 1e-3, f"OV/PT logit diff {diff} too large (expected <1e-3)"


def test_ov_f16_parity(checkpoint_path, tokenizer_name):
    """Verify f16 precision không lệch quá xa f32 (< 1e-2 typical với BERT-like)."""
    compiled_f16, tok, pt_model = _build_ov_model(checkpoint_path, tokenizer_name,
                                                  precision="f16")

    inp = tok(["short prompt", "longer test prompt for f16 numerical check"],
              padding=True, truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        pt_logits = pt_model(**inp).logits

    ov_outputs = compiled_f16([inp["input_ids"].numpy(), inp["attention_mask"].numpy()])
    ov_logits = list(ov_outputs.values())[0]

    diff = np.abs(pt_logits.numpy() - ov_logits).max()
    print(f"max abs diff (PT vs OV-f16): {diff:.6f}")
    # f16 thường lệch nhiều hơn f32. Threshold lỏng hơn: 5e-2.
    assert diff < 5e-2, f"OV-f16 vs PT logit diff {diff} too large"
