"""
Shared pytest fixtures cho test suite test_opt_cpu.

- CHECKPOINT_PATH: đường dẫn tới checkpoint OPT-125m fine-tuned cho ranking.
- TOKENIZER_NAME: tokenizer gốc OPT-125m từ HuggingFace.
- JSON_CONFIG_PATH: file JSON cấu hình OpenVINO predictor (sẽ được tạo ở Edit A4).
"""
import pytest

# Đường dẫn checkpoint đã verify trong Phase 1 — head: OPTForSequenceClassification, num_labels=1
CHECKPOINT_PATH = (
    "/home/antn/vllm-ltr/benchmarks/MODEL/results/"
    "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned"
)

# Tokenizer base — checkpoint dùng cùng tokenizer của OPT gốc
TOKENIZER_NAME = "facebook/opt-125m"

# Config JSON cho server (sẽ tồn tại sau Edit A4)
JSON_CONFIG_PATH = (
    "/home/antn/vllm-ltr/benchmarks/MODEL/results/"
    "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config_ov.json"
)


# Scope = "module" để fixture có thể được consume bởi module-scoped fixtures
# trong test files (vd predictor instance được cache qua nhiều test).
@pytest.fixture(scope="module")
def checkpoint_path():
    return CHECKPOINT_PATH


@pytest.fixture(scope="module")
def tokenizer_name():
    return TOKENIZER_NAME


@pytest.fixture(scope="module")
def json_config_path():
    return JSON_CONFIG_PATH
