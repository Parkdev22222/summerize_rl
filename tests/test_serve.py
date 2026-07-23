"""SummarizerService tests (MockBackend, no sockets, no model download).

Exercises the server's request-handling logic directly; the HTTP layer around
it is thin glue.
"""

import os

import pytest
import torch

from summarize_rl.config import Config, DecodeConfig, PolicyConfig
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy

from examples.serve import SummarizerService

DEFAULT_QUERY = "요약하시오."
SOURCE = "적 부대가 이동 중이며 고지를 점령했다."


def _config():
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    return cfg


def _service():
    cfg = _config()
    policy = WeightPolicy(cfg.policy)
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    glossary = Glossary({"기동": ["이동"], "점령": ["점령"]})
    summarizer = Summarizer(backend, policy, cfg, glossary=glossary)
    return SummarizerService(summarizer, DEFAULT_QUERY)


def test_summarize_returns_expected_fields():
    svc = _service()
    out = svc.summarize({"source": SOURCE})
    assert out["text"] != ""
    assert set(out["active_terms"]) == {"기동", "점령"}
    assert out["query"] == DEFAULT_QUERY
    assert len(out["mean_weights"]) == 4
    # JSON-serializable: mean_weights is a list, not a tuple.
    assert isinstance(out["mean_weights"], list)


def test_summarize_uses_query_override():
    svc = _service()
    out = svc.summarize({"source": SOURCE, "query": "핵심만 요약하시오."})
    assert out["query"] == "핵심만 요약하시오."


def test_summarize_missing_source_raises():
    svc = _service()
    with pytest.raises(ValueError):
        svc.summarize({})
    with pytest.raises(ValueError):
        svc.summarize({"source": "   "})


def test_reload_ckpt(tmp_path):
    svc = _service()
    # Save the current policy weights and reload them through the service.
    path = os.path.join(str(tmp_path), "ckpt.pt")
    torch.save({"policy": svc.summarizer.policy.state_dict(), "step": 7}, path)
    out = svc.reload({"ckpt": path})
    assert out["ckpt"] == path
    assert out["step"] == 7


def test_reload_missing_field_raises():
    svc = _service()
    with pytest.raises(ValueError):
        svc.reload({})


def test_reload_missing_file_raises():
    svc = _service()
    with pytest.raises(FileNotFoundError):
        svc.reload({"ckpt": "/no/such/file.pt"})
