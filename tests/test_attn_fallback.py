"""HFBackend attention-kernel selection + fallback (no transformers needed).

Drives the pure ``_attn_fallback_chain`` / ``_load_model`` logic with a fake
loader, so we verify the flash_attention_2 -> sdpa -> eager fallback without a
GPU, flash-attn, or a real model.
"""

import torch

from summarize_rl.llm_backend import HFBackend


def test_fallback_chain():
    assert HFBackend._attn_fallback_chain("flash_attention_2") == [
        "flash_attention_2", "sdpa", "eager"
    ]
    assert HFBackend._attn_fallback_chain("sdpa") == ["sdpa", "eager"]
    assert HFBackend._attn_fallback_chain("eager") == ["eager"]
    assert HFBackend._attn_fallback_chain(None) == [None]


class _FakeAuto:
    """Stand-in for AutoModelForCausalLM: fails for chosen kernels."""

    def __init__(self, fail_impls):
        self.fail_impls = set(fail_impls)
        self.calls = []

    def from_pretrained(self, name, **kwargs):
        impl = kwargs.get("attn_implementation")
        self.calls.append(impl)
        assert kwargs["output_hidden_states"] is True
        if impl in self.fail_impls:
            raise ImportError(f"{impl} not installed")
        return f"model[{impl}]"


def test_load_model_uses_requested_when_available():
    fake = _FakeAuto(fail_impls=[])
    model, impl = HFBackend._load_model(fake, "m", torch.float32, "flash_attention_2")
    assert impl == "flash_attention_2"
    assert model == "model[flash_attention_2]"
    assert fake.calls == ["flash_attention_2"]


def test_load_model_falls_back_to_sdpa():
    fake = _FakeAuto(fail_impls=["flash_attention_2"])
    model, impl = HFBackend._load_model(fake, "m", torch.float32, "flash_attention_2")
    assert impl == "sdpa"
    assert model == "model[sdpa]"
    assert fake.calls == ["flash_attention_2", "sdpa"]


def test_load_model_falls_back_to_eager():
    fake = _FakeAuto(fail_impls=["flash_attention_2", "sdpa"])
    model, impl = HFBackend._load_model(fake, "m", torch.float32, "flash_attention_2")
    assert impl == "eager"
    assert fake.calls == ["flash_attention_2", "sdpa", "eager"]


def test_load_model_default_passes_no_attn_kwarg():
    fake = _FakeAuto(fail_impls=[])
    model, impl = HFBackend._load_model(fake, "m", torch.float32, None)
    assert impl is None
    assert fake.calls == [None]  # transformers default, no attn_implementation kwarg
