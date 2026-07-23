import pytest

from summarize_rl.llm_backend import plan_devices


def test_none_is_backward_compatible():
    plan = plan_devices(num_gpus=None, device="cpu")
    assert plan == {"device": "cpu", "device_map": None, "max_memory": None}


def test_none_keeps_explicit_device_map():
    plan = plan_devices(num_gpus=None, device="cuda", device_map="auto")
    assert plan["device_map"] == "auto"
    assert plan["max_memory"] is None


def test_single_gpu():
    plan = plan_devices(num_gpus=1)
    assert plan["device"] == "cuda:0"
    assert plan["device_map"] is None
    assert plan["max_memory"] is None


def test_multi_gpu_sharding():
    plan = plan_devices(num_gpus=4, max_memory_per_gpu="120GiB")
    assert plan["device_map"] == "auto"
    assert plan["max_memory"] == {0: "120GiB", 1: "120GiB", 2: "120GiB", 3: "120GiB"}


def test_multi_gpu_respects_explicit_device_map():
    plan = plan_devices(num_gpus=2, device_map="balanced")
    assert plan["device_map"] == "balanced"
    assert set(plan["max_memory"].keys()) == {0, 1}


def test_invalid_num_gpus():
    with pytest.raises(ValueError):
        plan_devices(num_gpus=0)
