import pytest

from summarize_rl.logging_utils import (
    NullLogger,
    TensorBoardLogger,
    _scalar_items,
    make_logger,
)
from summarize_rl.train import StepMetrics


def _metrics():
    return StepMetrics(
        step=3, loss=-0.5, mean_reward=2.1, faithfulness=0.4, coverage=0.6,
        term_usage=0.5, contrast=3.2, length_penalty=0.1, mean_len=18.0,
        weight_a=0.55, weight_b=0.3, weight_c=0.33, weight_d=0.37,
        entropy=1.05, grad_norm=0.8, lr=2e-3,
    )


def test_make_logger_disabled_paths():
    for arg in ("", "none", "NONE", None):
        logger = make_logger(arg)
        assert isinstance(logger, NullLogger)
        logger.log_metrics(_metrics(), 3)  # no-op, must not raise
        logger.close()


def test_scalar_items_maps_and_skips_step():
    items = dict(_scalar_items(_metrics()))
    assert "step" not in items and "metrics/step" not in items
    assert items["reward/mean"] == 2.1
    assert items["weights/a"] == 0.55
    assert items["reward/contrast"] == pytest.approx(3.2)
    assert items["optim/lr"] == pytest.approx(2e-3)
    assert "grpo/kl" not in items  # SCST metrics have no kl field
    assert all(isinstance(v, float) for v in items.values())


def test_tensorboard_logger_writes_events(tmp_path):
    pytest.importorskip("tensorboard")
    logger = TensorBoardLogger(str(tmp_path))
    logger.log_metrics(_metrics(), 0)
    logger.log_metrics(_metrics(), 1)
    logger.close()
    assert any(p.name.startswith("events.out.tfevents") for p in tmp_path.iterdir())
