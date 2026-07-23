import os

from summarize_rl.logging_utils import TensorBoardLogger, _metrics_to_scalars
from summarize_rl.train import StepMetrics


def _metrics(step=1):
    return StepMetrics(
        step=step, loss=1.0, mean_reward=0.5, faithfulness=0.4, coverage=0.6,
        term_usage=0.3, length_penalty=0.1, mean_len=20.0, weight_a=0.5,
        weight_b=0.33, weight_c=0.33, weight_d=0.34, entropy=1.0,
        grad_norm=2.0, lr=3e-5,
    )


def test_metrics_to_scalars_grouping():
    tags = _metrics_to_scalars(_metrics())
    assert "reward/mean" in tags
    assert "weights/a" in tags
    assert "train/loss" in tags
    assert tags["reward/coverage"] == 0.6


def test_disabled_logger_is_noop():
    logger = TensorBoardLogger("unused", enabled=False)
    assert logger.enabled is False
    # None of these should raise.
    logger.log_metrics(_metrics())
    logger.log_eval(1, {"coverage": 0.5})
    logger.log_scalar("x", 1.0, 1)
    logger.flush()
    logger.close()


def test_enabled_logger_writes_event_file(tmp_path):
    log_dir = os.path.join(tmp_path, "tb")
    logger = TensorBoardLogger(log_dir, enabled=True)
    # tensorboard should be installed in this env; if not, skip the assertion.
    if not logger.enabled:
        return
    logger.log_metrics(_metrics(1))
    logger.log_metrics(_metrics(2))
    logger.log_eval(2, {"coverage": 0.7})
    logger.flush()
    logger.close()
    files = os.listdir(log_dir)
    assert any(f.startswith("events.out.tfevents") for f in files)
