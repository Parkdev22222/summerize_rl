"""TensorBoard logging for training (Section 2.5.9).

Wraps torch's SummaryWriter with a graceful no-op fallback when tensorboard is
not installed, so training never hard-fails on a logging dependency. Scalars are
grouped so the TensorBoard UI shows reward/, weights/, train/, gen/ panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a runtime import cycle with train.py
    from .train import StepMetrics


def _metrics_to_scalars(m: "StepMetrics") -> dict[str, float]:
    """Map a StepMetrics into grouped TensorBoard scalar tags."""
    return {
        "train/loss": m.loss,
        "train/grad_norm": m.grad_norm,
        "train/lr": m.lr,
        "train/entropy": m.entropy,
        "reward/mean": m.mean_reward,
        "reward/faithfulness": m.faithfulness,
        "reward/coverage": m.coverage,
        "reward/term_usage": m.term_usage,
        "reward/length_penalty": m.length_penalty,
        "weights/a": m.weight_a,
        "weights/b": m.weight_b,
        "weights/c": m.weight_c,
        "weights/d": m.weight_d,
        "gen/mean_len": m.mean_len,
    }


class TensorBoardLogger:
    """Thin SummaryWriter wrapper. Disabled instances are safe no-ops."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer: Any = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception as exc:  # tensorboard not installed / init failure
            print(f"[logging] TensorBoard disabled ({exc}); continuing without it.")
            self.writer = None

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def log_metrics(self, metrics: "StepMetrics") -> None:
        if self.writer is None:
            return
        for tag, value in _metrics_to_scalars(metrics).items():
            self.writer.add_scalar(tag, value, metrics.step)

    def log_eval(self, step: int, eval_metrics: dict[str, float], prefix: str = "eval") -> None:
        if self.writer is None:
            return
        for key, value in eval_metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
