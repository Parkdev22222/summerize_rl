"""TensorBoard logging for training metrics.

Both trainers return a metrics dataclass per step (StepMetrics for SCST,
GRPOMetrics for GRPO). `TensorBoardLogger.log_metrics` introspects any such
dataclass and writes every numeric field as a scalar under a tidy tag
namespace (reward/*, weights/*, loss/*, grpo/*, ...), so the two objectives
land on comparable dashboards. `tensorboard` is an optional dependency; the
import error is deferred until a logger is actually requested.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

# field name -> TensorBoard tag (grouped for a readable dashboard).
# Fields not listed fall back to "metrics/<name>"; `step` is the x-axis.
_TAG_MAP = {
    "loss": "loss/loss",
    "grad_norm": "loss/grad_norm",
    "lr": "optim/lr",
    "mean_reward": "reward/mean",
    "faithfulness": "reward/faithfulness",
    "coverage": "reward/coverage",
    "term_usage": "reward/term_usage",
    "contrast": "reward/contrast",
    "length_penalty": "reward/length_penalty",
    "mean_len": "misc/mean_len",
    "weight_a": "weights/a",
    "weight_b": "weights/b",
    "weight_c": "weights/c",
    "weight_d": "weights/d",
    "entropy": "policy/entropy",
    "kl": "grpo/kl",
    "clip_frac": "grpo/clip_frac",
}


def _scalar_items(metrics: Any) -> list[tuple[str, float]]:
    """(tag, value) for every numeric field of a metrics dataclass except step."""
    if not is_dataclass(metrics):
        raise TypeError(f"expected a metrics dataclass, got {type(metrics)!r}")
    out = []
    for f in fields(metrics):
        if f.name == "step":
            continue
        val = getattr(metrics, f.name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        out.append((_TAG_MAP.get(f.name, f"metrics/{f.name}"), float(val)))
    return out


class NullLogger:
    """No-op logger (logging disabled)."""

    def log_metrics(self, metrics: Any, step: int) -> None:  # noqa: D102
        pass

    def close(self) -> None:  # noqa: D102
        pass


class TensorBoardLogger:
    """Writes metrics-dataclass fields to a TensorBoard run directory."""

    def __init__(self, logdir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:  # pragma: no cover - exercised only without tb
            raise ImportError(
                "TensorBoard logging requires the 'tensorboard' package. "
                "Install it with `uv sync --extra tb` or `pip install tensorboard`, "
                "or disable logging (e.g. --logdir '')."
            ) from e
        self.writer = SummaryWriter(logdir)
        self.logdir = logdir

    def log_metrics(self, metrics: Any, step: int) -> None:
        for tag, value in _scalar_items(metrics):
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def make_logger(logdir: str | None):
    """TensorBoardLogger for a truthy logdir, else a NullLogger.

    An empty string or the literal "none" disables logging, so callers can
    default `--logdir` on and let users opt out.
    """
    if not logdir or str(logdir).lower() == "none":
        return NullLogger()
    return TensorBoardLogger(logdir)
