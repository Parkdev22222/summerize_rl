"""PMI-weight RL summarization library.

Learns only the PMI combination weights [a, b, c, d] on top of a frozen LLM,
via Self-Critical Sequence Training with reference-free rewards.
"""

from .config import (
    Config,
    DecodeConfig,
    PolicyConfig,
    RewardConfig,
    TrainConfig,
)

__all__ = [
    "Config",
    "PolicyConfig",
    "DecodeConfig",
    "RewardConfig",
    "TrainConfig",
]
