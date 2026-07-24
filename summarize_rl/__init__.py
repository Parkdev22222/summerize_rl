"""PMI-weight RL summarization library.

Learns only the PMI combination weights [a, b, c, d] on top of a frozen LLM,
with reference-free rewards. Two RL objectives are available: Self-Critical
Sequence Training (SCSTTrainer) and Group Relative Policy Optimization
(GRPOTrainer).
"""

from .config import (
    Config,
    DecodeConfig,
    GRPOConfig,
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
    "GRPOConfig",
]
