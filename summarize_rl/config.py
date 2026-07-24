"""Configuration dataclasses for the PMI-weight RL summarization library.

All hyperparameters from the work plan (Section 5) live here so that the actual
backbone LLM, dataset, and glossary can be swapped by editing config only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyConfig:
    """Weight-generating policy network (Section 2.2)."""

    llm_hidden_size: int = 4096
    hidden_dim: int = 512
    dropout: float = 0.1
    use_contrast_features: bool = True  # append (h_* - h_Q) contrast features
    init_std: float = 0.01  # last-layer weight init N(0, init_std)
    # Prior-removal strength `a`. Default: learn it per token via sigmoid
    # (the reward's PMI contrast term trains it). This is safe now that the
    # contrast reward is tanh-bounded and the entropy bonus regularizes `a`,
    # which together prevent the earlier a->boundary collapse. Set learn_a=False
    # to instead hold `a` fixed at `fixed_a`, shrinking the action space to the
    # (b,c,d) simplex. The policy head stays 4-wide either way, so a checkpoint
    # loads under both settings (but must match training to reproduce results).
    learn_a: bool = True
    fixed_a: float = 0.5  # used when learn_a is False; must be in (0, 1)

    @property
    def num_branches(self) -> int:
        return 4  # XQ, SQ, GQ, Q

    @property
    def input_dim(self) -> int:
        """Feature dimension fed to the policy MLP.

        Base: concat of 4 branch hidden states.
        Contrast: plus (h_XQ - h_Q), (h_SQ - h_Q), (h_GQ - h_Q).
        """
        base = self.num_branches * self.llm_hidden_size
        if self.use_contrast_features:
            base += 3 * self.llm_hidden_size
        return base


@dataclass
class DecodeConfig:
    """Multi-branch PMI decoding (Section 2.1, 2.5.6)."""

    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 120
    min_new_tokens: int = 20
    eos_token_id: int | None = None
    pad_token_id: int | None = None


@dataclass
class RewardConfig:
    """Reference-free reward weights (Section 2.4, 2.5.7)."""

    w_faithfulness: float = 1.0
    w_coverage: float = 2.0  # triplet-entity coverage: the main on-topic anchor
    w_term: float = 0.25  # standard-term use; low so generic term-dropping can't win
    w_contrast: float = 0.5  # PMI contrast (source vs. prior); trains weight `a`
    w_length: float = 0.2
    w_copy: float = 1.0  # penalty for verbatim source copying (anti-reward-hacking)
    target_length: int = 120  # tokens; overage penalized
    repeat_ngram: int = 3  # n-gram size for repetition penalty
    copy_ngram: int = 4  # n-gram size for the extractive-copy penalty
    norm_eps: float = 1e-8


@dataclass
class TrainConfig:
    """SCST training loop (Section 2.5.8, Section 5)."""

    num_samples: int = 5  # N rollouts per input for self-critical baseline
    lr: float = 3e-4  # policy is a small MLP; a too-small lr leaves weights frozen
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.07  # 7% of total steps
    total_steps: int = 2000
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    entropy_beta: float = 0.01  # entropy bonus coefficient (0 disables)
    reward_norm: bool = True  # standardize rewards within the sample group
    baseline: str = "mean"  # "mean" (self-critical mean) or "greedy"
    save_every: int = 200
    seed: int = 42
    ckpt_dir: str = "checkpoints"


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization loop.

    Reuses TrainConfig for the optimizer/schedule (lr, betas, warmup,
    total_steps, grad_clip, grad_accum_steps, seed); the fields here are the
    GRPO-specific knobs that replace the SCST baseline/reward-norm settings.
    """

    group_size: int = 8  # G: rollouts per prompt; advantage normalized within
    clip_eps: float = 0.2  # PPO clip epsilon on the importance ratio
    kl_beta: float = 0.04  # KL-to-reference coefficient (DeepSeek default)
    inner_epochs: int = 2  # mu: gradient updates per sampled group (>1 uses clip)
    adv_eps: float = 1e-6  # std stabilizer for group-normalized advantage
    entropy_beta: float = 0.01  # entropy bonus on ALL weights; guards vs. collapse


@dataclass
class Config:
    """Top-level bundle."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
