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
    # Upper bound on the learned `a` (prior-removal strength). With a byte-level
    # BPE tokenizer, aggressive prior removal (a -> 1) over-tilts the next-token
    # distribution and can select invalid UTF-8 byte continuations, producing `�`
    # replacement chars in the output. Capping a to (0, a_max) via a scaled
    # sigmoid keeps contrastive decoding from distorting the byte distribution
    # that far. Default 1.0 = no cap (unchanged behavior); set < 1 (e.g. 0.5) to
    # bound it. Applies only when learn_a is True.
    a_max: float = 1.0

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
    max_new_tokens: int = 1024
    min_new_tokens: int = 20
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    # Adaptive plausibility constraint (Contrastive Decoding, Li et al. 2022).
    # Before sampling from the PMI-combined logits, drop every token the base
    # (XQ = source+query) distribution deems implausible -- prob < alpha * (max
    # base prob). The base is on-source and always UTF-8-valid, so this prunes
    # exactly the invalid byte continuations that aggressive prior removal (large
    # `a`) would otherwise let through as `�`, while leaving fluent on-source
    # tokens alone. Applied identically in sampling and GRPO re-scoring so
    # importance ratios stay well defined. 0 = disabled (unchanged behavior);
    # a typical enabled value is 0.1. Works at train AND inference time (no
    # retraining needed to protect a loaded checkpoint).
    plausibility_alpha: float = 0.0


@dataclass
class RewardConfig:
    """Reference-free reward weights (Section 2.4, 2.5.7)."""

    w_faithfulness: float = 1.0
    w_coverage: float = 2.0  # triplet-entity coverage: the main on-topic anchor
    w_keysent: float = 0.3  # cheap always-on lexical floor; judge does semantic key-sentence scoring
    w_contrast: float = 0.5  # PMI contrast (source vs. prior); trains weight `a`
    w_length: float = 0.2
    w_copy: float = 1.0  # penalty for verbatim source copying (anti-reward-hacking)
    w_hallucination: float = 1.0  # penalty for inventing units/quantities absent from source
    # LLM-as-judge accuracy reward (judge.py). Off by default (w_judge=0) because
    # it costs one extra LLM generation per scored summary. Set > 0 to add a
    # semantic accuracy score (catches errors lexical terms miss, e.g. 소대 vs
    # 소총중대). The trainers build a BackboneJudge (local frozen model) when
    # w_judge>0; a custom JudgeModel (e.g. an external API) can be injected too.
    w_judge: float = 0.0
    # Comparative judge (vs. the RAW frozen-LLM summary) instead of absolute
    # 0-100 scoring. The absolute score anchors near-constant (~0.85 for any
    # decent summary), giving almost no learning signal; a pairwise "is this
    # better than RAW?" verdict is discriminative per rollout and directly
    # optimizes the objective "beat the base model". Requires w_judge>0 and a
    # reference summary (the trainers generate + cache the RAW one per source).
    # judge in [0,1] then reads as the win-rate vs RAW (1 win / 0.5 tie / 0 loss).
    judge_comparative: bool = False
    judge_max_new_tokens: int = 24  # room for a short preamble before the score / A|B|T verdict
    target_length: int = 1024  # tokens; overage penalized (kept == decode.max_new_tokens)
    repeat_ngram: int = 3  # n-gram size for repetition penalty
    copy_ngram: int = 4  # n-gram size for the extractive-copy penalty
    # Key-sentence extraction (used only when w_keysent > 0): the frozen LLM is
    # prompted once per source to pick the N most important sentences; the
    # summary is then rewarded for reflecting their content. Cached per source.
    keysent_n: int = 3  # how many key sentences to ask the LLM for
    keysent_max_new_tokens: int = 256  # generation budget for extraction
    # Military-importance weighting of key sentences (keysent.py). When on, every
    # source sentence describing a military event / echelon status / casualty /
    # request is ALWAYS a key sentence, weighted by its computed importance, so
    # the key-sentence reward (a weighted average) makes reflecting the important
    # events count more. keysent_use_llm additionally merges the LLM's salient
    # picks; keysent_max caps the set; keysent_min_weight is the importance
    # threshold for a sentence to count as "military".
    keysent_importance: bool = True
    keysent_use_llm: bool = True
    keysent_min_weight: float = 1.0
    keysent_max: int = 8
    norm_eps: float = 1e-8
    # Content gating (anti "fluent-but-off-topic"): when True, the fluency-style
    # components (faithfulness, term, contrast) are multiplied by how much source
    # content the summary actually captured (coverage + key-sentence recall), so a
    # summary that is grammatical/on-genre but not about *this* source cannot earn
    # them. Coverage and key-sentence stay additive (they must always pull toward
    # content). Off by default so the plain additive reward is unchanged; flip on
    # to directly counter reward-component imbalance. See compute_reward.
    balance_content: bool = False
    gate_floor: float = 0.1  # minimum gate so a cold-start summary still gets *some* signal


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
