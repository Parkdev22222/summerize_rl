"""Self-Critical Sequence Training for the weight policy (Section 2.5).

Only the policy network theta is updated. The LLM is frozen and its logits are
detached inside the decoder, so gradients cannot reach it. The loop:

  1. N stochastic rollouts per input.
  2. reference-free reward per rollout.
  3. self-critical baseline b (mean of the N rewards, or greedy reward).
  4. advantage A_i = R~_i - b.
  5. loss = -(1/N) sum_i A_i * (Lref/Li) * sum_t(logpi_t)   (- beta * entropy).
  6. AdamW step with grad clipping and accumulation.

Length handling (step 5): each rollout's sequence log-prob is scaled by
(group-mean length / its own length) rather than divided by its own length.
This down-weights over-long rollouts (the point of length normalization)
without collapsing the overall gradient magnitude by ~L, which previously
pushed the grad norm far below grad_clip and left the policy weights frozen.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .branches import Example, build_branches
from .config import Config
from .decoder import Rollout, generate, generate_batch
from .judge import BackboneJudge
from .keysent import KeySentenceExtractor
from .glossary import Glossary
from .llm_backend import LLMBackend
from .policy import WeightPolicy
from .rewards import (
    FaithfulnessModel,
    RewardBreakdown,
    compute_reward,
    normalize_rewards,
)
from .weakness import FailureLog


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def freeze_llm(backend: LLMBackend) -> None:
    """Ensure the backbone is frozen (no-op for backends without a model)."""
    model = getattr(backend, "model", None)
    if model is not None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)


def assert_llm_frozen(backend: LLMBackend) -> None:
    model = getattr(backend, "model", None)
    if model is not None:
        for p in model.parameters():
            assert not p.requires_grad, "LLM parameter is not frozen"


@dataclass
class StepMetrics:
    step: int
    loss: float
    mean_reward: float
    faithfulness: float
    coverage: float
    key_sentence: float
    contrast: float
    length_penalty: float
    copy_penalty: float
    mean_len: float
    weight_a: float
    weight_b: float
    weight_c: float
    weight_d: float
    entropy: float
    grad_norm: float
    lr: float
    hallucination: float = 0.0
    judge: float = 0.0


class SCSTTrainer:
    def __init__(
        self,
        policy: WeightPolicy,
        backend: LLMBackend,
        config: Config,
        *,
        glossary: Glossary | None = None,
        faithfulness_model: FaithfulnessModel | None = None,
        generator: torch.Generator | None = None,
        failure_log: "FailureLog | None" = None,
    ):
        self.policy = policy
        self.backend = backend
        self.config = config
        self.glossary = glossary
        self.faithfulness_model = faithfulness_model
        self.generator = generator
        self.failure_log = failure_log
        self.key_extractor = (
            KeySentenceExtractor(config.reward) if config.reward.w_keysent > 0 else None
        )
        self.judge = (
            BackboneJudge(backend, config.reward) if config.reward.w_judge > 0 else None
        )
        # RAW (frozen-LLM) reference summary per source, for the comparative
        # judge ("beat RAW"). Generated once per source and cached.
        self._raw_cache: dict[str, str] = {}

        freeze_llm(backend)

        t = config.train
        self.optimizer = AdamW(
            policy.parameters(),
            lr=t.lr,
            betas=t.betas,
            eps=t.eps,
            weight_decay=t.weight_decay,
        )
        warmup = int(t.warmup_ratio * t.total_steps)
        self.scheduler = build_scheduler(self.optimizer, warmup, t.total_steps)
        self._global_step = 0
        self.best_reward = -float("inf")

    # -- one input: N rollouts, reward, advantage, loss ---------------------

    def _rollouts_and_rewards(
        self, example: Example
    ) -> tuple[list[Rollout], list[RewardBreakdown], list[str]]:
        active = self.glossary.gate(example.source) if self.glossary else []
        active_terms = [a.term for a in active]
        branch_texts = build_branches(example, active).as_dict()
        key_sents = self._key_sentences(example)
        raw_ref = self._raw_reference(branch_texts, example.source)

        # All N rollouts share this prompt -> one batched forward (batch = N*4)
        # instead of N sequential decodes. The recorded log-probs carry grad, so
        # the on-policy SCST loss uses them directly (no re-scoring).
        rollouts = generate_batch(
            self.backend, branch_texts, self.policy, self.config.decode,
            n=self.config.train.num_samples, greedy=False, generator=self.generator,
        )
        breakdowns = [
            compute_reward(
                summary=r.text,
                source=example.source,
                triplets=example.triplets,
                active_terms=active_terms,
                config=self.config.reward,
                faithfulness_model=self.faithfulness_model,
                summary_length=r.length,
                contrast=r.mean_contrast(),
                key_sentences=key_sents,
                judge_model=self.judge,
                reference_summary=raw_ref,
            )
            for r in rollouts
        ]
        if self.failure_log is not None:
            for r, bd in zip(rollouts, breakdowns):
                self.failure_log.record(self._global_step, example, r.text, bd)
        return rollouts, breakdowns, active_terms

    def _key_sentences(self, example: Example) -> list[str] | None:
        """LLM-extracted key sentences for this source (cached), or None if off."""
        if self.key_extractor is None:
            return None
        return self.key_extractor.extract(self.backend, example.source)

    def _raw_reference(self, branch_texts: dict[str, str], source: str) -> str | None:
        """RAW frozen-LLM summary for this source (cached), for the comparative judge."""
        r = self.config.reward
        if not (r.w_judge > 0 and r.judge_comparative):
            return None
        if source not in self._raw_cache:
            with torch.no_grad():
                self._raw_cache[source] = self.backend.generate_text(
                    branch_texts["XQ"], self.config.decode.max_new_tokens
                ).strip()
        return self._raw_cache[source]

    def _greedy_reward(self, example: Example, active_terms: list[str]) -> float:
        active = self.glossary.gate(example.source) if self.glossary else []
        branch_texts = build_branches(example, active).as_dict()
        with torch.no_grad():
            r = generate(
                self.backend, branch_texts, self.policy, self.config.decode, greedy=True
            )
        bd = compute_reward(
            summary=r.text,
            source=example.source,
            triplets=example.triplets,
            active_terms=active_terms,
            config=self.config.reward,
            faithfulness_model=self.faithfulness_model,
            summary_length=r.length,
            contrast=r.mean_contrast(),
            key_sentences=self._key_sentences(example),
            judge_model=self.judge,
            reference_summary=self._raw_reference(branch_texts, example.source),
        )
        return bd.total

    def compute_loss(self, example: Example) -> tuple[torch.Tensor, dict]:
        t = self.config.train
        rollouts, breakdowns, active_terms = self._rollouts_and_rewards(example)
        raw_rewards = [bd.total for bd in breakdowns]

        if t.reward_norm:
            rewards = normalize_rewards(raw_rewards, eps=self.config.reward.norm_eps)
        else:
            rewards = list(raw_rewards)

        if t.baseline == "greedy":
            baseline = self._greedy_reward(example, active_terms)
        else:  # self-critical mean
            baseline = sum(rewards) / len(rewards)

        advantages = [r - baseline for r in rewards]

        # Group-relative length normalization: scale each rollout's sequence
        # log-prob by (mean_len / its_len) instead of dividing by its own
        # length. This keeps over-long rollouts from dominating (the goal of
        # length normalization) while holding the overall gradient magnitude at
        # the ~sum_logp scale, so grad_clip still binds and the policy actually
        # moves. Dividing by each rollout's own length shrank the gradient by
        # ~L (~24x here), dropping it below grad_clip and freezing the weights.
        lengths = [max(r.length, 1) for r in rollouts]
        ref_len = sum(lengths) / len(lengths)
        loss = torch.zeros((), dtype=torch.float32)
        for adv, rollout, length in zip(advantages, rollouts, lengths):
            norm_logp = rollout.sum_logp() * (ref_len / length)
            loss = loss - adv * norm_logp
        loss = loss / len(rollouts)

        entropy = torch.stack([r.mean_entropy() for r in rollouts]).mean()
        if t.entropy_beta > 0:
            loss = loss - t.entropy_beta * entropy

        metrics = self._make_metrics(rollouts, breakdowns, raw_rewards, entropy)
        return loss, metrics

    def _make_metrics(self, rollouts, breakdowns, raw_rewards, entropy) -> dict:
        weights = [w for r in rollouts for w in r.weight_trace]
        n = max(1, len(weights))
        a = sum(w[0] for w in weights) / n
        b = sum(w[1] for w in weights) / n
        c = sum(w[2] for w in weights) / n
        d = sum(w[3] for w in weights) / n
        k = len(breakdowns)
        return {
            "mean_reward": sum(raw_rewards) / k,
            "faithfulness": sum(bd.faithfulness for bd in breakdowns) / k,
            "coverage": sum(bd.coverage for bd in breakdowns) / k,
            "key_sentence": sum(bd.key_sentence for bd in breakdowns) / k,
            "hallucination": sum(bd.hallucination for bd in breakdowns) / k,
            "judge": sum(bd.judge for bd in breakdowns) / k,
            "contrast": sum(bd.contrast for bd in breakdowns) / k,
            "length_penalty": sum(bd.length_penalty for bd in breakdowns) / k,
            "copy_penalty": sum(bd.copy_penalty for bd in breakdowns) / k,
            "mean_len": sum(r.length for r in rollouts) / len(rollouts),
            "weight_a": a,
            "weight_b": b,
            "weight_c": c,
            "weight_d": d,
            "entropy": float(entropy.item()),
        }

    # -- one optimizer step over a micro-batch (grad accumulation) ----------

    def train_step(self, batch: list[Example]) -> StepMetrics:
        """One optimizer step; `batch` is the grad-accumulation micro-batch."""
        self.policy.train()
        self.optimizer.zero_grad()

        agg: dict[str, float] = {}
        batch_loss = 0.0  # mean loss over the micro-batch = the actual objective
        for example in batch:
            loss, m = self.compute_loss(example)
            (loss / len(batch)).backward()
            batch_loss += float(loss.item()) / len(batch)
            for key, val in m.items():
                agg[key] = agg.get(key, 0.0) + val / len(batch)

        assert_llm_frozen(self.backend)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.train.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self._global_step += 1

        metrics = StepMetrics(
            step=self._global_step,
            loss=batch_loss,
            grad_norm=float(grad_norm),
            lr=float(self.scheduler.get_last_lr()[0]),
            **{k: agg[k] for k in (
                "mean_reward", "faithfulness", "coverage",
                "key_sentence", "contrast", "length_penalty", "copy_penalty",
                "hallucination", "judge", "mean_len", "weight_a", "weight_b",
                "weight_c", "weight_d", "entropy",
            )},
        )
        return metrics

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, path: str, *, is_best: bool = False) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self._global_step,
            "best_reward": self.best_reward,
        }
        torch.save(payload, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(path) or ".", "best.pt")
            torch.save(payload, best_path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self._global_step = payload["step"]
        self.best_reward = payload["best_reward"]

    def maybe_update_best(self, mean_reward: float) -> bool:
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            return True
        return False
