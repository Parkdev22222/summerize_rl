"""Self-Critical Sequence Training for the weight policy (Section 2.5).

Only the policy network theta is updated. The LLM is frozen and its logits are
detached inside the decoder, so gradients cannot reach it. The loop:

  1. N stochastic rollouts per input.
  2. reference-free reward per rollout.
  3. self-critical baseline b (mean of the N rewards, or greedy reward).
  4. advantage A_i = R~_i - b.
  5. loss = -(1/N) sum_i A_i * sum_t (mask_t * logpi_t)   (- beta * entropy).
  6. AdamW step with grad clipping and accumulation.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .branches import Example, build_branches
from .config import Config
from .decoder import Rollout, generate
from .glossary import Glossary
from .llm_backend import LLMBackend
from .logging_utils import TensorBoardLogger
from .policy import WeightPolicy
from .rewards import (
    FaithfulnessModel,
    RewardBreakdown,
    compute_reward,
    normalize_rewards,
)


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
    term_usage: float
    length_penalty: float
    mean_len: float
    weight_a: float
    weight_b: float
    weight_c: float
    weight_d: float
    entropy: float
    grad_norm: float
    lr: float


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
    ):
        self.policy = policy
        self.backend = backend
        self.config = config
        self.glossary = glossary
        self.faithfulness_model = faithfulness_model
        self.generator = generator

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

        rollouts, breakdowns = [], []
        for _ in range(self.config.train.num_samples):
            r = generate(
                self.backend,
                branch_texts,
                self.policy,
                self.config.decode,
                greedy=False,
                generator=self.generator,
            )
            bd = compute_reward(
                summary=r.text,
                source=example.source,
                triplets=example.triplets,
                active_terms=active_terms,
                config=self.config.reward,
                faithfulness_model=self.faithfulness_model,
                summary_length=r.length,
            )
            rollouts.append(r)
            breakdowns.append(bd)
        return rollouts, breakdowns, active_terms

    def _greedy_score(self, example: Example) -> tuple[Rollout, RewardBreakdown]:
        """Greedy (deterministic) generation + reward, no gradient."""
        active = self.glossary.gate(example.source) if self.glossary else []
        active_terms = [a.term for a in active]
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
        )
        return r, bd

    def _greedy_reward(self, example: Example, active_terms: list[str]) -> float:
        _, bd = self._greedy_score(example)
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

        loss = torch.zeros((), dtype=torch.float32)
        for adv, rollout in zip(advantages, rollouts):
            loss = loss - adv * rollout.sum_logp()
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
            "term_usage": sum(bd.term_usage for bd in breakdowns) / k,
            "length_penalty": sum(bd.length_penalty for bd in breakdowns) / k,
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
        last_loss = 0.0
        for example in batch:
            loss, m = self.compute_loss(example)
            (loss / len(batch)).backward()
            last_loss = float(loss.item())
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
            loss=last_loss,
            grad_norm=float(grad_norm),
            lr=float(self.scheduler.get_last_lr()[0]),
            **{k: agg[k] for k in (
                "mean_reward", "faithfulness", "coverage", "term_usage",
                "length_penalty", "mean_len", "weight_a", "weight_b",
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

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataset: Sequence[Example]) -> dict[str, float]:
        """Greedy-decode the dataset and average reward components.

        Used for the periodic validation hook (Section 2.5.9) to watch for
        collapse/over-fitting. Returns a dict of mean metrics.
        """
        was_training = self.policy.training
        self.policy.eval()
        agg = {
            "mean_reward": 0.0, "faithfulness": 0.0, "coverage": 0.0,
            "term_usage": 0.0, "length_penalty": 0.0, "mean_len": 0.0,
        }
        n = 0
        for example in dataset:
            rollout, bd = self._greedy_score(example)
            agg["mean_reward"] += bd.total
            agg["faithfulness"] += bd.faithfulness
            agg["coverage"] += bd.coverage
            agg["term_usage"] += bd.term_usage
            agg["length_penalty"] += bd.length_penalty
            agg["mean_len"] += rollout.length
            n += 1
        if was_training:
            self.policy.train()
        if n == 0:
            return agg
        return {k: v / n for k, v in agg.items()}

    # -- high-level training driver ----------------------------------------

    def _batch_iterator(
        self, dataset: Sequence[Example], total_steps: int
    ) -> Iterable[list[Example]]:
        """Yield `total_steps` micro-batches of size grad_accum_steps.

        Cycles the dataset with a seeded shuffle each pass for reproducibility.
        """
        accum = self.config.train.grad_accum_steps
        rng = random.Random(self.config.train.seed)
        order: list[int] = []

        def refill() -> None:
            idx = list(range(len(dataset)))
            rng.shuffle(idx)
            order.extend(idx)

        for _ in range(total_steps):
            batch = []
            for _ in range(accum):
                if not order:
                    refill()
                batch.append(dataset[order.pop(0)])
            yield batch

    def fit(
        self,
        dataset: Sequence[Example],
        *,
        logger: TensorBoardLogger | None = None,
        val_dataset: Sequence[Example] | None = None,
        eval_every: int = 0,
        log_every: int = 1,
        print_every: int = 10,
        max_steps: int | None = None,
    ) -> None:
        """Run the SCST training loop with logging, eval, and checkpointing."""
        t = self.config.train
        total_steps = max_steps if max_steps is not None else t.total_steps

        for batch in self._batch_iterator(dataset, total_steps):
            metrics = self.train_step(batch)
            step = metrics.step

            if logger is not None and step % max(1, log_every) == 0:
                logger.log_metrics(metrics)

            if print_every and step % print_every == 0:
                print(
                    f"step {step:>5} | loss {metrics.loss:8.3f} | "
                    f"R {metrics.mean_reward:6.3f} | faith {metrics.faithfulness:.3f} "
                    f"cov {metrics.coverage:.3f} term {metrics.term_usage:.3f} | "
                    f"a{metrics.weight_a:.2f} b{metrics.weight_b:.2f} "
                    f"c{metrics.weight_c:.2f} d{metrics.weight_d:.2f} | "
                    f"gnorm {metrics.grad_norm:.2f} lr {metrics.lr:.2e}"
                )

            is_best = self.maybe_update_best(metrics.mean_reward)

            if val_dataset is not None and eval_every and step % eval_every == 0:
                ev = self.evaluate(val_dataset)
                if logger is not None:
                    logger.log_eval(step, ev)
                print(f"  [eval @ {step}] " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))

            if t.save_every and step % t.save_every == 0:
                self.save_checkpoint(
                    os.path.join(t.ckpt_dir, f"step_{step}.pt"), is_best=is_best
                )
            elif is_best:
                self.save_checkpoint(
                    os.path.join(t.ckpt_dir, "best.pt"), is_best=False
                )

        if logger is not None:
            logger.flush()
