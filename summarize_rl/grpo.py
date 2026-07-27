"""Group Relative Policy Optimization for the weight policy.

Everything is identical to the SCST setup (frozen LLM, 4-branch PMI decoding,
the PMI-weight MLP policy, and the reference-free reward including the PMI
contrast term) EXCEPT the RL objective, which is GRPO instead of self-critical
REINFORCE:

  1. G rollouts per prompt (a "group").
  2. reference-free reward per rollout.
  3. group-relative advantage  A_i = (r_i - mean_G) / (std_G + eps),
     broadcast to every token of rollout i  (no critic, no greedy baseline).
  4. PPO-clipped surrogate with the importance ratio
        rho_{i,t} = pi_theta(o_{i,t}) / pi_theta_old(o_{i,t})
        L_clip = mean_t min(rho * A_i, clip(rho, 1-eps, 1+eps) * A_i)
     letting the same sampled group drive `inner_epochs` gradient steps.
  5. token-level KL penalty to a frozen reference policy (k3 estimator):
        kl_{i,t} = exp(r) - r - 1,   r = logp_ref - logp_theta
  6. loss = -(1/G) sum_i mean_t (L_clip - kl_beta * kl)  (- entropy bonus).

Only the policy MLP theta is trained; the LLM stays frozen and its logits are
detached inside the decoder. Dropout is disabled during GRPO so that the
importance ratio pi_theta/pi_theta_old is exact (a stochastic policy would make
pi_theta_old ill-defined across the inner epochs).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace

import torch
from torch.optim import AdamW

from .branches import Example, build_branches
from .config import Config
from .decoder import Rollout, generate_batch, score_tokens_batch
from .glossary import Glossary
from .judge import BackboneJudge
from .keysent import KeySentenceExtractor
from .llm_backend import LLMBackend
from .policy import WeightPolicy
from .rewards import FaithfulnessModel, RewardBreakdown, compute_reward
from .train import assert_llm_frozen, build_scheduler, freeze_llm


@dataclass
class GRPOMetrics:
    step: int
    loss: float
    mean_reward: float
    faithfulness: float
    coverage: float
    key_sentence: float
    contrast: float
    length_penalty: float
    copy_penalty: float
    hallucination: float
    judge: float
    mean_len: float
    weight_a: float
    weight_b: float
    weight_c: float
    weight_d: float
    kl: float
    clip_frac: float
    entropy: float
    grad_norm: float
    lr: float


@dataclass
class _Group:
    """One prompt's sampled rollouts plus the frozen data GRPO reuses."""

    branch_texts: dict
    rollouts: list[Rollout]
    breakdowns: list[RewardBreakdown]
    advantages: list[float]  # group-normalized, one scalar per rollout
    old_logps: list[torch.Tensor]  # per rollout: [T] detached
    ref_logps: list[torch.Tensor]  # per rollout: [T] detached


class GRPOTrainer:
    def __init__(
        self,
        policy: WeightPolicy,
        backend: LLMBackend,
        config: Config,
        *,
        glossary: Glossary | None = None,
        faithfulness_model: FaithfulnessModel | None = None,
        judge_model: object | None = None,
        generator: torch.Generator | None = None,
    ):
        self.policy = policy
        self.backend = backend
        self.config = config
        self.glossary = glossary
        self.faithfulness_model = faithfulness_model
        self.generator = generator
        self.key_extractor = (
            KeySentenceExtractor(config.reward) if config.reward.w_keysent > 0 else None
        )
        # An injected judge (e.g. GeminiJudge) wins; else build the local
        # BackboneJudge when the judge reward is enabled.
        self.judge = judge_model if judge_model is not None else (
            BackboneJudge(backend, config.reward) if config.reward.w_judge > 0 else None
        )
        # RAW (frozen-LLM) reference summary per source, for the comparative
        # judge ("beat RAW"). Generated once per source and cached.
        self._raw_cache: dict[str, str] = {}

        freeze_llm(backend)

        # Frozen reference policy pi_ref = the initial (warm-started) policy.
        self.ref_policy = copy.deepcopy(policy)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # GRPO scores with the full softmax; sample without nucleus truncation
        # so old/new/ref log-probs share one well-defined distribution.
        self.decode = replace(config.decode, top_p=1.0)

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

    # -- sampling: one group of G rollouts under pi_theta_old ----------------

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

    def _sample_group(self, example: Example) -> _Group:
        g = self.config.grpo
        active = self.glossary.gate(example.source) if self.glossary else []
        active_terms = [a.term for a in active]
        branch_texts = build_branches(example, active).as_dict()
        key_sents = (
            self.key_extractor.extract(self.backend, example.source)
            if self.key_extractor else None
        )
        raw_ref = self._raw_reference(branch_texts, example.source)

        # Sampling and old/ref scoring run without gradient and without dropout.
        # All G rollouts share this prompt, so they are generated in one batched
        # forward (batch = G*4) instead of G sequential single-stream decodes.
        self.policy.eval()
        with torch.no_grad():
            rollouts = generate_batch(
                self.backend, branch_texts, self.policy, self.decode,
                n=g.group_size, greedy=False, generator=self.generator,
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

            rewards = [bd.total for bd in breakdowns]
            advantages = self._group_advantages(rewards)

            # pi_theta_old and pi_ref log-probs are fixed for all inner epochs;
            # score the whole group in one batched pass under each policy.
            token_ids_list = [r.token_ids for r in rollouts]
            old_scored = score_tokens_batch(
                self.backend, branch_texts, self.policy, token_ids_list, self.decode
            )
            ref_scored = score_tokens_batch(
                self.backend, branch_texts, self.ref_policy, token_ids_list, self.decode
            )
            old_logps = [torch.stack(s.logps).detach() for s in old_scored]
            ref_logps = [torch.stack(s.logps).detach() for s in ref_scored]

        return _Group(
            branch_texts=branch_texts,
            rollouts=rollouts,
            breakdowns=breakdowns,
            advantages=advantages,
            old_logps=old_logps,
            ref_logps=ref_logps,
        )

    def _group_advantages(self, rewards: list[float]) -> list[float]:
        n = len(rewards)
        mu = sum(rewards) / n
        var = sum((r - mu) ** 2 for r in rewards) / n
        sigma = var ** 0.5
        eps = self.config.grpo.adv_eps
        return [(r - mu) / (sigma + eps) for r in rewards]

    # -- one clipped surrogate pass over a group (current policy) ------------

    def _surrogate_loss(self, group: _Group) -> tuple[torch.Tensor, dict]:
        g = self.config.grpo
        # Re-score the whole group under the current policy in one batched pass.
        token_ids_list = [r.token_ids for r in group.rollouts]
        new_scored = score_tokens_batch(
            self.backend, group.branch_texts, self.policy, token_ids_list, self.decode
        )
        loss = torch.zeros((), dtype=torch.float32)
        kl_sum, clip_sum, ent_sum, tok_total = 0.0, 0.0, 0.0, 0
        for scored, adv, old_lp, ref_lp in zip(
            new_scored, group.advantages, group.old_logps, group.ref_logps
        ):
            new_lp = torch.stack(scored.logps)  # [T], grad
            ratio = torch.exp(new_lp - old_lp)  # [T]

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - g.clip_eps, 1.0 + g.clip_eps) * adv
            surrogate = torch.min(unclipped, clipped)  # [T]

            log_r = ref_lp - new_lp
            kl = torch.exp(log_r) - log_r - 1.0  # [T], k3 estimator >= 0

            ent = torch.stack(scored.entropies)  # [T]
            per_token = surrogate - g.kl_beta * kl + g.entropy_beta * ent
            loss = loss - per_token.mean()

            with torch.no_grad():
                kl_sum += float(kl.mean())
                clip_sum += float(
                    ((ratio > 1.0 + g.clip_eps) | (ratio < 1.0 - g.clip_eps)).float().mean()
                )
                ent_sum += float(ent.mean())
                tok_total += 1
        loss = loss / len(group.rollouts)
        stats = {
            "kl": kl_sum / max(tok_total, 1),
            "clip_frac": clip_sum / max(tok_total, 1),
            "entropy": ent_sum / max(tok_total, 1),
        }
        return loss, stats

    def _reward_metrics(self, groups: list[_Group]) -> dict:
        breakdowns = [bd for gr in groups for bd in gr.breakdowns]
        rollouts = [r for gr in groups for r in gr.rollouts]
        weights = [w for r in rollouts for w in r.weight_trace]
        nw = max(1, len(weights))
        k = len(breakdowns)
        return {
            "mean_reward": sum(bd.total for bd in breakdowns) / k,
            "faithfulness": sum(bd.faithfulness for bd in breakdowns) / k,
            "coverage": sum(bd.coverage for bd in breakdowns) / k,
            "key_sentence": sum(bd.key_sentence for bd in breakdowns) / k,
            "hallucination": sum(bd.hallucination for bd in breakdowns) / k,
            "judge": sum(bd.judge for bd in breakdowns) / k,
            "contrast": sum(bd.contrast for bd in breakdowns) / k,
            "length_penalty": sum(bd.length_penalty for bd in breakdowns) / k,
            "copy_penalty": sum(bd.copy_penalty for bd in breakdowns) / k,
            "mean_len": sum(r.length for r in rollouts) / len(rollouts),
            "weight_a": sum(w[0] for w in weights) / nw,
            "weight_b": sum(w[1] for w in weights) / nw,
            "weight_c": sum(w[2] for w in weights) / nw,
            "weight_d": sum(w[3] for w in weights) / nw,
        }

    # -- one optimizer iteration: sample once, update inner_epochs times ------

    def train_step(self, batch: list[Example]) -> GRPOMetrics:
        g = self.config.grpo
        groups = [self._sample_group(ex) for ex in batch]
        reward_m = self._reward_metrics(groups)

        last_loss, last_stats, grad_norm = 0.0, {}, 0.0
        for _ in range(g.inner_epochs):
            self.policy.eval()  # keep dropout off so the ratio stays exact
            self.optimizer.zero_grad()
            batch_loss = 0.0
            agg = {"kl": 0.0, "clip_frac": 0.0, "entropy": 0.0}
            for group in groups:
                loss, stats = self._surrogate_loss(group)
                (loss / len(groups)).backward()
                batch_loss += float(loss.item()) / len(groups)
                for key in agg:
                    agg[key] += stats[key] / len(groups)

            assert_llm_frozen(self.backend)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.train.grad_clip
                )
            )
            self.optimizer.step()
            last_loss, last_stats = batch_loss, agg

        self.scheduler.step()
        self._global_step += 1

        return GRPOMetrics(
            step=self._global_step,
            loss=last_loss,
            grad_norm=grad_norm,
            lr=float(self.scheduler.get_last_lr()[0]),
            kl=last_stats["kl"],
            clip_frac=last_stats["clip_frac"],
            entropy=last_stats["entropy"],
            **{k: reward_m[k] for k in (
                "mean_reward", "faithfulness", "coverage",
                "key_sentence", "contrast", "length_penalty", "copy_penalty",
                "hallucination", "judge", "mean_len", "weight_a", "weight_b",
                "weight_c", "weight_d",
            )},
        )

    # -- checkpointing (same payload shape as SCSTTrainer) -------------------

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
