"""End-to-end GRPO demo on the MockBackend (CPU, no model download).

Same pipeline as run_demo.py (gating -> 4-branch build -> PMI decode -> reward)
but the RL objective is Group Relative Policy Optimization instead of SCST:
group-normalized advantages, a PPO-clipped importance ratio over `inner_epochs`
updates per sampled group, and a KL penalty to a frozen reference policy.

    python -m examples.run_grpo_demo --steps 20

Extra columns vs. the SCST demo: `kl` (mean KL to the reference policy) and
`clip` (fraction of tokens whose ratio hit the clip band). As with SCST, judge
progress by the smoothed reward (rew_ema) and the weight trajectory -- not by
the per-step loss.
"""

from __future__ import annotations

import argparse

import torch

from summarize_rl.config import Config, DecodeConfig, GRPOConfig, PolicyConfig, TrainConfig
from summarize_rl.grpo import GRPOTrainer
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from examples.sample_data import EXAMPLES, MILITARY_GLOSSARY
from examples.run_demo import DEMO_BRANCH_BIAS, DEMO_VOCAB, VOCAB_SIZE


def build_trainer(seed: int) -> GRPOTrainer:
    torch.manual_seed(seed)
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=32, hidden_dim=64)
    cfg.decode = DecodeConfig(max_new_tokens=24, min_new_tokens=6, eos_token_id=1)
    cfg.train = TrainConfig(total_steps=20, grad_accum_steps=1, lr=2e-3, seed=seed)
    cfg.grpo = GRPOConfig(group_size=8, clip_eps=0.2, kl_beta=0.04, inner_epochs=2)

    backend = MockBackend(
        vocab_size=VOCAB_SIZE, hidden_size=32, eos_token_id=1, seed=seed,
        vocab=DEMO_VOCAB, branch_bias=DEMO_BRANCH_BIAS,
    )
    policy = WeightPolicy(cfg.policy)
    gen = torch.Generator().manual_seed(seed)
    return GRPOTrainer(
        policy, backend, cfg, glossary=MILITARY_GLOSSARY, generator=gen
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = build_trainer(args.seed)

    print(f"{'step':>4} {'loss':>8} {'reward':>7} {'rew_ema':>8} {'kl':>6} "
          f"{'clip':>5} {'faith':>6} {'cov':>5} {'term':>5} {'ctr':>6} "
          f"{'a':>5} {'b':>5} {'c':>5} {'d':>5} {'gnorm':>6}")
    rew_ema = None
    for step in range(args.steps):
        example = EXAMPLES[step % len(EXAMPLES)]
        m = trainer.train_step([example])
        trainer.maybe_update_best(m.mean_reward)
        rew_ema = m.mean_reward if rew_ema is None else 0.9 * rew_ema + 0.1 * m.mean_reward
        print(f"{m.step:>4} {m.loss:>8.3f} {m.mean_reward:>7.3f} {rew_ema:>8.3f} "
              f"{m.kl:>6.3f} {m.clip_frac:>5.2f} {m.faithfulness:>6.3f} "
              f"{m.coverage:>5.3f} {m.term_usage:>5.3f} {m.contrast:>6.3f} "
              f"{m.weight_a:>5.2f} {m.weight_b:>5.2f} {m.weight_c:>5.2f} "
              f"{m.weight_d:>5.2f} {m.grad_norm:>6.2f}")

    print(f"\nbest mean reward: {trainer.best_reward:.4f}")


if __name__ == "__main__":
    main()
