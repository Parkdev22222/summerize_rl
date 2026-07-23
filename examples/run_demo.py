"""End-to-end SCST demo on the MockBackend (CPU, no model download).

Runs a handful of training steps over the synthetic corpus and prints per-step
metrics. Demonstrates the full pipeline: gating -> 4-branch build -> PMI decode
-> reward -> self-critical advantage -> policy update. Swap MockBackend for
HFBackend(model_name) to run on a real backbone.

    python -m examples.run_demo
"""

from __future__ import annotations

import argparse

import torch

from summarize_rl.config import Config, DecodeConfig, PolicyConfig, TrainConfig
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from examples.sample_data import EXAMPLES, MILITARY_GLOSSARY


# A tiny "vocabulary" so the MockBackend produces real Korean words. Words are
# grouped so each branch can be steered toward semantically distinct content,
# giving the reward (coverage/term) a non-trivial signal to learn from.
_WORDS = [
    "<pad>", "<eos>",                                   # 0,1
    "적중대", "145고지", "이동", "포병지원", "적기갑부대",   # 2-6  source/triplet entities
    "정찰조", "수색", "저지진지", "아군소대", "도하",        # 7-11 more entities
    "기동", "정찰", "화력지원", "방어", "점령",             # 12-16 glossary standard terms
    "그리고", "하였다", "관측", "지시", "임무",             # 17-21 filler
]
DEMO_VOCAB = {i: w for i, w in enumerate(_WORDS)}
VOCAB_SIZE = len(_WORDS)

# Steer branches: SQ(core) -> entities, GQ(gloss) -> standard terms,
# XQ(source) -> entities+filler, Q(prior) -> filler only.
DEMO_BRANCH_BIAS = {
    0: [2, 3, 4, 17, 18],       # XQ
    1: [2, 3, 4, 5, 6, 7, 8],   # SQ (triplet entities)
    2: [12, 13, 14, 15, 16],    # GQ (standard terms)
    3: [19, 20, 21],            # Q (filler prior)
}


def build_trainer(seed: int) -> SCSTTrainer:
    torch.manual_seed(seed)
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=32, hidden_dim=64)
    cfg.decode = DecodeConfig(max_new_tokens=24, min_new_tokens=6, eos_token_id=1)
    cfg.train = TrainConfig(
        num_samples=5, total_steps=20, grad_accum_steps=1, lr=2e-3, seed=seed
    )

    backend = MockBackend(
        vocab_size=VOCAB_SIZE, hidden_size=32, eos_token_id=1, seed=seed,
        vocab=DEMO_VOCAB, branch_bias=DEMO_BRANCH_BIAS,
    )
    policy = WeightPolicy(cfg.policy)
    gen = torch.Generator().manual_seed(seed)
    return SCSTTrainer(
        policy, backend, cfg, glossary=MILITARY_GLOSSARY, generator=gen
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = build_trainer(args.seed)

    print(f"{'step':>4} {'loss':>8} {'reward':>7} {'faith':>6} {'cov':>5} "
          f"{'term':>5} {'a':>5} {'b':>5} {'c':>5} {'d':>5} {'ent':>5} {'gnorm':>6}")
    for step in range(args.steps):
        example = EXAMPLES[step % len(EXAMPLES)]
        m = trainer.train_step([example])
        if trainer.maybe_update_best(m.mean_reward):
            pass
        print(f"{m.step:>4} {m.loss:>8.3f} {m.mean_reward:>7.3f} "
              f"{m.faithfulness:>6.3f} {m.coverage:>5.3f} {m.term_usage:>5.3f} "
              f"{m.weight_a:>5.2f} {m.weight_b:>5.2f} {m.weight_c:>5.2f} "
              f"{m.weight_d:>5.2f} {m.entropy:>5.2f} {m.grad_norm:>6.2f}")

    print(f"\nbest mean reward: {trainer.best_reward:.4f}")


if __name__ == "__main__":
    main()
