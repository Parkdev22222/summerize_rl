"""Train the PMI-weight policy on a real HF backbone (EXAONE) with TensorBoard.

Run on a CUDA machine that can download the model from the Hugging Face hub:

    python -m examples.train_exaone \
        --model LGAI-EXAONE/EXAONE-Deep-7.8B \
        --steps 2000 --log-dir runs/exaone --device cuda

Then watch live:

    tensorboard --logdir runs/exaone

Notes on EXAONE:
  * EXAONE ships custom modeling code -> trust_remote_code=True (set by default).
  * It is instruction/reasoning-tuned -> the tokenizer chat template is applied
    to each branch (--no-chat-template to disable).
  * EXAONE-Deep is a *reasoning* model (emits long <thought> chains). For plain
    summarization, LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct is usually a better fit;
    pass it with --model.

This script uses the small synthetic corpus/glossary in examples.sample_data as
a placeholder. Swap `load_dataset()` for the real KG source/triplet loader.
"""

from __future__ import annotations

import argparse

import torch

from summarize_rl.config import Config
from summarize_rl.llm_backend import HFBackend
from summarize_rl.logging_utils import TensorBoardLogger
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from examples.sample_data import EXAMPLES, MILITARY_GLOSSARY


def load_dataset():
    """Placeholder: return (train_examples, val_examples).

    Replace with the real corpus loader. Here we reuse the synthetic examples
    and hold out one for validation.
    """
    train = EXAMPLES
    val = EXAMPLES[:1]
    return train, val


def build_config(args) -> Config:
    cfg = Config()
    cfg.decode.max_new_tokens = args.max_new_tokens
    cfg.decode.min_new_tokens = args.min_new_tokens
    cfg.decode.temperature = args.temperature
    cfg.decode.top_p = args.top_p

    cfg.train.num_samples = args.num_samples
    cfg.train.lr = args.lr
    cfg.train.total_steps = args.steps
    cfg.train.grad_accum_steps = args.grad_accum
    cfg.train.save_every = args.save_every
    cfg.train.seed = args.seed
    cfg.train.ckpt_dir = args.ckpt_dir
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="LGAI-EXAONE/EXAONE-Deep-7.8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default=None, help='e.g. "auto" for multi-GPU')
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--no-trust-remote-code", action="store_true")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--min-new-tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--log-dir", default="runs/exaone")
    p.add_argument("--ckpt-dir", default="checkpoints/exaone")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading backbone: {args.model} ({args.dtype} on {args.device}) ...")
    backend = HFBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        use_chat_template=not args.no_chat_template,
        device_map=args.device_map,
    )
    print(f"  hidden_size={backend.hidden_size} vocab_size={backend.vocab_size} "
          f"eos={backend.eos_token_id}")

    cfg = build_config(args)
    cfg.policy.llm_hidden_size = backend.hidden_size
    cfg.policy.hidden_dim = args.hidden_dim
    cfg.decode.eos_token_id = backend.eos_token_id
    cfg.decode.pad_token_id = backend.pad_token_id

    policy = WeightPolicy(cfg.policy)  # fp32
    generator = torch.Generator().manual_seed(args.seed)
    trainer = SCSTTrainer(
        policy, backend, cfg, glossary=MILITARY_GLOSSARY, generator=generator
    )

    train_data, val_data = load_dataset()
    logger = TensorBoardLogger(args.log_dir, enabled=True)
    print(f"TensorBoard logging -> {args.log_dir}  (tensorboard --logdir {args.log_dir})")

    trainer.fit(
        train_data,
        logger=logger,
        val_dataset=val_data,
        eval_every=args.eval_every,
        log_every=args.log_every,
        print_every=args.print_every,
    )
    logger.close()
    print(f"Done. best mean reward = {trainer.best_reward:.4f}")


if __name__ == "__main__":
    main()
