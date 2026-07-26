"""Real-backbone SCST training driver (single GPU).

Wires the frozen Hugging Face backbone (`HFBackend`) + a real source/triplet
corpus (JSONL) + the `SCSTTrainer` into a runnable command. Only the small
weight-policy MLP is trained; the LLM stays frozen and is used forward-only for
logits/hidden.

A 7-13B backbone in bf16 fits on a single H100 80GB, so this driver targets one
GPU. Pick the card with CUDA_VISIBLE_DEVICES. (The library does not implement
model sharding or data-parallel rollouts; using a *second* H100 to shard a
larger backbone would need `device_map` support in HFBackend.)

Example:
    CUDA_VISIBLE_DEVICES=0 python -m examples.train_real \
        --model <korean-7B-model> --dtype bfloat16 \
        --data data/scenarios_ko.jsonl \
        --steps 2000 --num-samples 5 --grad-accum 4 \
        --ckpt-dir checkpoints

Smoke test (few steps, few examples):
    CUDA_VISIBLE_DEVICES=0 python -m examples.train_real \
        --model <model> --data data/scenarios_ko.jsonl --steps 5 --limit 4
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config
from summarize_rl.glossary import Glossary
from summarize_rl.grpo import GRPOTrainer
from summarize_rl.llm_backend import HFBackend
from summarize_rl.logging_utils import make_logger
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer


def load_corpus(path: str, query: str | None, limit: int | None) -> list[Example]:
    """Read a JSONL corpus of {source_text, triplets:[[h,r,t],...]} into Examples."""
    if not os.path.exists(path):
        raise SystemExit(
            f"corpus not found: {path}\n"
            "Build it first (e.g. python data/build_dataset.py) or pass --data."
        )
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source_text") or ""
            triplets = [Triplet(*t) for t in rec.get("triplets", []) if len(t) == 3]
            kwargs = {"source": src, "triplets": triplets}
            if query:
                kwargs["query"] = query
            examples.append(Example(**kwargs))
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise SystemExit(f"no usable examples in {path}")
    return examples


def load_glossary(path: str | None) -> Glossary:
    if path:
        with open(path, encoding="utf-8") as fh:
            mapping = json.load(fh)  # {"표준용어": ["트리거", ...], ...}
        return Glossary(mapping)
    # Fallback to the small demo glossary shipped with the repo.
    from examples.sample_data import MILITARY_GLOSSARY

    return MILITARY_GLOSSARY


def main() -> None:
    p = argparse.ArgumentParser(description="Real-backbone SCST training (single GPU).")
    p.add_argument("--model", required=True, help="HF model name/path for the frozen backbone")
    p.add_argument("--data", default="data/scenarios_ko.jsonl", help="JSONL corpus")
    p.add_argument("--glossary", default=None, help="JSON {term: [triggers]}; omit for demo glossary")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--dtype", default="bfloat16", help="backbone dtype (bfloat16/float16/float32)")
    p.add_argument("--attn", default="sdpa",
                   help="attention kernel: sdpa (default, no install) | flash_attention_2 "
                        "(needs flash-attn) | eager. Auto-falls back if unavailable.")
    p.add_argument("--compile-decode", dest="compile_decode", action="store_true", default=True,
                   help="compile the single-token decode step (StaticCache/CUDA graph). On by default.")
    p.add_argument("--no-compile-decode", dest="compile_decode", action="store_false",
                   help="disable compiled decode (needed for non-StaticCache architectures)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="prompt+generation bound for the static cache when --compile-decode")
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=False,
                   help="allow custom modeling code from the HF repo (needed for EXAONE and other "
                        "models that ship their own architecture).")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false", default=True,
                   help="do NOT wrap branch prompts in the model's chat template. On by default for "
                        "instruct models (EXAONE) so they emit one clean summary and stop.")
    p.add_argument("--steps", type=int, default=None, help="override total_steps")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--num-samples", type=int, default=None, help="rollouts per input (self-critical)")
    p.add_argument("--grad-accum", type=int, default=1, help="examples per optimizer step (micro-batch size)")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--min-new-tokens", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--rl", choices=["scst", "grpo"], default="scst",
                   help="RL algorithm: scst (self-critical) | grpo (group relative PO)")
    # GRPO-only knobs (ignored under --rl scst); default None keeps GRPOConfig defaults.
    p.add_argument("--group-size", type=int, default=None, help="[grpo] rollouts per prompt (G)")
    p.add_argument("--kl-beta", type=float, default=None, help="[grpo] KL-to-reference coefficient")
    p.add_argument("--inner-epochs", type=int, default=None, help="[grpo] gradient updates per group")
    p.add_argument("--clip-eps", type=float, default=None, help="[grpo] PPO clip epsilon")
    p.add_argument("--balance-content", dest="balance_content", action="store_true", default=False,
                   help="gate fluency terms (faith/term/contrast) by content recall (cov+keysent) "
                        "to stop fluent-but-off-topic summaries from winning. Off by default.")
    p.add_argument("--w-judge", type=float, default=None,
                   help="weight of the LLM-as-judge accuracy reward (0=off, default). >0 builds a "
                        "BackboneJudge from the frozen model: catches semantic errors lexical terms "
                        "miss (e.g. 소대 vs 소총중대), at ~1 extra generation per rollout (slower).")
    p.add_argument("--keysent-n", type=int, default=None,
                   help="how many source key sentences the LLM extracts for the key-sentence "
                        "reward (RewardConfig.keysent_n; default 3). Set 0 to disable the term.")
    p.add_argument("--keysent-max-new-tokens", type=int, default=None,
                   help="generation budget for key-sentence extraction; raise it when --keysent-n "
                        "is large (default 256, ~enough for 3 sentences).")
    p.add_argument("--query", default=None, help="override the instruction/query")
    p.add_argument("--limit", type=int, default=None, help="use only the first N examples (smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--logdir", default="runs/train_real",
        help="TensorBoard run dir (empty or 'none' disables). View: tensorboard --logdir runs",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- frozen backbone -------------------------------------------------
    # flash-attn + compiled decode on by default for throughput; both fall back
    # / can be disabled (--attn, --no-compile-decode) for incompatible models.
    backend = HFBackend(
        args.model, device=args.device, dtype=args.dtype,
        attn_implementation=args.attn, compile_decode=args.compile_decode,
        max_seq_len=args.max_seq_len, trust_remote_code=args.trust_remote_code,
        use_chat_template=args.use_chat_template,
    )

    # --- config, synced to the backbone ----------------------------------
    cfg = Config()
    cfg.policy.llm_hidden_size = backend.hidden_size
    cfg.decode.eos_token_id = backend.eos_token_id
    cfg.decode.pad_token_id = backend.pad_token_id
    cfg.train.seed = args.seed
    cfg.train.ckpt_dir = args.ckpt_dir
    if args.steps is not None:
        cfg.train.total_steps = args.steps
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.num_samples is not None:
        cfg.train.num_samples = args.num_samples
    if args.save_every is not None:
        cfg.train.save_every = args.save_every
    if args.max_new_tokens is not None:
        cfg.decode.max_new_tokens = args.max_new_tokens
    if args.min_new_tokens is not None:
        cfg.decode.min_new_tokens = args.min_new_tokens
    # GRPO-specific overrides (no-ops under --rl scst).
    if args.group_size is not None:
        cfg.grpo.group_size = args.group_size
    if args.kl_beta is not None:
        cfg.grpo.kl_beta = args.kl_beta
    if args.inner_epochs is not None:
        cfg.grpo.inner_epochs = args.inner_epochs
    if args.clip_eps is not None:
        cfg.grpo.clip_eps = args.clip_eps
    cfg.reward.balance_content = args.balance_content
    if args.w_judge is not None:
        cfg.reward.w_judge = args.w_judge
    if args.keysent_n is not None:
        cfg.reward.keysent_n = args.keysent_n
    if args.keysent_max_new_tokens is not None:
        cfg.reward.keysent_max_new_tokens = args.keysent_max_new_tokens

    # --- policy on the SAME device as the backbone (stays fp32) ----------
    # HFBackend emits logits/hidden on `device`; the policy MLP must match, and
    # the sampling generator must live on the same device as the logits.
    dev = torch.device(args.device)
    policy = WeightPolicy(cfg.policy).to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    glossary = load_glossary(args.glossary)
    examples = load_corpus(args.data, args.query, args.limit)

    is_grpo = args.rl == "grpo"
    if is_grpo:
        trainer = GRPOTrainer(policy, backend, cfg, glossary=glossary, generator=gen)
    else:
        trainer = SCSTTrainer(policy, backend, cfg, glossary=glossary, generator=gen)
    logger = make_logger(args.logdir)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    rollouts = cfg.grpo.group_size if is_grpo else cfg.train.num_samples
    roll_label = "group_size" if is_grpo else "num_samples"
    print(
        f"rl={args.rl} model={args.model} dtype={args.dtype} device={args.device} "
        f"hidden={backend.hidden_size} vocab={backend.vocab_size} "
        f"examples={len(examples)} steps={cfg.train.total_steps} "
        f"{roll_label}={rollouts} grad_accum(micro-batch)={args.grad_accum}"
    )
    print(
        f"reward: balance_content={cfg.reward.balance_content} "
        f"w_judge={cfg.reward.w_judge} "
        f"chat_template={getattr(backend, 'use_chat_template', False)} "
        f"(judge {'ON' if cfg.reward.w_judge > 0 else 'off'})"
    )
    # GRPO adds kl / clip columns; the rest of the row is shared.
    extra_hdr = f" {'kl':>6} {'clip':>5}" if is_grpo else ""
    print(f"{'step':>5} {'loss':>8} {'reward':>7} {'faith':>6} {'cov':>5}"
          f"{extra_hdr} {'a':>5} {'b':>5} {'c':>5} {'d':>5} "
          f"{'gnorm':>6} {'lr':>9}")

    ga = max(1, args.grad_accum)
    for step in range(cfg.train.total_steps):
        start = (step * ga) % len(examples)
        micro = [examples[(start + i) % len(examples)] for i in range(ga)]
        m = trainer.train_step(micro)
        is_best = trainer.maybe_update_best(m.mean_reward)
        if is_best:
            trainer.save_checkpoint(os.path.join(args.ckpt_dir, "best.pt"), is_best=True)
        if cfg.train.save_every and (step + 1) % cfg.train.save_every == 0:
            trainer.save_checkpoint(os.path.join(args.ckpt_dir, f"step{step+1}.pt"))
        logger.log_metrics(m, m.step)
        if step % args.log_every == 0:
            extra = f" {m.kl:>6.3f} {m.clip_frac:>5.2f}" if is_grpo else ""
            print(f"{m.step:>5} {m.loss:>8.3f} {m.mean_reward:>7.3f} "
                  f"{m.faithfulness:>6.3f} {m.coverage:>5.3f}{extra} "
                  f"{m.weight_a:>5.2f} {m.weight_b:>5.2f} {m.weight_c:>5.2f} "
                  f"{m.weight_d:>5.2f} {m.grad_norm:>6.2f} {m.lr:>9.2e}")

    logger.close()
    print(f"\nbest mean reward: {trainer.best_reward:.4f}  (checkpoints in {args.ckpt_dir}/)")


if __name__ == "__main__":
    main()
