"""Localize "military but wrong-content" summaries: base model vs policy vs reward.

The weight policy can only *reweight* the four branch distributions; it cannot
make the frozen LLM produce content the LLM doesn't generate. So when trained
summaries stay off-topic, the cause is one of three layers. This tool prints,
for one source, three outputs side by side so you can tell which layer is at
fault:

  1. RAW base model  — backend.generate_text on the XQ prompt (source+instruction).
     What the frozen LLM does on its own, with no PMI mixing and no policy.
  2. NEUTRAL policy  — a fresh, untrained policy (a=0.5, b=c=d=1/3) greedy decode.
  3. TRAINED policy  — the loaded checkpoint greedy decode.

Reading it:
  * (1) already off-topic  -> the frozen backbone / prompt is the ceiling. No
    reward or policy change fixes this; use a stronger/instruction-tuned model
    or improve the branch prompts (branches.build_branches).
  * (1) ok but (2)/(3) off -> the PMI decode / weights are hurting.
  * (2) ok but (3) off     -> training pushed the weights the wrong way (reward
    still gameable on this source).

    python -m examples.diagnose --model <M> --ckpt checkpoints/best.pt --index 0
"""

from __future__ import annotations

import argparse
import json

import torch

from summarize_rl.branches import Example, Triplet, build_branches
from summarize_rl.config import Config
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer
from summarize_rl.keysent import KeySentenceExtractor
from summarize_rl.llm_backend import LLMBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.rewards import compute_reward


def _reward_line(summary, example, active_terms, cfg, key_sents):
    bd = compute_reward(
        summary=summary, source=example.source, triplets=example.triplets,
        active_terms=active_terms, config=cfg.reward, key_sentences=key_sents,
        contrast=0.0,
    )
    return (f"faith={bd.faithfulness:.2f} cov={bd.coverage:.2f} "
            f"keysent={bd.key_sentence:.2f} hallu={bd.hallucination:.2f} "
            f"copy={bd.copy_penalty:.2f} total={bd.total:.3f}")


def run_diagnosis(
    backend: LLMBackend, example: Example, cfg: Config,
    glossary: Glossary | None, ckpt_path: str | None,
) -> None:
    active = glossary.gate(example.source) if glossary else []
    active_terms = [a.term for a in active]
    branch_texts = build_branches(example, active).as_dict()

    key_sents = None
    if cfg.reward.w_keysent > 0:
        key_sents = KeySentenceExtractor(cfg.reward).extract(backend, example.source)
        print("── 핵심 문장 (군사 중요도 가중) ──")
        for i, item in enumerate(key_sents, 1):
            sent, weight = item if isinstance(item, (tuple, list)) else (item, 1.0)
            print(f"  {i}. [{weight:.1f}] {sent}")
        print()

    # 1) RAW base model: feed the XQ prompt straight to the frozen LLM.
    raw = backend.generate_text(branch_texts["XQ"], cfg.decode.max_new_tokens)
    print("① RAW base 모델 (원문+지시 → LLM 단독, PMI 없음)")
    print(f"   {raw.strip()[:400]}")
    print(f"   [{_reward_line(raw, example, active_terms, cfg, key_sents)}]\n")

    # 2) NEUTRAL policy (fresh, untrained).
    neutral_policy = WeightPolicy(cfg.policy)
    neu = Summarizer(backend, neutral_policy, cfg, glossary=glossary).summarize(
        example.source, triplets=example.triplets)
    print("② NEUTRAL 정책 (미학습, a=0.5 b=c=d=1/3)")
    print(f"   {neu.text.strip()[:400]}")
    print(f"   weights(a,b,c,d)={tuple(round(w,2) for w in neu.mean_weights)}")
    print(f"   [{_reward_line(neu.text, example, active_terms, cfg, key_sents)}]\n")

    # 3) TRAINED policy (loaded checkpoint).
    if ckpt_path:
        trained_policy = WeightPolicy(cfg.policy)
        s = Summarizer(backend, trained_policy, cfg, glossary=glossary)
        step = s.load_checkpoint(ckpt_path)
        tr = s.summarize(example.source, triplets=example.triplets)
        print(f"③ TRAINED 정책 (ckpt={ckpt_path}, step={step})")
        print(f"   {tr.text.strip()[:400]}")
        print(f"   weights(a,b,c,d)={tuple(round(w,2) for w in tr.mean_weights)}")
        print(f"   [{_reward_line(tr.text, example, active_terms, cfg, key_sents)}]\n")
    else:
        print("③ TRAINED 정책 — --ckpt 미지정으로 건너뜀\n")


def main() -> None:
    from summarize_rl.llm_backend import HFBackend
    from examples.sample_data import MILITARY_GLOSSARY

    p = argparse.ArgumentParser(description="Localize off-topic summaries.")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/scenarios_ko.jsonl")
    p.add_argument("--index", type=int, default=0, help="which corpus record")
    p.add_argument("--ckpt", default=None, help="trained checkpoint (optional)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--glossary", default=None)
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=False,
                   help="allow custom modeling code from the HF repo (needed for EXAONE etc.)")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false", default=True,
                   help="do NOT wrap prompts in the model's chat template (on by default).")
    args = p.parse_args()

    torch.manual_seed(0)
    backend = HFBackend(args.model, device=args.device, dtype=args.dtype,
                        trust_remote_code=args.trust_remote_code,
                        use_chat_template=args.use_chat_template)
    cfg = Config()
    cfg.policy.llm_hidden_size = backend.hidden_size
    cfg.decode.eos_token_id = backend.eos_token_id
    cfg.decode.pad_token_id = backend.pad_token_id

    with open(args.data, encoding="utf-8") as fh:
        rec = json.loads(list(fh)[args.index])
    example = Example(
        source=rec["source_text"],
        triplets=[Triplet(*t) for t in rec.get("triplets", []) if len(t) == 3],
    )
    if args.glossary:
        with open(args.glossary, encoding="utf-8") as fh:
            glossary = Glossary(json.load(fh))
    else:
        glossary = MILITARY_GLOSSARY

    print(f"원문(앞 120자): {example.source[:120]}...\n")
    run_diagnosis(backend, example, cfg, glossary, args.ckpt)


if __name__ == "__main__":
    main()
