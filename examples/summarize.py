"""Interactive summarization REPL over a frozen backbone + trained checkpoint.

Loads the small weight-policy checkpoint produced by training (default
``checkpoints/best.pt``) onto a frozen Hugging Face backbone, then reads source
text from the terminal and prints a summary for each input.

The "query" is the summarization *instruction* (default: the standard
military-terminology instruction); change it live with ``:query``.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m examples.summarize \
        --model <korean-7B-model> --dtype bfloat16 \
        --ckpt checkpoints/best.pt

REPL:
    paste the source text, then an empty line to summarize it.
    :query <instruction>   change the summarization instruction
    :ckpt <path>           reload a different checkpoint
    :help                  show commands
    :q                     quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from summarize_rl.branches import Example
from summarize_rl.config import Config
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer
from summarize_rl.llm_backend import HFBackend
from summarize_rl.policy import WeightPolicy

DEFAULT_QUERY = Example.__dataclass_fields__["query"].default

HELP = """\
명령어:
  (원문 붙여넣기 후 빈 줄)  붙여넣은 원문을 요약
  :query <지시문>           요약 지시문(query) 변경
  :ckpt <경로>              다른 체크포인트 가중치 재로드
  :help                     이 도움말
  :q / :quit / :exit        종료
"""


def load_glossary(path: str | None) -> Glossary:
    if path:
        with open(path, encoding="utf-8") as fh:
            mapping = json.load(fh)  # {"표준용어": ["트리거", ...], ...}
        return Glossary(mapping)
    # Fallback to the small demo glossary shipped with the repo.
    from examples.sample_data import MILITARY_GLOSSARY

    return MILITARY_GLOSSARY


def read_source(prompt: str = "원문> ") -> str | list[str] | None:
    """Read a multi-line source block (ends on a blank line) or a ``:`` command.

    Returns:
      * ``None`` on EOF (Ctrl-D),
      * ``[":command", "args"]`` when the first non-empty line is a command,
      * the joined source text otherwise.
    """
    lines: list[str] = []
    first = True
    while True:
        try:
            line = input(prompt if first else "")
        except EOFError:
            if first:
                return None
            break
        if first and line.strip().startswith(":"):
            return line.strip().split(maxsplit=1)
        if line.strip() == "":
            if first:
                # Ignore leading blank lines; keep waiting for input.
                continue
            break
        lines.append(line)
        first = False
    return "\n".join(lines)


def print_summary(result) -> None:
    a, b, c, d = result.mean_weights
    print(f"\n[요약] {result.text}")
    terms = ", ".join(result.active_terms) if result.active_terms else "(없음)"
    print(f"[표준용어] {terms}  [가중치 a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f}]\n")


def build_summarizer(args) -> tuple[Summarizer, Config]:
    backend = HFBackend(args.model, device=args.device, dtype=args.dtype)

    cfg = Config()
    cfg.policy.llm_hidden_size = backend.hidden_size
    cfg.decode.eos_token_id = backend.eos_token_id
    cfg.decode.pad_token_id = backend.pad_token_id
    if args.max_new_tokens is not None:
        cfg.decode.max_new_tokens = args.max_new_tokens
    if args.min_new_tokens is not None:
        cfg.decode.min_new_tokens = args.min_new_tokens

    dev = torch.device(args.device)
    policy = WeightPolicy(cfg.policy).to(dev)
    glossary = load_glossary(args.glossary)
    summarizer = Summarizer(backend, policy, cfg, glossary=glossary)

    step = summarizer.load_checkpoint(args.ckpt)
    print(
        f"model={args.model} dtype={args.dtype} device={args.device} "
        f"hidden={backend.hidden_size} ckpt={args.ckpt} (step={step})"
    )
    return summarizer, cfg


def repl(summarizer: Summarizer, query: str) -> None:
    print(HELP)
    print(f"[query] {query}\n")
    while True:
        block = read_source()
        if block is None:
            print("\n종료합니다.")
            return
        if isinstance(block, list):
            cmd = block[0].lower()
            arg = block[1] if len(block) > 1 else ""
            if cmd in (":q", ":quit", ":exit"):
                print("종료합니다.")
                return
            if cmd == ":help":
                print(HELP)
                continue
            if cmd == ":query":
                if arg:
                    query = arg
                    print(f"[query 변경됨] {query}\n")
                else:
                    print(f"[query] {query}\n")
                continue
            if cmd == ":ckpt":
                if not arg:
                    print("사용법: :ckpt <경로>\n")
                    continue
                try:
                    step = summarizer.load_checkpoint(arg)
                    print(f"[체크포인트 재로드됨] {arg} (step={step})\n")
                except (FileNotFoundError, ValueError, RuntimeError) as e:
                    print(f"[오류] {e}\n")
                continue
            print(f"알 수 없는 명령어: {cmd}  (:help 참고)\n")
            continue

        source = block.strip()
        if not source:
            continue
        result = summarizer.summarize(source, query=query)
        print_summary(result)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive summarization REPL (frozen backbone + trained checkpoint)."
    )
    p.add_argument("--model", required=True, help="HF model name/path for the frozen backbone")
    p.add_argument("--ckpt", default=os.path.join("checkpoints", "best.pt"),
                   help="policy checkpoint to load (default: checkpoints/best.pt)")
    p.add_argument("--glossary", default=None, help="JSON {term: [triggers]}; omit for demo glossary")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--dtype", default="bfloat16", help="backbone dtype (bfloat16/float16/float32)")
    p.add_argument("--query", default=DEFAULT_QUERY, help="summarization instruction")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--min-new-tokens", type=int, default=None)
    args = p.parse_args()

    try:
        summarizer, _ = build_summarizer(args)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    try:
        repl(summarizer, args.query)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
