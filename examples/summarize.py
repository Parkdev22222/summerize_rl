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

from summarize_rl.branches import Example
from summarize_rl.config import Config
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer, build_hf_summarizer

DEFAULT_QUERY = Example.__dataclass_fields__["query"].default

HELP = """\
명령어:
  (원문 붙여넣기 후 Enter 2번=빈 줄 2번)  붙여넣은 원문을 요약
       ※ 문단 사이 한 줄 공백은 원문의 일부로 유지됩니다(하나의 원문). 끝에 빈 줄 2번.
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
    """Read a multi-line source block, submitted by two blank lines (Enter x2).

    A *single* blank line is kept as a paragraph break — real reports (and ones
    ending in "끝.") contain blank lines between sections, and submitting on the
    first one split one pasted source into several, producing a summary per
    paragraph. The block is submitted only on TWO consecutive blank lines or EOF
    (Ctrl-D), and neither collides with report punctuation, so one paste is one
    source.

    Returns:
      * ``None`` on EOF (Ctrl-D) at the start,
      * ``[":command", "args"]`` when the first non-empty line is a command,
      * the joined source text otherwise.
    """
    lines: list[str] = []
    first = True
    blanks = 0
    while True:
        try:
            line = input(prompt if first else "")
        except EOFError:
            if first:
                return None
            break  # Ctrl-D also submits what's been typed
        stripped = line.strip()
        if first and stripped.startswith(":"):
            return stripped.split(maxsplit=1)
        if stripped == "":
            if first:
                continue  # ignore leading blank lines; keep waiting
            blanks += 1
            if blanks >= 2:
                break  # two consecutive blank lines -> submit
            lines.append(line)  # keep a single blank as a paragraph break
            continue
        blanks = 0
        lines.append(line)
        first = False
    return "\n".join(lines).strip()


def print_summary(result) -> None:
    a, b, c, d = result.mean_weights
    print(f"\n[요약] {result.text}")
    terms = ", ".join(result.active_terms) if result.active_terms else "(없음)"
    print(f"[표준용어] {terms}  [가중치 a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f}]\n")


def build_summarizer(args) -> tuple[Summarizer, Config]:
    glossary = load_glossary(args.glossary)
    summarizer, cfg, step = build_hf_summarizer(
        args.model,
        args.ckpt,
        glossary=glossary,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        compile_decode=args.compile,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=args.use_chat_template,
        extract_triplets=args.extract_triplets,
    )
    print(
        f"model={args.model} dtype={args.dtype} device={args.device} "
        f"attn={summarizer.backend.attn_implementation} compile={args.compile} "
        f"hidden={summarizer.backend.hidden_size} ckpt={args.ckpt} (step={step})"
    )
    if args.compile:
        print("[note] --compile: 첫 요약은 컴파일 때문에 느립니다(이후 빨라짐).")
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
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="attention kernel (sdpa=safe fast default; flash_attention_2 fastest, needs flash-attn)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile + StaticCache decode (CUDA-graph step; needs Llama/Qwen-like model; first call slow)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="static cache/mask length bound for --compile (prompt+generation)")
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=False,
                   help="allow custom modeling code from the HF repo (needed for EXAONE etc.)")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false", default=True,
                   help="do NOT wrap prompts in the model's chat template (on by default).")
    p.add_argument("--no-extract-triplets", dest="extract_triplets", action="store_false", default=True,
                   help="do NOT extract triplets from the source at inference. On by default: the "
                        "SQ-heavy trained policy fabricates when the SQ branch is empty, so the LLM "
                        "extracts triplets to fill it (matching training).")
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
