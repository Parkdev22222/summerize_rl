"""Terminal REPL client for the summarization server (examples/serve.py).

Thin client: keeps no model, just POSTs source text to a running server and
prints the answer. The heavy backbone stays warm in the server process (pinned
to one GPU), so you only "check the answers" here.

Example:
    # 1) start the server (on the GPU box), pinning one card:
    CUDA_VISIBLE_DEVICES=3 python -m examples.serve --model <model> --ckpt checkpoints/best.pt
    # 2) talk to it from a terminal:
    python -m examples.summarize_client --server http://127.0.0.1:8000

REPL:
    paste the source text, then an empty line to summarize it.
    :query <instruction>   change the summarization instruction (client-side)
    :ckpt <path>           ask the server to reload a different checkpoint
    :help                  show commands
    :q                     quit
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from summarize_rl.infer import SummaryResult

from examples.summarize import DEFAULT_QUERY, print_summary, read_source

HELP = """\
명령어:
  (원문 붙여넣기 후, 마침표(.)만 있는 줄)  원문을 서버에 보내 요약
       ※ 빈 줄로는 제출되지 않습니다(문단 사이 빈 줄이 있어도 하나의 원문). 끝에 . 한 줄.
  :query <지시문>           요약 지시문(query) 변경 (클라이언트 측)
  :baseline <on|off>        순수 LLM(제안 방식 아님) 요약도 함께 표시 (기본 on)
  :ckpt <경로>              서버가 다른 체크포인트를 재로드하도록 요청
  :help                     이 도움말
  :q / :quit / :exit        종료
"""


def post(server: str, path: str, payload: dict) -> dict:
    """POST JSON to the server; returns the decoded response.

    Raises RuntimeError with the server's error message on a 4xx/5xx.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise RuntimeError(msg) from None


def check_health(server: str) -> None:
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=5) as resp:
            json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(
            f"서버에 연결할 수 없습니다: {server}\n"
            f"먼저 서버를 띄우세요 (python -m examples.serve ...). ({e})"
        )


def repl(server: str, query: str) -> None:
    print(HELP)
    show_baseline = True
    print(f"[server] {server}   [query] {query}   [baseline] on\n")
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
            if cmd == ":baseline":
                val = arg.strip().lower()
                if val in ("on", "off"):
                    show_baseline = val == "on"
                    print(f"[baseline {'on' if show_baseline else 'off'}]\n")
                else:
                    print(f"[baseline] {'on' if show_baseline else 'off'}  "
                          "(사용법: :baseline on|off)\n")
                continue
            if cmd == ":ckpt":
                if not arg:
                    print("사용법: :ckpt <경로>\n")
                    continue
                try:
                    resp = post(server, "/reload", {"ckpt": arg})
                    print(f"[체크포인트 재로드됨] {resp['ckpt']} (step={resp['step']})\n")
                except RuntimeError as e:
                    print(f"[오류] {e}\n")
                continue
            print(f"알 수 없는 명령어: {cmd}  (:help 참고)\n")
            continue

        source = block.strip()
        if not source:
            continue
        try:
            resp = post(
                server, "/summarize",
                {"source": source, "query": query, "baseline": show_baseline},
            )
        except RuntimeError as e:
            print(f"[오류] {e}\n")
            continue
        if resp.get("baseline"):
            print(f"\n[순수 LLM] {resp['baseline']}")
        print_summary(
            SummaryResult(
                text=resp["text"],
                active_terms=resp.get("active_terms", []),
                mean_weights=tuple(resp.get("mean_weights", (0.0, 0.0, 0.0, 0.0))),
            )
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Terminal REPL client for the summarization server.")
    p.add_argument("--server", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--query", default=DEFAULT_QUERY, help="summarization instruction")
    args = p.parse_args()

    check_health(args.server)
    try:
        repl(args.server, args.query)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
