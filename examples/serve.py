"""HTTP summarization server: a warm backbone + checkpoint on one GPU.

Standard vLLM cannot serve this model: the summary comes from a 4-branch PMI
*policy* decode that needs, at every step, all four branches' next-token logits
AND their last-layer hidden states (to feed the trained weight policy), plus a
cross-branch contrastive logit combination — none of which vLLM's API exposes.
So this is a small custom server wrapping :class:`~summarize_rl.infer.Summarizer`.

The model is loaded once and stays resident; requests are serialized (one GPU,
one model). The terminal REPL client is :mod:`examples.summarize_client`.

Endpoints:
    GET  /health              -> {"status": "ok"}
    POST /summarize {source, query?}  -> {text, active_terms, mean_weights, query}
    POST /reload   {ckpt}     -> {ckpt, step}

Example (pin one GPU):
    CUDA_VISIBLE_DEVICES=3 uv run python -m examples.serve \
        --model <korean-7B-model> --dtype bfloat16 \
        --ckpt checkpoints/best.pt --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from summarize_rl.infer import Summarizer, build_hf_summarizer

from examples.summarize import DEFAULT_QUERY, load_glossary


class SummarizerService:
    """Thread-safe wrapper: serializes summarize/reload on the single model.

    Kept free of any HTTP concern so it can be unit-tested with a MockBackend
    Summarizer (no sockets, no model download).
    """

    def __init__(self, summarizer: Summarizer, default_query: str):
        self.summarizer = summarizer
        self.default_query = default_query
        self._lock = threading.Lock()

    def summarize(self, payload: dict) -> dict:
        source = (payload.get("source") or "").strip()
        if not source:
            raise ValueError("field 'source' is required and must be non-empty")
        query = payload.get("query") or self.default_query
        with self._lock:
            result = self.summarizer.summarize(source, query=query)
        return {
            "text": result.text,
            "active_terms": result.active_terms,
            "mean_weights": list(result.mean_weights),
            "query": query,
        }

    def reload(self, payload: dict) -> dict:
        ckpt = (payload.get("ckpt") or "").strip()
        if not ckpt:
            raise ValueError("field 'ckpt' is required")
        with self._lock:
            step = self.summarizer.load_checkpoint(ckpt)
        return {"ckpt": ckpt, "step": step}


def make_handler(service: SummarizerService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw or b"{}")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            try:
                payload = self._read_json()
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON body"})
                return
            try:
                if self.path == "/summarize":
                    self._send(200, service.summarize(payload))
                elif self.path == "/reload":
                    self._send(200, service.reload(payload))
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, FileNotFoundError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # keep the server alive on unexpected errors
                self._send(500, {"error": str(e)})

        def log_message(self, *args) -> None:  # quiet default request logging
            pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(
        description="Custom summarization HTTP server (frozen backbone + trained checkpoint)."
    )
    p.add_argument("--model", required=True, help="HF model name/path for the frozen backbone")
    p.add_argument("--ckpt", default="checkpoints/best.pt",
                   help="policy checkpoint to load (default: checkpoints/best.pt)")
    p.add_argument("--glossary", default=None, help="JSON {term: [triggers]}; omit for demo glossary")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--dtype", default="bfloat16", help="backbone dtype (bfloat16/float16/float32)")
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="attention kernel (sdpa=safe fast default; flash_attention_2 fastest, needs flash-attn)")
    p.add_argument("--query", default=DEFAULT_QUERY, help="default summarization instruction")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--min-new-tokens", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    glossary = load_glossary(args.glossary)
    try:
        summarizer, _cfg, step = build_hf_summarizer(
            args.model,
            args.ckpt,
            glossary=glossary,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    service = SummarizerService(summarizer, args.query)
    server = HTTPServer((args.host, args.port), make_handler(service))
    print(
        f"serving on http://{args.host}:{args.port}  "
        f"model={args.model} device={args.device} attn={summarizer.backend.attn_implementation} "
        f"ckpt={args.ckpt} (step={step})"
    )
    print("endpoints: GET /health | POST /summarize {source, query?} | POST /reload {ckpt}")
    print("stop with Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        server.server_close()


if __name__ == "__main__":
    main()
