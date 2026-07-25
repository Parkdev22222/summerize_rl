"""Checkpoint-loaded summarization inference.

Training (``SCSTTrainer.save_checkpoint``) writes checkpoints under
``checkpoints/`` whose payload is::

    {"policy": state_dict, "optimizer": ..., "scheduler": ..., "step", "best_reward"}

Only the small weight-policy MLP is trained; the LLM backbone is frozen. So
inference is: load ``payload["policy"]`` into a ``WeightPolicy``, build the four
branch prompts for the input, and run a single **greedy** PMI-combined decode.

``Summarizer`` is backend-agnostic (works with ``MockBackend`` in tests and
``HFBackend`` in real use); the CLI (:mod:`examples.summarize`) wires it to a
real Hugging Face backbone and a REPL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from .branches import Example, Triplet, build_branches
from .config import Config
from .decoder import generate
from .glossary import Glossary
from .llm_backend import LLMBackend
from .policy import WeightPolicy
from .train import freeze_llm


@dataclass
class SummaryResult:
    """One summary plus decoding insight."""

    text: str
    active_terms: list[str] = field(default_factory=list)
    # Mean PMI weights (a, b, c, d) over the generated tokens.
    mean_weights: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _looks_like_state_dict(obj: object) -> bool:
    """A bare state_dict is a mapping whose values are tensors."""
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(v, torch.Tensor) for v in obj.values())


class Summarizer:
    """Load a trained weight-policy checkpoint and summarize inputs with it.

    The policy is put in eval mode (dropout off) so summaries are deterministic
    under greedy decoding. The LLM backbone is frozen.
    """

    def __init__(
        self,
        backend: LLMBackend,
        policy: WeightPolicy,
        config: Config,
        *,
        glossary: Glossary | None = None,
    ):
        self.backend = backend
        self.policy = policy
        self.config = config
        self.glossary = glossary

        freeze_llm(backend)
        self.policy.eval()

    # -- checkpoint loading -------------------------------------------------

    def load_checkpoint(self, path: str) -> int:
        """Load policy weights from a training checkpoint. Returns the step.

        Accepts either the full training payload (``{"policy": state_dict,
        ...}``) or a bare policy ``state_dict``. Raises a clear error when the
        file is missing or the contents are not a recognizable checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"checkpoint not found: {path}\n"
                "Train first (e.g. python -m examples.train_real ... --ckpt-dir checkpoints) "
                "or pass a valid --ckpt path."
            )

        payload = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(payload, dict) and "policy" in payload:
            state_dict = payload["policy"]
            step = int(payload.get("step", 0))
        elif _looks_like_state_dict(payload):
            state_dict = payload
            step = 0
        else:
            raise ValueError(
                f"unrecognized checkpoint format in {path}: expected a payload "
                "with a 'policy' key or a bare state_dict."
            )

        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        return step

    # -- inference ----------------------------------------------------------

    def _prepare(
        self,
        source: str,
        query: str | None,
        triplets: list[Triplet] | None,
    ) -> tuple[list[str], dict[str, str]]:
        """Build (active_terms, branch_texts) for one input. Shared by the PMI
        decode and the pure-LLM baseline so both see identical prompts."""
        kwargs: dict = {"source": source, "triplets": triplets or []}
        if query is not None:
            kwargs["query"] = query
        example = Example(**kwargs)
        active = self.glossary.gate(source) if self.glossary else []
        active_terms = [a.term for a in active]
        branch_texts = build_branches(example, active).as_dict()
        return active_terms, branch_texts

    def baseline_summary(
        self,
        source: str,
        *,
        query: str | None = None,
        triplets: list[Triplet] | None = None,
    ) -> str:
        """Pure-LLM summary: the frozen backbone's own answer, no PMI, no policy.

        Feeds the XQ prompt (source + instruction) straight to the backbone via
        ``generate_text`` — no branch mixing and no trained weight policy. This
        is the baseline the PMI policy is meant to improve on, so showing both
        side by side makes the policy's effect visible. Uses the same
        ``max_new_tokens`` budget as the PMI decode.
        """
        _active, branch_texts = self._prepare(source, query, triplets)
        with torch.no_grad():
            text = self.backend.generate_text(
                branch_texts["XQ"], self.config.decode.max_new_tokens
            )
        return text.strip()

    def summarize(
        self,
        source: str,
        *,
        query: str | None = None,
        triplets: list[Triplet] | None = None,
    ) -> SummaryResult:
        """Summarize ``source`` under an instruction ``query`` (greedy decode).

        ``query`` defaults to the standard instruction on :class:`Example`.
        ``triplets`` feed the SQ (core-info) branch; omit for source-only input.
        """
        active_terms, branch_texts = self._prepare(source, query, triplets)

        with torch.no_grad():
            rollout = generate(
                self.backend,
                branch_texts,
                self.policy,
                self.config.decode,
                greedy=True,
            )

        trace = rollout.weight_trace
        if trace:
            n = len(trace)
            mean_weights = tuple(sum(w[i] for w in trace) / n for i in range(4))
        else:
            mean_weights = (0.0, 0.0, 0.0, 0.0)

        return SummaryResult(
            text=rollout.text,
            active_terms=active_terms,
            mean_weights=mean_weights,  # type: ignore[arg-type]
        )


def build_hf_summarizer(
    model: str,
    ckpt: str,
    *,
    glossary: Glossary | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    compile_decode: bool = False,
    max_seq_len: int = 2048,
    max_new_tokens: int | None = None,
    min_new_tokens: int | None = None,
    trust_remote_code: bool = False,
) -> tuple[Summarizer, Config, int]:
    """Build a real-backbone Summarizer and load a checkpoint. Returns (summarizer, config, step).

    Loads a frozen ``HFBackend``, syncs the default ``Config`` to it (hidden
    size, eos/pad ids), puts a fresh ``WeightPolicy`` on the same device, and
    loads the checkpoint. Shared by the REPL (:mod:`examples.summarize`) and the
    server (:mod:`examples.serve`). ``HFBackend`` is imported lazily so the
    backend-agnostic core above never requires transformers.
    """
    from .llm_backend import HFBackend  # lazy: keeps the core transformers-free

    backend = HFBackend(
        model,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        compile_decode=compile_decode,
        max_seq_len=max_seq_len,
        trust_remote_code=trust_remote_code,
    )

    cfg = Config()
    cfg.policy.llm_hidden_size = backend.hidden_size
    cfg.decode.eos_token_id = backend.eos_token_id
    cfg.decode.pad_token_id = backend.pad_token_id
    if max_new_tokens is not None:
        cfg.decode.max_new_tokens = max_new_tokens
    if min_new_tokens is not None:
        cfg.decode.min_new_tokens = min_new_tokens

    dev = torch.device(device)
    policy = WeightPolicy(cfg.policy).to(dev)
    summarizer = Summarizer(backend, policy, cfg, glossary=glossary)
    step = summarizer.load_checkpoint(ckpt)
    return summarizer, cfg, step
