"""LLM backend: the single boundary to the frozen backbone (Section 3, table).

The decoder needs, at every step and for each of the four branches:
  * next-token logits   (vocab-sized), and
  * the last-layer hidden state (feeds the policy network).

All four branches decode the SAME sampled token each step (contrastive
decoding), so the backend keeps four independent KV caches but advances them
with one shared token.

`LLMBackend` is the abstract interface. `MockBackend` is a deterministic,
dependency-free implementation for CPU unit/e2e tests. `HFBackend` wraps a real
Hugging Face causal LM and is imported lazily so torch-only test runs never need
transformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

# Fixed branch order used everywhere logits/hidden are stacked as rows.
BRANCH_ORDER = ("XQ", "SQ", "GQ", "Q")


@dataclass
class StepOutput:
    """One decoding step, all four branches stacked.

    logits: [num_branches, vocab_size]   -- next-token logits (detached / const)
    hidden: [num_branches, hidden_size]  -- last-layer hidden state
    """

    logits: torch.Tensor
    hidden: torch.Tensor


class LLMBackend(ABC):
    hidden_size: int
    vocab_size: int
    eos_token_id: int | None
    pad_token_id: int | None

    @abstractmethod
    def start(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
        """Prefill the four branches. Returns (state, first StepOutput)."""

    @abstractmethod
    def step(self, state: Any, token_id: int) -> StepOutput:
        """Advance all branches by one shared token; mutate state in place."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Detokenize generated ids to text (for reward computation)."""


class MockBackend(LLMBackend):
    """Deterministic backend for tests. No external model.

    Logits and hidden states are cheap deterministic functions of the branch
    index and the running context, so decoder/training behaviour is fully
    reproducible. Each branch is biased toward a distinct "favourite" token so
    that changing the PMI weights provably changes the combined distribution.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 8,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        seed: int = 0,
        vocab: dict[int, str] | None = None,
        branch_bias: dict[int, list[int]] | None = None,
    ):
        """vocab maps token id -> surface word for decode() (else "t<id>").

        branch_bias maps branch index -> token ids that branch should favour;
        lets a demo steer each branch toward semantically distinct words so the
        reward signal (and thus the gradient) is non-trivial. When omitted, each
        branch simply favours a single distinct token.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.vocab = vocab
        # Each branch favours token (branch_index + 3) so branches disagree.
        self.branch_bias = branch_bias or {
            i: [(i + 3) % vocab_size] for i in range(len(BRANCH_ORDER))
        }

    def _branch_logits(self, branch_idx: int, ctx_len: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(
            self.seed * 1000 + branch_idx * 97 + ctx_len
        )
        logits = torch.randn(self.vocab_size, generator=g)
        # Strong, stable bias toward this branch's favoured tokens.
        for tok in self.branch_bias.get(branch_idx, []):
            logits[tok] += 4.0
        return logits

    def _branch_hidden(self, branch_idx: int, ctx_len: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(
            self.seed * 7919 + branch_idx * 31 + ctx_len
        )
        return torch.randn(self.hidden_size, generator=g)

    def _emit(self, ctx_lens: list[int]) -> StepOutput:
        logits = torch.stack(
            [self._branch_logits(i, ctx_lens[i]) for i in range(len(BRANCH_ORDER))]
        )
        hidden = torch.stack(
            [self._branch_hidden(i, ctx_lens[i]) for i in range(len(BRANCH_ORDER))]
        )
        return StepOutput(logits=logits, hidden=hidden)

    def start(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
        # State = per-branch context length (prefill length + generated tokens).
        ctx_lens = [max(1, len(branch_texts.get(name, ""))) for name in BRANCH_ORDER]
        state = {"ctx_lens": ctx_lens}
        return state, self._emit(ctx_lens)

    def step(self, state: Any, token_id: int) -> StepOutput:
        state["ctx_lens"] = [n + 1 for n in state["ctx_lens"]]
        return self._emit(state["ctx_lens"])

    def decode(self, token_ids: list[int]) -> str:
        if self.vocab is not None:
            words = [self.vocab.get(t, f"t{t}") for t in token_ids if t != self.eos_token_id]
            return " ".join(words)
        # Deterministic surface form: one token -> "t<id>".
        return " ".join(f"t{t}" for t in token_ids)


class HFBackend(LLMBackend):
    """Wraps a frozen Hugging Face causal LM. transformers imported lazily.

    Keeps four independent past_key_values, one per branch, and advances them
    with the shared sampled token. The model is frozen (requires_grad_(False))
    and always run under no_grad, so its outputs are constants w.r.t. the policy.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, output_hidden_states=True
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.vocab_size = int(self.model.config.vocab_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self.eos_token_id = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.eos_token_id
        # Left-pad so the last position aligns across branches after prefill.
        self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def start(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
        texts = [branch_texts[name] for name in BRANCH_ORDER]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            use_cache=True,
            output_hidden_states=True,
        )
        state = {
            "past": out.past_key_values,
            "attention_mask": enc["attention_mask"],
        }
        logits = out.logits[:, -1, :].float()
        hidden = out.hidden_states[-1][:, -1, :].float()
        return state, StepOutput(logits=logits, hidden=hidden)

    @torch.no_grad()
    def step(self, state: Any, token_id: int) -> StepOutput:
        n = len(BRANCH_ORDER)
        input_ids = torch.full((n, 1), token_id, device=self.device, dtype=torch.long)
        state["attention_mask"] = torch.cat(
            [state["attention_mask"], torch.ones((n, 1), device=self.device, dtype=torch.long)],
            dim=1,
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=state["attention_mask"],
            past_key_values=state["past"],
            use_cache=True,
            output_hidden_states=True,
        )
        state["past"] = out.past_key_values
        logits = out.logits[:, -1, :].float()
        hidden = out.hidden_states[-1][:, -1, :].float()
        return StepOutput(logits=logits, hidden=hidden)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
