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

    # -- batched decoding: N rollouts of the SAME prompt at once -------------
    #
    # All N rollouts share one prompt (same 4 branch texts); they diverge only
    # in the tokens they sample. Batching them into one forward (batch = N*4)
    # is the main training speedup, turning N sequential single-stream decodes
    # into one wide decode. Batched StepOutput stacks a rollout axis:
    #   logits [N, num_branches, vocab],  hidden [N, num_branches, hidden].

    def start_batch(self, branch_texts: dict[str, str], n: int) -> tuple[Any, StepOutput]:
        """Prefill N copies of the prompt. Returns (state, batched StepOutput)."""
        raise NotImplementedError

    def step_batch(self, state: Any, tokens: list[int]) -> StepOutput:
        """Advance every rollout by its own token (len(tokens) == N)."""
        raise NotImplementedError

    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Plain greedy text generation for an arbitrary prompt.

        Separate from the 4-branch PMI decode: used to have the frozen LLM
        extract key sentences from a source for the reward. Optional; backends
        that don't support it can leave this unimplemented.
        """
        raise NotImplementedError


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

    # -- batched decoding ---------------------------------------------------

    def _emit_batch(self, ctx_lens_per_rollout: list[list[int]]) -> StepOutput:
        logits = torch.stack([
            torch.stack([self._branch_logits(i, cl[i]) for i in range(len(BRANCH_ORDER))])
            for cl in ctx_lens_per_rollout
        ])  # [N, 4, V]
        hidden = torch.stack([
            torch.stack([self._branch_hidden(i, cl[i]) for i in range(len(BRANCH_ORDER))])
            for cl in ctx_lens_per_rollout
        ])  # [N, 4, D]
        return StepOutput(logits=logits, hidden=hidden)

    def start_batch(self, branch_texts: dict[str, str], n: int) -> tuple[Any, StepOutput]:
        base = [max(1, len(branch_texts.get(name, ""))) for name in BRANCH_ORDER]
        state = {"ctx_lens": [list(base) for _ in range(n)], "n": n}
        return state, self._emit_batch(state["ctx_lens"])

    def step_batch(self, state: Any, tokens: list[int]) -> StepOutput:
        # Mock logits depend only on (branch, ctx_len), not on the token, so we
        # just advance every rollout's context length by one.
        state["ctx_lens"] = [[c + 1 for c in cl] for cl in state["ctx_lens"]]
        return self._emit_batch(state["ctx_lens"])

    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Deterministic stub: echo the last chunk of the prompt (which contains
        # the source), so key-sentence extraction is reproducible in tests.
        return prompt.strip()[-max_new_tokens:]


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
        attn_implementation: str | None = "sdpa",
        compile_decode: bool = False,
        max_seq_len: int = 2048,
        trust_remote_code: bool = False,
    ):
        """attn_implementation: None (transformers default) | "sdpa" |
        "flash_attention_2" | "eager". Default "sdpa" is built into PyTorch
        (no extra install) and already dispatches to FlashAttention / memory-
        efficient kernels on GPU, so it is fast with identical math. Pass
        "flash_attention_2" only if the flash-attn package is installed. If the
        requested kernel is unavailable it falls back (flash_attention_2 -> sdpa
        -> eager) with a notice, so this never hard-fails.

        compile_decode: if True, decode with a fixed-size ``StaticCache`` and a
        ``torch.compile``d model so the single-token step has static shapes and
        can be captured as a CUDA graph (much lower per-step launch overhead).
        Requires a StaticCache-compatible architecture (Llama/Qwen/Mistral/...).
        ``max_seq_len`` bounds prompt+generation for the static cache/mask; keep
        it near your real maximum (larger = more wasted per-step attention).
        The first call pays a one-time compilation cost.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        torch_dtype = getattr(torch, dtype)
        self.model, self.attn_implementation = self._load_model(
            AutoModelForCausalLM, model_name, torch_dtype, attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model = self.model.to(device)
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

        self.compile_decode = compile_decode
        self.max_seq_len = max_seq_len
        # Compile only the single-token decode step (static shapes); prefill
        # stays eager since its length varies per input.
        self._decode_model = (
            torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            if compile_decode
            else self.model
        )

    @staticmethod
    def _attn_fallback_chain(requested: str | None) -> list[str | None]:
        """Ordered kernels to try: requested first, then safe fallbacks."""
        if requested == "flash_attention_2":
            return ["flash_attention_2", "sdpa", "eager"]
        if requested == "sdpa":
            return ["sdpa", "eager"]
        return [requested]  # None (transformers default) or explicit "eager"

    @classmethod
    def _load_model(cls, auto_cls, model_name, torch_dtype, attn_implementation,
                    trust_remote_code=False):
        """Load the causal LM, honoring attn_implementation with fallback.

        Returns (model, resolved_impl). ``resolved_impl`` is the kernel actually
        used (may differ from requested if it fell back).
        """
        last_err: Exception | None = None
        chain = cls._attn_fallback_chain(attn_implementation)
        for impl in chain:
            kwargs: dict = {
                "dtype": torch_dtype,
                "output_hidden_states": True,
                "trust_remote_code": trust_remote_code,
            }
            if impl is not None:
                kwargs["attn_implementation"] = impl
            try:
                return auto_cls.from_pretrained(model_name, **kwargs), impl
            except (ImportError, ValueError) as e:
                last_err = e
                if impl != chain[-1]:
                    print(f"[info] attn_implementation={impl} 미설치/미지원 → 다음 커널로 폴백")
        raise last_err  # type: ignore[misc]

    def start(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
        if self.compile_decode:
            return self._start_static(branch_texts)
        return self._start_dynamic(branch_texts)

    def step(self, state: Any, token_id: int) -> StepOutput:
        if self.compile_decode:
            return self._step_static(state, token_id)
        return self._step_dynamic(state, token_id)

    # -- dynamic path (DynamicCache, eager) ---------------------------------

    @torch.no_grad()
    def _start_dynamic(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
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
    def _step_dynamic(self, state: Any, token_id: int) -> StepOutput:
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

    # -- batched decoding (dynamic cache; N rollouts of one prompt) ----------
    #
    # The N rollouts share the same 4 branch prompts, so we tokenize once and
    # repeat the 4-row block N times -> batch 4N, laid out rollout-major
    # (rows [4r, 4r+4) belong to rollout r). This is exactly `_start_dynamic`
    # with more rows, so it reuses the proven left-padding + KV-cache path.

    @torch.no_grad()
    def start_batch(self, branch_texts: dict[str, str], n: int) -> tuple[Any, StepOutput]:
        texts = [branch_texts[name] for name in BRANCH_ORDER]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        input_ids = enc["input_ids"].repeat(n, 1)          # [4N, L]
        attention_mask = enc["attention_mask"].repeat(n, 1)  # [4N, L]
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        state = {"past": out.past_key_values, "attention_mask": attention_mask, "n": n}
        logits = out.logits[:, -1, :].float().view(n, len(BRANCH_ORDER), self.vocab_size)
        hidden = out.hidden_states[-1][:, -1, :].float().view(n, len(BRANCH_ORDER), self.hidden_size)
        return state, StepOutput(logits=logits, hidden=hidden)

    @torch.no_grad()
    def step_batch(self, state: Any, tokens: list[int]) -> StepOutput:
        n = state["n"]
        nb = len(BRANCH_ORDER)
        tok = torch.as_tensor(tokens, device=self.device, dtype=torch.long)  # [N]
        input_ids = tok.repeat_interleave(nb).unsqueeze(1)  # [4N, 1], rollout-major
        state["attention_mask"] = torch.cat(
            [state["attention_mask"], torch.ones((n * nb, 1), device=self.device, dtype=torch.long)],
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
        logits = out.logits[:, -1, :].float().view(n, nb, self.vocab_size)
        hidden = out.hidden_states[-1][:, -1, :].float().view(n, nb, self.hidden_size)
        return StepOutput(logits=logits, hidden=hidden)

    # -- static path (StaticCache, compiled single-token step) --------------
    #
    # A fixed-size StaticCache + a fixed-width attention mask keep every decode
    # step at identical shapes, so the compiled step reuses one CUDA graph
    # across tokens AND across requests. Prefill runs eager (length varies).
    # Verified to produce token-identical output to the dynamic path.

    @torch.no_grad()
    def _start_static(self, branch_texts: dict[str, str]) -> tuple[Any, StepOutput]:
        from transformers import StaticCache

        texts = [branch_texts[name] for name in BRANCH_ORDER]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        b, prompt_len = input_ids.shape
        if prompt_len >= self.max_seq_len:
            raise ValueError(
                f"prompt length {prompt_len} >= max_seq_len {self.max_seq_len}; "
                "increase --max-seq-len (or shorten the input)."
            )

        cache = StaticCache(config=self.model.config, max_cache_len=self.max_seq_len)
        cache_position = torch.arange(prompt_len, device=self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=cache_position,
        )
        # Fixed-width mask: 1 for real prompt tokens, filled in as we generate.
        mask = torch.zeros((b, self.max_seq_len), dtype=attn.dtype, device=self.device)
        mask[:, :prompt_len] = attn
        state = {"cache": cache, "mask": mask, "pos": prompt_len}
        logits = out.logits[:, -1, :].float()
        hidden = out.hidden_states[-1][:, -1, :].float()
        return state, StepOutput(logits=logits, hidden=hidden)

    @torch.no_grad()
    def _step_static(self, state: Any, token_id: int) -> StepOutput:
        n = len(BRANCH_ORDER)
        pos = state["pos"]
        if pos >= self.max_seq_len:
            raise ValueError(
                f"generation exceeded max_seq_len {self.max_seq_len}; "
                "increase --max-seq-len."
            )
        input_ids = torch.full((n, 1), token_id, device=self.device, dtype=torch.long)
        state["mask"][:, pos] = 1
        cache_position = torch.tensor([pos], device=self.device)
        out = self._decode_model(
            input_ids=input_ids,
            attention_mask=state["mask"],  # fixed width -> static shape
            past_key_values=state["cache"],
            use_cache=True,
            output_hidden_states=True,
            cache_position=cache_position,
        )
        state["pos"] = pos + 1
        # Clone: reduce-overhead reuses static output buffers across calls.
        logits = out.logits[:, -1, :].float().clone()
        hidden = out.hidden_states[-1][:, -1, :].float().clone()
        return StepOutput(logits=logits, hidden=hidden)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Greedy text generation for an arbitrary prompt (key-sentence extraction)."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
