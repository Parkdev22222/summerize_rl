"""Structural test for the model-agnostic SAD head (modeling_exaone_sad.py).

This does NOT need EXAONE or a GPU: it builds a tiny fake `*ForCausalLM` (any
class exposing get_decoder()/get_output_embeddings()), attaches the SAD head via
add_sad_head, and checks that forward() returns the
(main, presumm, null, weight) 4-tuple with weight of shape [bs, 3] — the exact
contract the fork's generation loop consumes.

Requires torch; skipped (prints SKIP, exits 0) when torch is unavailable so the
stdlib suite still runs. On a machine with torch:  python -m pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    from transformers.modeling_outputs import BaseModelOutputWithPast
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def _build_fake_causal_lm():
    class Cfg:
        hidden_size = 16
        vocab_size = 32
        alpha_et_hidden_size = 16
        dropout_rate = 0.0
        sqrt_dimension = 1
        sqrt_method = "concate_dim"
        output_attentions = False
        output_hidden_states = False
        use_return_dict = True

    class FakeDecoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        def forward(self, input_ids=None, inputs_embeds=None, **kw):
            h = self.embed(input_ids)
            return BaseModelOutputWithPast(
                last_hidden_state=h, past_key_values=None,
                hidden_states=None, attentions=None,
            )

    class FakeForCausalLM(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.transformer = FakeDecoder(cfg)      # EXAONE names it `transformer`
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def get_decoder(self):
            return self.transformer

        def get_output_embeddings(self):
            # Emulate EXAONE's tied-embedding case where this returns None; the SAD
            # generate must not rely on it (it calls the native forward instead).
            return None

        def forward(self, input_ids=None, output_hidden_states=False, **kw):
            h = self.transformer(input_ids=input_ids).last_hidden_state
            logits = self.lm_head(h)
            return SimpleNamespace(logits=logits, hidden_states=(h,), past_key_values=None)

    return FakeForCausalLM(Cfg()), Cfg()


def test_sad_head_forward_contract():
    if not HAVE_TORCH:
        print("SKIP test_sad_head_forward_contract (torch unavailable)")
        return
    from modeling_exaone_sad import add_sad_head

    model, cfg = _build_fake_causal_lm()
    model = add_sad_head(model, cfg, fc_fp32=True)

    bs, seq = 2, 5
    ids = torch.randint(0, cfg.vocab_size, (bs, seq))
    presumm = torch.randint(0, cfg.vocab_size, (bs, seq))
    null = torch.randint(0, cfg.vocab_size, (bs, seq))

    out = model(input_ids=ids, presumm_input_ids=presumm, null_input_ids=null)
    assert isinstance(out, tuple) and len(out) == 4, "forward must return 4-tuple"
    main_out, presumm_out, null_out, weight = out
    assert main_out.logits.shape == (bs, seq, cfg.vocab_size)
    assert presumm_out.logits.shape == (bs, seq, cfg.vocab_size)
    assert null_out.logits.shape == (bs, seq, cfg.vocab_size)
    assert weight.shape == (bs, 3), f"weight must be [bs,3], got {tuple(weight.shape)}"

    # The combination the fork's sample() applies must be finite/well-formed.
    alpha_beta = torch.softmax(weight[:, :2], dim=1)
    gamma = torch.sigmoid(weight[:, 2]).unsqueeze(1)
    alpha, beta = alpha_beta[:, 0:1], alpha_beta[:, 1:2]
    main_l = main_out.logits[:, -1, :]
    combined = (1 + gamma) * (alpha * main_l + beta * presumm_out.logits[:, -1, :]) - gamma * null_out.logits[:, -1, :]
    assert combined.shape == (bs, cfg.vocab_size)
    assert torch.isfinite(combined).all()
    print("PASS test_sad_head_forward_contract")


def test_sad_generate_contract():
    if not HAVE_TORCH:
        print("SKIP test_sad_generate_contract (torch unavailable)")
        return
    from types import SimpleNamespace
    from modeling_exaone_sad import add_sad_head

    model, cfg = _build_fake_causal_lm()
    model = add_sad_head(model, cfg, fc_fp32=True)

    bs, seq, nrs, max_new = 2, 5, 2, 4
    ids = torch.randint(0, cfg.vocab_size, (bs, seq))
    mask = torch.ones(bs, seq, dtype=torch.long)
    p = torch.randint(0, cfg.vocab_size, (bs, 3))
    n = torch.randint(0, cfg.vocab_size, (bs, 2))
    gc = SimpleNamespace(do_sample=True, top_k=0, top_p=1.0, temperature=1.0,
                         max_new_tokens=max_new, min_new_tokens=2,
                         eos_token_id=None, pad_token_id=0)

    out = model.generate(
        input_ids=ids, attention_mask=mask,
        presumm_input=p, presumm_attention_mask=torch.ones_like(p),
        null_inputs=n, null_attention_mask=torch.ones_like(n),
        generation_config=gc, num_return_sequences=nrs,
        return_dict_in_generate=True, output_scores=True,
    )
    # sequences = expanded main prompt (seq) + generated tokens; scores per step.
    assert out.sequences.shape[0] == bs * nrs, "num_return_sequences expansion"
    assert len(out.scores) == max_new, f"expected {max_new} score steps"
    assert out.sequences.shape[1] == seq + len(out.scores), "prompt prefix + generated"
    assert out.scores[0].shape == (bs * nrs, cfg.vocab_size)
    # slicing off the prompt (as the training loop does) yields the generated ids.
    gen = out.sequences[:, seq:]
    assert gen.shape == (bs * nrs, max_new)
    print("PASS test_sad_generate_contract")


def test_sad_generate_grad_flows_to_fc():
    """With the backbone frozen (only my_* trainable, as SARA does), generate's
    scores must carry grad to the FC head so loss.backward() works."""
    if not HAVE_TORCH:
        print("SKIP test_sad_generate_grad_flows_to_fc (torch unavailable)")
        return
    from types import SimpleNamespace
    from modeling_exaone_sad import add_sad_head

    model, cfg = _build_fake_causal_lm()
    model = add_sad_head(model, cfg, fc_fp32=True)
    for name, p in model.named_parameters():
        p.requires_grad = ("my_" in name)  # freeze backbone, train FC only

    bs = 2
    ids = torch.randint(0, cfg.vocab_size, (bs, 5))
    p = torch.randint(0, cfg.vocab_size, (bs, 3))
    n = torch.randint(0, cfg.vocab_size, (bs, 2))
    gc = SimpleNamespace(do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
                         max_new_tokens=3, min_new_tokens=1,
                         eos_token_id=None, pad_token_id=0)
    out = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        presumm_input=p, presumm_attention_mask=torch.ones_like(p),
        null_inputs=n, null_attention_mask=torch.ones_like(n),
        generation_config=gc, num_return_sequences=1,
    )
    scores = torch.stack(out.scores, dim=0)  # [steps, bs, vocab]
    assert scores.requires_grad, "generate scores must carry grad (no @torch.no_grad)"
    scores.sum().backward()
    assert model.my_f.weight.grad is not None, "grad must reach the FC head"
    print("PASS test_sad_generate_grad_flows_to_fc")


if __name__ == "__main__":
    test_sad_head_forward_contract()
    test_sad_generate_contract()
    test_sad_generate_grad_flows_to_fc()
