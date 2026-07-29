"""Inject SARA's Salience-Aware (context-aware) decoding head into EXAONE.

SARA implements context-aware decoding by modifying each backbone's
`*ForCausalLM` in the vendored transformers fork (see
transformers/src/transformers/models/llama/modeling_llama.py: LlamaForCausalLM):

  * __init__ adds three FC layers (my_all_f, my_all_f1, my_f) that map the
    concatenated last-token hidden states of the main / presumm / null branches
    to a 3-vector `weight = [alpha_logit, beta_logit, gamma_logit]`.
  * forward() runs the base model THREE times (main, presumm, null via
    `forward_once`) and returns `(outputs, presumm_outputs, null_outputs, weight)`.
  * The fork's generation loop (generation/utils.py: sample) combines them as
        alpha,beta = softmax(weight[:, :2]);  gamma = sigmoid(weight[:, 2])
        logits = (1+gamma)*(alpha*main + beta*presumm) - gamma*null

EXAONE 3.5 uses its own architecture (`ExaoneForCausalLM`, loaded via
trust_remote_code) that is NOT in the fork's model zoo, so we cannot edit a
static modeling file. Instead we attach the SAME head to whatever
`*ForCausalLM` class EXAONE loads as, at load time. The head logic here is
model-agnostic: it uses `self.get_decoder()` (EXAONE's base transformer) and
`self.get_output_embeddings()` (its lm_head), so it works for EXAONE — or any
other causal LM — without touching the backbone's own code.

IMPORTANT (must be verified on real hardware): the vendored fork is
transformers==4.36.0. EXAONE's remote modeling targets a newer transformers, so
loading it under the fork may need adaptation (e.g. `prepare_inputs_for_generation`
returning a `cache_position` key). `forward_once` below therefore accepts and
ignores unexpected kwargs. Validate a few decode steps with the real model +
GPU before training; see INTEGRATION_ko.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


# --------------------------------------------------------------------------- #
# Model-agnostic SAD head (mirrors the fork's LlamaForCausalLM modification)    #
# --------------------------------------------------------------------------- #

def _forward_once(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,  # tolerate backbone-specific extras (e.g. cache_position)
):
    """Single-branch forward: run the backbone once -> (CausalLMOutput, last_hidden).

    Uses the backbone's OWN forward (``_sad_base_forward``, captured in
    add_sad_head) so logits are computed natively — EXAONE ties embeddings and
    get_output_embeddings() can be None, so we never call lm_head ourselves.
    ``output_hidden_states=True`` yields the last-layer hidden for the FC head.
    """
    base_forward = getattr(self, "_sad_base_forward", None)
    if base_forward is not None:
        out = base_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        return out, out.hidden_states[-1]

    # Fallback for backbones that expose lm_head (get_output_embeddings not None).
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    decoder = self.get_decoder()
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden_state = outputs.last_hidden_state
    logits = self.get_output_embeddings()(outputs[0]).float()
    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    ), last_hidden_state


def _sad_forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    null_input_ids=None,
    null_past_key_values=None,
    null_attention_mask=None,
    null_position_ids=None,
    null_inputs_embeds=None,
    null_use_cache=None,
    presumm_input_ids=None,
    presumm_past_key_values=None,
    presumm_attention_mask=None,
    presumm_position_ids=None,
    presumm_inputs_embeds=None,
    presumm_use_cache=None,
    **kwargs,
):
    """Three-branch SAD forward returning (main, presumm, null, weight).

    Mirrors LlamaForCausalLM.forward in the fork. The fork's generation loop
    prefixes prepare_inputs_for_generation outputs with null_/presumm_ and calls
    this; we route each group through forward_once, then produce the [bs, 3]
    combination weight from the three last-token hidden states.
    """
    outputs, hidden_states = self.forward_once(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        labels=labels,
    )

    presumm_outputs, presumm_states = self.forward_once(
        input_ids=presumm_input_ids,
        past_key_values=presumm_past_key_values,
        attention_mask=presumm_attention_mask,
        position_ids=presumm_position_ids,
        inputs_embeds=presumm_inputs_embeds,
        use_cache=presumm_use_cache,
        labels=labels,
    )

    null_outputs, null_states = self.forward_once(
        input_ids=null_input_ids,
        past_key_values=null_past_key_values,
        attention_mask=null_attention_mask,
        position_ids=null_position_ids,
        inputs_embeds=null_inputs_embeds,
        use_cache=null_use_cache,
        labels=labels,
    )

    hidden_states = hidden_states[:, -1, :]      # [bs, hidden]
    null_states = null_states[:, -1, :]
    presumm_states = presumm_states[:, -1, :]

    concate_hidden_states = torch.cat([hidden_states, presumm_states, null_states], axis=-1)  # [bs, hidden*3]
    # FC layers may be fp32 (see add_sad_head / --my_fc_fp32); match their dtype.
    concate_hidden_states = concate_hidden_states.to(self.my_all_f.weight.dtype)
    concate_hidden_states = self.my_all_f(concate_hidden_states)

    if self.sqrt_dimension:
        if self.sqrt_method == "concate_dim":
            concate_hidden_states = concate_hidden_states / torch.sqrt(
                torch.tensor(concate_hidden_states.size(-1), dtype=torch.float32))
        if self.sqrt_method == "sqrt05":
            concate_hidden_states = concate_hidden_states / torch.sqrt(
                torch.tensor(hidden_states.size(-1), dtype=torch.float32))

    concate_hidden_states = self.relu(concate_hidden_states)
    concate_hidden_states = self.dropout(concate_hidden_states)
    concate_hidden_states = self.my_all_f1(concate_hidden_states)

    if self.sqrt_dimension:
        if self.sqrt_method == "concate_dim":
            concate_hidden_states = concate_hidden_states / torch.sqrt(
                torch.tensor(concate_hidden_states.size(-1), dtype=torch.float32))
        if self.sqrt_method == "sqrt05":
            concate_hidden_states = concate_hidden_states / torch.sqrt(
                torch.tensor(hidden_states.size(-1), dtype=torch.float32))

    concate_hidden_states = self.relu2(concate_hidden_states)
    concate_hidden_states = self.dropout(concate_hidden_states)
    weight = self.my_f(concate_hidden_states)  # [bs, 3]

    assert weight.size(0) == hidden_states.size(0), "weight batch size must match"
    return outputs, presumm_outputs, null_outputs, weight


# --------------------------------------------------------------------------- #
# Self-contained context-aware generation (works on modern transformers)        #
#                                                                              #
# SARA's original 3-branch decoding lives in its transformers *fork*'s         #
# generation loop. EXAONE only runs on modern transformers, so instead of      #
# depending on the fork we reproduce that loop here, using ONLY the base        #
# decoder's core forward contract (input_ids/attention_mask/position_ids/       #
# past_key_values/cache_position) so it is robust across transformers versions. #
# --------------------------------------------------------------------------- #

def _weight_from_hidden(model, mh, ph, nh):
    """FC head: concat last-token hidden of (main, presumm, null) -> weight[bs,3]."""
    x = torch.cat([mh, ph, nh], dim=-1).to(model.my_all_f.weight.dtype)
    x = model.my_all_f(x)
    if model.sqrt_dimension:
        if model.sqrt_method == "concate_dim":
            x = x / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        elif model.sqrt_method == "sqrt05":
            x = x / torch.sqrt(torch.tensor(mh.size(-1), dtype=torch.float32))
    x = model.dropout(model.relu(x))
    x = model.my_all_f1(x)
    if model.sqrt_dimension:
        if model.sqrt_method == "concate_dim":
            x = x / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        elif model.sqrt_method == "sqrt05":
            x = x / torch.sqrt(torch.tensor(mh.size(-1), dtype=torch.float32))
    x = model.dropout(model.relu2(x))
    return model.my_f(x)  # [bs, 3]


def _position_ids_from_mask(attn_mask, cur_len):
    """Left-padding-aware position ids; return the last `cur_len` columns."""
    pos = attn_mask.long().cumsum(-1) - 1
    pos = pos.masked_fill(attn_mask == 0, 0)
    return pos[:, -cur_len:]


def _branch_step(model, input_step, attn_mask, past, past_len):
    """One branch forward for the current step -> (last_logits, last_hidden, new_past).

    Calls the backbone's OWN ``*ForCausalLM.forward`` (captured as
    ``model._sad_base_forward``) so that lm_head / tied embeddings / any
    architecture-specific logit computation are handled natively — we do not call
    lm_head ourselves (EXAONE's get_output_embeddings() can be None when tied).
    ``output_hidden_states=True`` gives the last-layer hidden for the FC head.
    Uses only stable forward args so it works across transformers versions.
    ``attn_mask`` is the FULL mask (past+current); ``input_step`` is the full
    prompt on step 0 and the single new token afterwards.
    """
    cur_len = input_step.shape[1]
    device = input_step.device
    position_ids = _position_ids_from_mask(attn_mask, cur_len)
    cache_position = torch.arange(past_len, past_len + cur_len, device=device)
    out = model._sad_base_forward(
        model,
        input_ids=input_step,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
        cache_position=cache_position,
        output_hidden_states=True,
        return_dict=True,
    )
    logits_last = out.logits[:, -1, :].float()
    hidden_last = out.hidden_states[-1][:, -1, :]
    return logits_last, hidden_last, out.past_key_values


def _filter_and_pick(logits, gc):
    """Apply temperature/top-k/top-p and sample (or argmax if do_sample is False)."""
    temp = getattr(gc, "temperature", 1.0) or 1.0
    if temp != 1.0:
        logits = logits / temp
    if getattr(gc, "do_sample", False):
        top_k = getattr(gc, "top_k", 0) or 0
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        top_p = getattr(gc, "top_p", 1.0)
        top_p = 1.0 if top_p is None else top_p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cum > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = sorted_remove.scatter(-1, sorted_idx, sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.argmax(logits, dim=-1)


def _sad_generate(
    self,
    input_ids=None,
    attention_mask=None,
    presumm_input=None,
    presumm_attention_mask=None,
    null_inputs=None,
    null_attention_mask=None,
    generation_config=None,
    num_return_sequences=1,
    return_dict_in_generate=False,
    output_scores=True,
    **kwargs,
):
    """Context-aware 3-branch generation returning .sequences and .scores.

    Mirrors SARA's fork sample()/greedy loop:
        alpha,beta = softmax(weight[:, :2]);  gamma = sigmoid(weight[:, 2])
        logits = (1+gamma)*(alpha*main + beta*presumm) - gamma*null
    but is self-contained so it runs on the model's own (modern) transformers.
    `sequences` includes the main prompt prefix (the caller slices it off);
    `scores` is a per-step tuple of the combined logits (used by the RL loss).
    """
    gc = generation_config
    device = input_ids.device

    eos_id = getattr(gc, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    pad_id = getattr(gc, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    max_new = int(getattr(gc, "max_new_tokens", 64) or 64)
    min_new = int(getattr(gc, "min_new_tokens", 0) or 0)

    nrs = int(num_return_sequences or 1)

    def _expand(t):
        return None if t is None else t.repeat_interleave(nrs, dim=0)

    def _mask_for(ids, mask):
        if ids is None:
            return None
        return _expand(mask) if mask is not None else torch.ones_like(ids).repeat_interleave(nrs, dim=0)

    m_ids = _expand(input_ids)
    m_mask = _expand(attention_mask) if attention_mask is not None else torch.ones_like(m_ids)
    p_ids, p_mask = _expand(presumm_input), _mask_for(presumm_input, presumm_attention_mask)
    n_ids, n_mask = _expand(null_inputs), _mask_for(null_inputs, null_attention_mask)

    bsz = m_ids.shape[0]
    # Context-aware combination needs both extra branches; otherwise fall back to
    # plain single-branch decoding (the training path always supplies them).
    have_ctx = p_ids is not None and n_ids is not None
    generated = m_ids
    unfinished = torch.ones(bsz, dtype=torch.long, device=device)
    scores = []

    cur_m, cur_p, cur_n = m_ids, p_ids, n_ids
    past_m = past_p = past_n = None
    plen_m = plen_p = plen_n = 0

    for step in range(max_new):
        m_logits, m_h, past_m = _branch_step(self, cur_m, m_mask, past_m, plen_m)
        plen_m += cur_m.shape[1]

        if have_ctx:
            p_logits, p_h, past_p = _branch_step(self, cur_p, p_mask, past_p, plen_p)
            n_logits, n_h, past_n = _branch_step(self, cur_n, n_mask, past_n, plen_n)
            plen_p += cur_p.shape[1]
            plen_n += cur_n.shape[1]

            weight = _weight_from_hidden(self, m_h, p_h, n_h)
            ab = torch.softmax(weight[:, :2], dim=1)
            alpha, beta = ab[:, 0:1], ab[:, 1:2]
            gamma = torch.sigmoid(weight[:, 2]).unsqueeze(1)
            combined = (1 + gamma) * (alpha * m_logits + beta * p_logits) - gamma * n_logits
        else:
            combined = m_logits

        if step < min_new and eos_id is not None:
            # clone before in-place so autograd stays valid when grad flows.
            combined = combined.clone()
            combined[:, eos_id] = float("-inf")

        scores.append(combined)
        next_token = _filter_and_pick(combined, gc)
        if eos_id is not None:
            next_token = next_token * unfinished + pad_id * (1 - unfinished)

        generated = torch.cat([generated, next_token[:, None]], dim=1)
        ones = torch.ones((bsz, 1), dtype=m_mask.dtype, device=device)
        m_mask = torch.cat([m_mask, ones], dim=1)
        cur_m = next_token[:, None]
        if have_ctx:
            p_mask = torch.cat([p_mask, ones], dim=1)
            n_mask = torch.cat([n_mask, ones], dim=1)
            cur_p = cur_n = next_token[:, None]

        if eos_id is not None:
            unfinished = unfinished * (next_token != eos_id).long()
            if int(unfinished.max()) == 0:
                break

    if return_dict_in_generate:
        return SimpleNamespace(sequences=generated, scores=tuple(scores))
    return generated  # plain tensor, matching HF generate's default contract


# --------------------------------------------------------------------------- #
# Attaching the head to a loaded backbone                                      #
# --------------------------------------------------------------------------- #

def add_sad_head(model, config, fc_fp32: bool = True):
    """Rebless ``model`` to a SAD subclass and attach the 3 FC layers.

    ``model`` is a loaded ``*ForCausalLM`` (e.g. EXAONE). We create a subclass
    that overrides forward/forward_once with the model-agnostic SAD versions,
    then add the FC layers on the head's device (fp32 by default, matching
    SARA's --my_fc_fp32 convention).
    """
    base_cls = type(model)
    sad_cls = type(
        base_cls.__name__ + "_SAD",
        (base_cls,),
        {
            "forward": _sad_forward,
            "forward_once": _forward_once,
            "generate": _sad_generate,  # self-contained 3-branch context-aware decode
        },
    )
    model.__class__ = sad_cls
    # Keep the backbone's ORIGINAL forward so per-branch steps compute logits the
    # native way (handles lm_head / tied embeddings); _branch_step calls this.
    model._sad_base_forward = base_cls.forward

    # SAD hyperparameters (set on config by the loader).
    model.alpha_et_hidden_size = config.alpha_et_hidden_size
    model.sqrt_dimension = getattr(config, "sqrt_dimension", 1)
    model.sqrt_method = getattr(config, "sqrt_method", "concate_dim")

    hidden = config.hidden_size
    alpha_et = config.alpha_et_hidden_size
    dropout_rate = getattr(config, "dropout_rate", 0.0)

    model.my_all_f = nn.Linear(hidden * 3, alpha_et)
    model.my_all_f1 = nn.Linear(alpha_et, alpha_et)
    model.my_f = nn.Linear(alpha_et, 3)
    model.relu = nn.ReLU()
    model.relu2 = nn.ReLU()
    model.dropout = nn.Dropout(p=dropout_rate)

    # Place the new layers on the model's device; fp32 for stable weight learning.
    # (get_output_embeddings() can be None on tied models, so don't rely on it.)
    ref_param = next(model.parameters())
    dev = ref_param.device
    dtype = torch.float32 if fc_fp32 else ref_param.dtype
    for m in (model.my_all_f, model.my_all_f1, model.my_f):
        m.to(device=dev, dtype=dtype)
    return model


def load_exaone_sad(args):
    """Load EXAONE 3.5 Instruct and attach SARA's SAD head.

    Uses trust_remote_code so EXAONE's own architecture provides the base
    transformer + lm_head; we only add the context-aware FC head. Returns a
    model whose forward yields (main, presumm, null, weight) for the fork's
    context-aware generation loop.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        getattr(args, "loading_mode", "bf16"), torch.bfloat16)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # SAD head hyperparameters (same knobs SARA feeds Llama/GPT-Neo via config).
    config.alpha_et_hidden_size = args.alpha_et_hidden_size
    config.dropout_rate = getattr(args, "dropout_rate", 0.0)
    config.sqrt_dimension = getattr(args, "sqrt_dimension", 1)
    config.sqrt_method = getattr(args, "sqrt_method", "concate_dim")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=dtype,
        device_map="balanced",
        trust_remote_code=True,
    )
    model = add_sad_head(model, config, fc_fp32=getattr(args, "my_fc_fp32", True))
    return model


# --------------------------------------------------------------------------- #
# Teacher-forced re-scoring of the context-aware policy (for multi-epoch PPO)    #
#                                                                              #
# `_sad_generate` only produces the combined (context-aware) logits ONE token   #
# at a time, so the rollout's per-position logprobs exist only as a by-product   #
# of the sampling pass. Multi-epoch PPO/GRPO (μ>1) needs to re-score the SAME     #
# rollout under the UPDATED policy each inner iteration. These two helpers do     #
# exactly that with teacher forcing:                                             #
#                                                                              #
#   sad_branch_features        — run each of the 3 branches once over            #
#                                [prompt_b ; response] (backbone is frozen, so    #
#                                no_grad) and cache the per-position logits +     #
#                                last-layer hidden that predict the response.     #
#   sad_logprobs_from_features  — re-apply the FC head (the ONLY trainable part)  #
#                                to those cached hidden states to get a           #
#                                per-position mix weight, combine the branch      #
#                                logits with SARA's (1+γ)(α·m+β·p)−γ·n formula,    #
#                                and return the differentiable per-token logprob. #
#                                                                              #
# Because only the FC re-mix carries gradient, the expensive backbone pass runs   #
# once per batch and every inner epoch is a cheap FC forward — so μ>1 is nearly   #
# free. The slice `[:, Pb-1:-1, :]` mirrors reference_logprobs and assumes        #
# left-padded prompts (the last real prompt token sits at column Pb-1), which is  #
# what batched generation already requires.                                      #
# --------------------------------------------------------------------------- #

def _branch_seq_features(model, prompt_ids, prompt_mask, gen, nr):
    """One branch: teacher-forced forward over [prompt;gen] -> per-position
    (logits, hidden) that predict the ``gen`` tokens. Detached (backbone frozen).
    """
    prompt = prompt_ids.repeat_interleave(nr, dim=0)
    if prompt_mask is None:
        pmask = torch.ones_like(prompt)
    else:
        pmask = prompt_mask.repeat_interleave(nr, dim=0)
    gen = gen.to(prompt.device)
    full = torch.cat([prompt, gen], dim=1)
    gmask = (gen > 0).to(pmask.dtype)
    full_mask = torch.cat([pmask, gmask], dim=1)
    position_ids = _position_ids_from_mask(full_mask, full.shape[1])

    with torch.no_grad():
        out = model._sad_base_forward(
            model,
            input_ids=full,
            attention_mask=full_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    plen = prompt.shape[1]
    # positions plen-1 .. plen+L-2 predict response tokens 0 .. L-1
    logits_seq = out.logits[:, plen - 1:-1, :].float().detach()          # [N, L, V]
    hidden_seq = out.hidden_states[-1][:, plen - 1:-1, :].detach()       # [N, L, H]
    return logits_seq, hidden_seq


def sad_branch_features(model, main_ids, main_mask, presumm_ids, presumm_mask,
                        null_ids, null_mask, gen_result):
    """Cache the 3 branches' per-position (logits, hidden) for the rollout.

    Runs the frozen backbone once per branch (no_grad) over
    ``[branch_prompt ; response]`` and returns a dict of six detached tensors
    (main/presumm/null × logits[N,L,V] / hidden[N,L,H]). Reused across every PPO
    inner epoch — only the cheap FC re-mix in ``sad_logprobs_from_features``
    re-runs per epoch, so the backbone cost is paid just once.
    """
    nr = gen_result.shape[0] // main_ids.shape[0]
    m_logits, m_hidden = _branch_seq_features(model, main_ids, main_mask, gen_result, nr)
    p_logits, p_hidden = _branch_seq_features(model, presumm_ids, presumm_mask, gen_result, nr)
    n_logits, n_hidden = _branch_seq_features(model, null_ids, null_mask, gen_result, nr)
    return {
        "m_logits": m_logits, "m_hidden": m_hidden,
        "p_logits": p_logits, "p_hidden": p_hidden,
        "n_logits": n_logits, "n_hidden": n_hidden,
    }


def sad_logprobs_from_features(model, feats, gen_result, deterministic=True):
    """Differentiable per-token logprob of ``gen_result`` under the context-aware
    policy, re-mixing the cached branch features with the CURRENT FC head.

    Gradient flows only through the FC head (my_all_f/my_all_f1/my_f) — the branch
    logits/hidden are detached constants — exactly matching SARA's training setup
    where only the ``my_*`` layers are trainable. With ``deterministic`` the FC
    dropout is disabled during the call (restored after) so the importance ratio
    old/new is well-defined across PPO epochs. Returns ``[N, L]``.
    """
    was_training = model.dropout.training
    if deterministic:
        model.dropout.eval()
    try:
        weight = _weight_from_hidden(model, feats["m_hidden"], feats["p_hidden"],
                                     feats["n_hidden"])              # [N, L, 3]
        ab = torch.softmax(weight[..., :2], dim=-1)
        alpha, beta = ab[..., 0:1], ab[..., 1:2]
        gamma = torch.sigmoid(weight[..., 2:3])
        combined = (1 + gamma) * (alpha * feats["m_logits"] + beta * feats["p_logits"]) \
            - gamma * feats["n_logits"]                             # [N, L, V]
        logp = torch.log_softmax(combined.float(), dim=-1)
        gen = gen_result.to(logp.device)
        new_logp = logp.gather(2, gen.unsqueeze(2)).squeeze(2)      # [N, L]
    finally:
        if deterministic and was_training:
            model.dropout.train()
    return new_logp
