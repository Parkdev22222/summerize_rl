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
    """Single-branch forward: run the base transformer + lm_head once.

    Returns (CausalLMOutputWithPast, last_hidden_state) exactly like the fork's
    LlamaForCausalLM.forward_once, but resolves the base model and head via the
    generic get_decoder()/get_output_embeddings() accessors.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    decoder = self.get_decoder()
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )

    last_hidden_state = outputs.last_hidden_state  # [bs, seq, hidden]
    hidden_states = outputs[0]

    lm_head = self.get_output_embeddings()
    logits = lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return CausalLMOutputWithPast(
        loss=loss,
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
        {"forward": _sad_forward, "forward_once": _forward_once},
    )
    model.__class__ = sad_cls

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

    # Place the new layers on the head's device; fp32 for stable weight learning.
    try:
        dev = model.get_output_embeddings().weight.device
    except Exception:
        dev = next(model.parameters()).device
    dtype = torch.float32 if fc_fp32 else model.get_output_embeddings().weight.dtype
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
