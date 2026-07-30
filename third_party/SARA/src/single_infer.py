"""Load trained SARA FC weights and run inference on ONE example (by index).

Training saves only the small SAD/FC head (``model-best_fc_layers.pth`` or a
numbered ``model-<iter>_fc_layers.pth``) on top of the frozen backbone. This
script loads those weights, picks a single row from the train/val/test split by
index, decodes it exactly as ``test()`` does, and prints the actual text:

    [INPUT]  the templated prompt fed to the model
    [GOLD]   the reference (gold) summary
    [PRED]   the model's generated summary

It reuses the same data + decode path as ``test_performance_decoder_new_fc.py``
(``pretokenize`` -> ``template_input_decoder`` -> SAD ``generate``), so the output
matches what the validation loop scores for ``val/*`` -- only here you see the
text, for one example, instead of aggregate ROUGE/FactKB.

Example (run from third_party/SARA/src):

    python single_infer.py \
        --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
        --dataset summarize_rl_ko \
        --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1 \
        --split test --index 0

Pick a specific (non-best) checkpoint with ``--load_best 0 --load_ckpt_num 1100``.
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, GenerationConfig

from utils import (
    configure_model_loading,
    get_null_input_decoder,
    load_dataset,
    presumm_input_decoder,
    pretokenize,
    template_input_decoder,
)

# Shared with training (test_performance_decoder_new_fc) so the main-branch input
# is built identically at train and inference time.
from reward_extras import main_input_slot, render_triplets


def add_model_decode_args(p):
    """Model-loading + decoding args shared by single_infer and compare_gemini."""
    # data / backbone
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset", default="summarize_rl_ko")
    p.add_argument("--data_type", default="default")
    p.add_argument("--loading_mode", default="bf16")
    p.add_argument("--max_input_length", type=int, default=1024)
    # what to put in the main branch (the [보고서] slot). Training always used the
    # raw document; the triplet variants are off-distribution experiments.
    p.add_argument("--input_mode", choices=["document", "document+triplets", "triplets"],
                   default="document",
                   help="main-branch input: 원문만 / 원문+triplet / triplet만. "
                        "keyfacts(presumm) branch is unchanged; drop it with "
                        "--ablation_presumm_sequence.")
    # checkpoint (FC weights only)
    p.add_argument("--save_checkpoint_path", required=True,
                   help="dir where training wrote model-*_fc_layers.pth")
    p.add_argument("--load_best", type=int, default=1,
                   help="1 -> model-best_fc_layers.pth; 0 -> use --load_ckpt_num")
    p.add_argument("--load_ckpt_num", type=str, default=None,
                   help="iteration tag when --load_best 0 (e.g. 1100)")
    # SAD/FC head hyperparameters -- MUST match training
    p.add_argument("--alpha_et_hidden_size", type=int, default=4096)
    p.add_argument("--dropout_rate", type=float, default=0.0)
    p.add_argument("--sqrt_dimension", type=int, default=1)
    p.add_argument("--sqrt_dimension_first", type=int, default=1)
    p.add_argument("--sqrt_dimension_second", type=int, default=1)
    p.add_argument("--sqrt_method", type=str, default="concate_dim")
    p.add_argument("--fc_init", type=str, default="none", choices=["none", "oproj"])
    p.add_argument("--my_fc_fp32", action="store_true")
    p.add_argument("--ablation_main_sequence", action="store_true")
    p.add_argument("--ablation_presumm_sequence", action="store_true")
    p.add_argument("--ablation_null_sequence", action="store_true")
    # generation (defaults mirror the eval config)
    p.add_argument("--min_new_tokens", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="raise if the summary is cut off mid-sentence")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    p.add_argument("--cad_full", type=float, default=1.0)
    p.add_argument("--cad_salience", type=float, default=0.0)
    p.add_argument("--plausibility_alpha", type=float, default=0.0,
                   help="byte-level plausibility floor: mask tokens the main branch "
                        "gives prob < alpha*max. Fixes garbled digits/`?` from the "
                        "contrastive tilt. 0=off (faithful to eval); try 0.1.")
    return p


def build_args():
    p = argparse.ArgumentParser(
        description="Single-example SARA inference from saved FC weights."
    )
    add_model_decode_args(p)
    # which example to test
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--index", type=int, default=0)
    return p.parse_args()


def load_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    return tok


def load_model(args):
    """Backbone + trained FC head, in eval mode. Returns the model."""
    print("loading model checkpoint")
    model = configure_model_loading(args)
    load_fc_weights(model, args)
    if args.my_fc_fp32:
        from test_performance_decoder_new_fc import convert_my_layers_to_fp32
        model = convert_my_layers_to_fp32(model)
    model.eval()
    return model


def build_gen_config(args, tokenizer):
    """GenerationConfig with the same fields the eval path builds."""
    return GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        early_stopping=False,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        cad_full=args.cad_full,
        cad_salience=args.cad_salience,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ablation_main_sequence=args.ablation_main_sequence,
        ablation_presumm_sequence=args.ablation_presumm_sequence,
        ablation_null_sequence=args.ablation_null_sequence,
        plausibility_alpha=args.plausibility_alpha,
    )


def prepare_example(split_row, tokenizer, args):
    """Build the branch-list row + display meta for one dataset row.

    Mirrors the training/eval pipeline (truncate -> template) and applies
    --input_mode to the main branch's [보고서] slot. Returns (row, meta) where
    ``row`` = [templated_input, summary, presumm(keyfacts), triplets] and ``meta``
    carries raw_source / triplet_text / templated_input / reference / truncation.
    """
    raw_source = split_row[0]
    triplets_raw = split_row[3] if len(split_row) > 3 else []
    triplet_text = render_triplets(triplets_raw)

    row = pretokenize([split_row], tokenizer, args.max_input_length)[0]
    truncated_source = row[0]

    # same main-branch construction as training (reward_extras.main_input_slot)
    try:
        doc_slot = main_input_slot(truncated_source, triplets_raw, args.input_mode)
    except ValueError as e:
        raise SystemExit(str(e))

    row = [template_input_decoder([doc_slot, *row[1:]], args.dataset)] + list(row[1:])
    doc_in_input = args.input_mode in ("document", "document+triplets")
    meta = {
        "raw_source": raw_source,
        "triplet_text": triplet_text,
        "templated_input": row[0],
        "reference": row[1],
        "source_was_truncated": doc_in_input and truncated_source.strip() != raw_source.strip(),
    }
    return row, meta


def generate_summary(model, tokenizer, gen_cfg, row, args, device, pure=False):
    """Decode one prepared row with the exact branch logic of test().

    ``pure=True`` forces plain single-branch generation (no presumm/null branches),
    which makes _sad_generate use combined = main logits with the FC/SAD head
    untouched -- i.e. the raw backbone LLM, the pure-LLM baseline. Returns
    (prediction_text, num_new_tokens)."""
    templated_input = row[0]
    tok_in = tokenizer(
        [templated_input], return_tensors="pt", max_length=1800,
        padding=True, truncation=True,
    )
    input_len = tok_in.input_ids.shape[1]

    with torch.no_grad():
        if pure:  # raw backbone: no SAD combination, no MLP
            output = model.generate(
                input_ids=tok_in.input_ids.to(device),
                attention_mask=tok_in.attention_mask.to(device),
                generation_config=gen_cfg,
            )
        elif args.context_aware_decoding_alpha >= 0.0:  # full + salience + prompt
            tok_pre = tokenizer(
                [presumm_input_decoder(row, args.dataset)], return_tensors="pt",
                max_length=1800, padding=True, truncation=True,
            )
            tok_null = tokenizer(
                [get_null_input_decoder(row, args.dataset)], return_tensors="pt",
                max_length=1800, padding=True, truncation=True,
            )
            output = model.generate(
                input_ids=tok_in.input_ids.to(device),
                presumm_input=tok_pre.input_ids.to(device),
                null_inputs=tok_null.input_ids.to(device),
                attention_mask=tok_in.attention_mask.to(device),
                presumm_attention_mask=tok_pre.attention_mask.to(device),
                generation_config=gen_cfg,
            )
        elif args.cad_salience >= 0.0:  # full + salience
            tok_pre = tokenizer(
                [presumm_input_decoder(row, args.dataset)], return_tensors="pt",
                max_length=1800, padding=True, truncation=True,
            )
            output = model.generate(
                input_ids=tok_in.input_ids.to(device),
                presumm_input=tok_pre.input_ids.to(device),
                attention_mask=tok_in.attention_mask.to(device),
                presumm_attention_mask=tok_pre.attention_mask.to(device),
                generation_config=gen_cfg,
            )
        else:  # standard generation
            output = model.generate(
                input_ids=tok_in.input_ids.to(device),
                attention_mask=tok_in.attention_mask.to(device),
                generation_config=gen_cfg,
            )

    prediction = tokenizer.batch_decode(
        output[:, input_len:], skip_special_tokens=True, reduce_tokenization_space=True
    )[0]
    return prediction, output[:, input_len:].shape[1]


def load_fc_weights(model, args):
    """Load the three trained FC state_dicts into the SAD head (frozen backbone)."""
    tag = "best" if args.load_best else args.load_ckpt_num
    if tag is None:
        raise SystemExit("--load_best 0 requires --load_ckpt_num <iter>")
    fc_path = os.path.join(args.save_checkpoint_path, f"model-{tag}_fc_layers.pth")
    if not os.path.isfile(fc_path):
        raise SystemExit(f"checkpoint not found: {fc_path}")
    sd = torch.load(fc_path, map_location="cpu")
    model.my_all_f.load_state_dict(sd["my_all_f"])
    model.my_all_f1.load_state_dict(sd["my_all_f1"])
    model.my_f.load_state_dict(sd["my_f"])
    print(f"[ckpt] loaded FC weights <- {fc_path}")


def main():
    args = build_args()

    # 1) data -- same loader as training/eval
    train_set, val_set, test_set = load_dataset(args.dataset, args.data_type)
    split = {"train": train_set, "val": val_set, "test": test_set}[args.split]
    if not split:
        raise SystemExit(f"split '{args.split}' is empty")
    if not 0 <= args.index < len(split):
        raise SystemExit(
            f"index {args.index} out of range [0, {len(split)}) for split '{args.split}'"
        )

    tokenizer = load_tokenizer(args)
    row, meta = prepare_example(split[args.index], tokenizer, args)

    model = load_model(args)
    gen_cfg = build_gen_config(args, tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction, n_out = generate_summary(model, tokenizer, gen_cfg, row, args, device)

    hit_cap = n_out >= args.max_new_tokens
    garbled = "�" in prediction  # U+FFFD replacement char = broken byte-BPE output

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"[SPLIT] {args.split}   [INDEX] {args.index} / {len(split)}   [INPUT_MODE] {args.input_mode}")
    print(bar)
    trunc = " — TRUNCATED to --max_input_length" if meta["source_was_truncated"] else ""
    print(f"\n[SOURCE] (원문 원본{trunc})\n{meta['raw_source']}")
    fed = "in main input" if args.input_mode != "document" else "NOT fed to model"
    print(f"\n[TRIPLETS] ({fed})\n{meta['triplet_text'] or '(none)'}")
    print(f"\n[INPUT] (모델에 실제로 들어간 프롬프트)\n{meta['templated_input']}")
    print(f"\n[GOLD]\n{meta['reference']}")
    cap = " — HIT --max_new_tokens cap, likely cut off; raise it" if hit_cap else ""
    print(f"\n[PRED] ({n_out} new tokens{cap})\n{prediction}")
    if garbled and args.plausibility_alpha <= 0.0:
        print("\n[!] Output contains `�` (broken byte-BPE from the contrastive tilt). "
              "Re-run with --plausibility_alpha 0.1 to prune invalid byte continuations.")
    print(f"\n{bar}")


if __name__ == "__main__":
    main()
