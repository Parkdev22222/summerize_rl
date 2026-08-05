#!/usr/bin/env bash
# =============================================================================
# 다중 백본(Llama / Qwen ...) × ablation 실험 러너
#
#   MODELS × ABLATIONS 매트릭스로:
#     1) test_performance_decoder_new_fc.py 로 딱 --max_train_iters 스텝만 학습
#        (모델별 --alpha_et_hidden_size = 그 모델 hidden_size 로 맞춰 oproj 워밍업이 동작)
#     2) compare_gemini.py 로 test set 전체 평가 (ROUGE + Gemini 항목별 점수)
#        -> results/multi/<model>/<ablation>.json 에 모든 스코어 저장
#   마지막에 전체를 모아 model×ablation 비교표 + combined_summary.json 생성.
#
#   SARA 의 SAD 헤드는 model-agnostic 이라 백본은 --model_name_or_path 만 바꾸면 됩니다
#   (utils.py 의 _sad_backbone 이 exaone/llama/qwen 을 SAD 로더로 라우팅).
#
# 사용법 (src 디렉토리 기준, GEMINI_API_KEY 필요):
#   export GEMINI_API_KEY=...
#   # Llama 는 gated -> HuggingFace 로그인/토큰 필요:
#   export HF_TOKEN=hf_...        (또는  huggingface-cli login)
#   bash run_ablation_multimodel.sh
#
#   DO_TRAIN=0 ...                 학습 건너뛰고 기존 체크포인트로 평가만
#   ONLY_MODELS="qwen3_8b" ...     특정 모델만
#   ONLY="full nojudge" ...        특정 ablation 만
#   GPU=1 ...                      다른 GPU
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------- 공통 설정 -----------------------------------------
DATASET="${DATASET:-summarize_rl_ko}"
GPU="${GPU:-0}"
MAX_ITERS="${MAX_ITERS:-120}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash}"
ACC_W="${ACC_W:-2.0}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
TRAIN_GEMINI_VAL="${TRAIN_GEMINI_VAL:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

CKPT_ROOT="../checkpoints/multi"
RUN_ROOT="../runs/multi"
LOG_ROOT="../logs/multi"
RESULT_ROOT="../results/multi"
mkdir -p "$CKPT_ROOT" "$RUN_ROOT" "$LOG_ROOT" "$RESULT_ROOT"

# ----------------------- 백본 모델 정의 --------------------------------------
# 형식: "짧은이름|HuggingFace_ID|hidden_size"
#   * hidden_size 는 --alpha_et_hidden_size 로 쓰여 oproj 워밍업(hidden×hidden)이 동작하게 함.
#   * 둘 다 hidden 4096 -> EXAONE-3.5(4096)과 같은 스케일로 직접 비교 가능.
MODELS=(
  "llama3_8b|meta-llama/Llama-3.1-8B-Instruct|4096"   # Meta 최신 8B (gated: HF 토큰/라이선스 승인 필요)
  "qwen3_8b|Qwen/Qwen3-8B|4096"                       # Qwen 최신 8B (non-gated)
  # ---- 다른 옵션(원하면 주석 해제) ----
  # "exaone_7_8b|LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct|4096"   # 원래 백본 (재현/기준)
  # "qwen25_7b|Qwen/Qwen2.5-7B-Instruct|3584"                 # 안정적 7B
  # "llama3_3b|meta-llama/Llama-3.2-3B-Instruct|3072"         # 3B 사이즈 스윕 (gated)
  # "qwen25_3b|Qwen/Qwen2.5-3B-Instruct|2048"                 # 3B 사이즈 스윕 (non-gated)
)

# ----------------------- ablation 정의 --------------------------------------
# "이름 | 학습 시 리워드/초기화 인자(베이스 대비 바뀌는 부분)"
ABLATIONS=(
  "full|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init oproj"
  "nojudge|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.0 --fc_init oproj"
  "notriplet|--RougeL_reward_weight 1.0 --triplet_coverage_weight 0.0 --judge_weight 0.3 --fc_init oproj"
  "rougeonly|--RougeL_reward_weight 1.0 --triplet_coverage_weight 0.0 --judge_weight 0.0 --fc_init oproj"
  "norouge|--RougeL_reward_weight 0.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init oproj"
  "fcnone|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init none"
)

# ----------------------- 사전 점검 ------------------------------------------
if [[ "$DO_EVAL" == "1" && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[ERROR] 평가에는 GEMINI_API_KEY (또는 GOOGLE_API_KEY) 가 필요합니다. DO_EVAL=0 이면 학습만." >&2
  exit 1
fi
# Llama(gated) 사용 시 토큰 경고
if printf '%s\n' "${MODELS[@]}" | grep -qi 'meta-llama' ; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
    echo "[WARN] Llama 는 gated 모델입니다. HF 라이선스 승인 + 토큰이 없으면 다운로드가 실패합니다." >&2
    echo "       해결: HF 페이지에서 접근 승인 후  export HF_TOKEN=hf_...  (또는 huggingface-cli login)" >&2
  fi
fi

train_one () {
  local hf_id="$1" hidden="$2" ckpt="$3" name="$4" extra="$5" tag="$6"
  echo "================ [$tag] TRAIN ($hf_id, hidden=$hidden, iters=$MAX_ITERS) ================"
  local gemini_train_args="" gemini_env=""
  if [[ "$TRAIN_GEMINI_VAL" == "1" ]]; then
    gemini_train_args="--gemini_val_n 30 --gemini_val_every 1 --gemini_val_model $GEMINI_MODEL --gemini_val_accuracy_weight $ACC_W"
  else
    gemini_env="env -u GEMINI_API_KEY -u GOOGLE_API_KEY"   # 학습 중 Gemini val 끔(속도/비용)
  fi
  # shellcheck disable=SC2086
  $gemini_env uv run --extra tb test_performance_decoder_new_fc.py \
    --do_train --dataset "$DATASET" --model_name_or_path "$hf_id" \
    --loading_mode bf16 --context_aware_decoding_alpha 1.0 \
    --batch_size 2 --num_return_sequences 2 \
    --epoch_num 286 --warmup_step 200 --warmup_train_step 20000 \
    --max_train_iters "$MAX_ITERS" \
    --lr 5e-5 --max_grad_norm 1.0 --accumulation_steps 1 \
    --max_input_length 1800 --min_new_tokens 30 --max_new_tokens 200 \
    --do_sample --sample_top_k 50 --sample_top_p 0.9 \
    --alpha_et_hidden_size "$hidden" --my_fc_fp32 --dropout_rate 0.0 \
    --judge_max_new_tokens 384 --factkb_weight 0.0 \
    $extra $gemini_train_args \
    --save_checkpoint_path "$ckpt" --save_checkpoint_every 10 \
    --logging "$LOG_ROOT/train_${tag}.log" --id "abl_${tag}" \
    --tensorboard_logdir "$RUN_ROOT/${name}/${tag}" --device cuda:0 \
    2>&1 | tee "$LOG_ROOT/train_${tag}.stdout"
}

eval_one () {
  local hf_id="$1" hidden="$2" ckpt="$3" out="$4" tag="$5"
  echo "================ [$tag] EVAL (compare_gemini -> JSON) ================"
  # greedy 디코딩(재현성) + Gemini seed 고정
  uv run compare_gemini.py \
    --dataset "$DATASET" --model_name_or_path "$hf_id" \
    --loading_mode bf16 --max_input_length 1800 \
    --save_checkpoint_path "$ckpt" --load_best 1 \
    --alpha_et_hidden_size "$hidden" --my_fc_fp32 \
    --min_new_tokens 30 --max_new_tokens 200 \
    --context_aware_decoding_alpha 1.0 --plausibility_alpha 0.1 \
    --gemini_model "$GEMINI_MODEL" --accuracy_weight "$ACC_W" \
    --out "$out" \
    2>&1 | tee "$LOG_ROOT/eval_${tag}.stdout"
  echo "[$tag] 결과 저장 -> $out"
}

# ----------------------- 실행 매트릭스 --------------------------------------
for mentry in "${MODELS[@]}"; do
  IFS='|' read -r mname hf_id hidden <<< "$mentry"
  if [[ -n "${ONLY_MODELS:-}" ]] && [[ " $ONLY_MODELS " != *" $mname "* ]]; then
    continue
  fi
  for aentry in "${ABLATIONS[@]}"; do
    aname="${aentry%%|*}"
    extra="${aentry#*|}"
    if [[ -n "${ONLY:-}" ]] && [[ " $ONLY " != *" $aname "* ]]; then
      continue
    fi
    tag="${mname}__${aname}"
    ckpt="$CKPT_ROOT/$mname/$aname"
    out="$RESULT_ROOT/$mname/$aname.json"
    mkdir -p "$ckpt" "$RESULT_ROOT/$mname"
    if [[ "$DO_TRAIN" == "1" ]]; then train_one "$hf_id" "$hidden" "$ckpt" "$mname" "$extra" "$tag"; fi
    if [[ "$DO_EVAL"  == "1" ]]; then eval_one  "$hf_id" "$hidden" "$ckpt" "$out" "$tag"; fi
  done
done

# ----------------------- 요약 취합 ------------------------------------------
if [[ "$DO_EVAL" == "1" ]]; then
  echo ""
  echo "================ MODEL x ABLATION 요약 ================"
  RESULT_ROOT="$RESULT_ROOT" python3 - <<'PY'
import glob, json, os

root = os.environ["RESULT_ROOT"]
files = sorted(glob.glob(os.path.join(root, "*", "*.json")))
rows, combined = [], {}
for f in files:
    model = os.path.basename(os.path.dirname(f))
    abl = os.path.splitext(os.path.basename(f))[0]
    try:
        s = json.load(open(f, encoding="utf-8")).get("summary", {})
    except Exception as e:
        print(f"  [skip] {model}/{abl}: {e}"); continue
    combined[f"{model}/{abl}"] = s
    jw = s.get("judge_win_by_weighted_avg", {})
    dec = jw.get("llm_mlp", 0) + jw.get("llm", 0)
    winrate = 100.0 * jw.get("llm_mlp", 0) / dec if dec else float("nan")
    sm = s.get("score_llm_mlp_avg", {}); rm = s.get("rouge_llm_mlp_avg", {})
    rows.append({
        "model": model, "ablation": abl,
        "judge_winrate": winrate,
        "mine_weighted": sm.get("weighted", float("nan")),
        "mine_acc": sm.get("accuracy", float("nan")),
        "mine_cov": sm.get("coverage", float("nan")),
        "mine_brev": sm.get("brevity", float("nan")),
        "mine_rougeL": rm.get("rougeLsum", float("nan")),
    })

hdr = f'{"model":<12}{"ablation":<11}{"win%":>7}{"MINE가중":>9}{"정확성":>8}{"누락":>7}{"간결성":>8}{"RLsum":>8}'
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["model"], x["ablation"])):
    print(f'{r["model"]:<12}{r["ablation"]:<11}{r["judge_winrate"]:>7.1f}'
          f'{r["mine_weighted"]:>9.2f}{r["mine_acc"]:>8.2f}{r["mine_cov"]:>7.2f}'
          f'{r["mine_brev"]:>8.2f}{r["mine_rougeL"]:>8.3f}')

out = os.path.join(root, "combined_summary.json")
json.dump({"table": rows, "per_cell_summary": combined},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n[combined] -> {out}")
PY
fi

echo ""
echo "완료. 개별 스코어: $RESULT_ROOT/<model>/<ablation>.json  |  요약: $RESULT_ROOT/combined_summary.json"
