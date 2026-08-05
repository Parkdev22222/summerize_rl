#!/usr/bin/env bash
# =============================================================================
# Ablation study runner (본 학습 120스텝 고정) + compare_gemini.py 전체 스코어 저장
#
#   각 ablation 마다:
#     1) test_performance_decoder_new_fc.py 로 딱 --max_train_iters 만큼만 학습
#        -> 각자 별도 체크포인트 디렉토리에 저장 (model-best_fc_layers.pth 등)
#     2) compare_gemini.py 로 test set 전체를 평가 (ROUGE + Gemini 항목별 점수)
#        -> results/ablation/<name>.json 에 모든 스코어 저장
#   마지막에 모든 <name>.json 의 summary 를 모아 비교표 + combined_summary.json 생성.
#
# 사용법 (src 디렉토리 기준, GEMINI_API_KEY 필요):
#   export GEMINI_API_KEY=...        # compare_gemini(Gemini judge) 에 필수
#   bash run_ablation.sh             # 전체 (학습+평가)
#   DO_TRAIN=0 bash run_ablation.sh  # 학습 건너뛰고 기존 체크포인트로 평가만
#   DO_EVAL=0  bash run_ablation.sh  # 학습만
#   GPU=1 bash run_ablation.sh       # 다른 GPU 사용
#   ONLY="full nojudge" bash run_ablation.sh   # 특정 ablation 만
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"          # 항상 src 디렉토리에서 실행 (../ 상대경로 기준 맞춤)

# ----------------------- 공통 설정 (필요시 수정) -----------------------------
MODEL="${MODEL:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
DATASET="${DATASET:-summarize_rl_ko}"
GPU="${GPU:-0}"
MAX_ITERS="${MAX_ITERS:-120}"            # 본 학습 스텝 수 (당신의 좋았던 짧은 학습 재현)
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash}"
ACC_W="${ACC_W:-2.0}"                    # 정확성 가중치 (학습 val 과 동일하게)
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
TRAIN_GEMINI_VAL="${TRAIN_GEMINI_VAL:-0}"  # 1이면 학습 중에도 Gemini val 로깅(느리고 API 비용↑)

export CUDA_VISIBLE_DEVICES="$GPU"       # 이 프로세스는 GPU 하나만 사용

CKPT_ROOT="../checkpoints/ablation"
RUN_ROOT="../runs/ablation"
LOG_ROOT="../logs/ablation"
RESULT_ROOT="../results/ablation"        # compare_gemini JSON 저장 위치
mkdir -p "$CKPT_ROOT" "$RUN_ROOT" "$LOG_ROOT" "$RESULT_ROOT"

# GEMINI_API_KEY 는 평가(compare_gemini)에 필수
if [[ "$DO_EVAL" == "1" && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[ERROR] 평가에는 GEMINI_API_KEY (또는 GOOGLE_API_KEY) 가 필요합니다." >&2
  echo "        export GEMINI_API_KEY=... 후 다시 실행하거나, DO_EVAL=0 로 학습만 하세요." >&2
  exit 1
fi

# ----------------------- ablation 정의 --------------------------------------
# 형식: "이름 | 학습 시 리워드/초기화 인자(베이스 대비 바뀌는 부분)"
#   full      : 당신 방법 전체 (기준선)  = ROUGE + triplet + judge, fc oproj
#   nojudge   : LLM judge 리워드 제거
#   notriplet : triplet coverage 리워드 제거
#   rougeonly : triplet + judge 둘 다 제거 (순수 SCST/ROUGE)
#   norouge   : ROUGE 리워드 제거 (triplet + judge 만)
#   fcnone    : FC 워밍업(oproj) 대신 랜덤 초기화
ABLATIONS=(
  "full|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init oproj"
  "nojudge|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.0 --fc_init oproj"
  "notriplet|--RougeL_reward_weight 1.0 --triplet_coverage_weight 0.0 --judge_weight 0.3 --fc_init oproj"
  "rougeonly|--RougeL_reward_weight 1.0 --triplet_coverage_weight 0.0 --judge_weight 0.0 --fc_init oproj"
  "norouge|--RougeL_reward_weight 0.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init oproj"
  "fcnone|--RougeL_reward_weight 1.0 --triplet_coverage_weight 1.0 --judge_weight 0.3 --fc_init none"
)

run_one () {
  local name="$1" extra="$2"
  local ckpt="$CKPT_ROOT/$name"
  mkdir -p "$ckpt"

  # ---------------- 1) 학습 (딱 MAX_ITERS 스텝) ----------------
  if [[ "$DO_TRAIN" == "1" ]]; then
    echo "================ [$name] TRAIN (max_train_iters=$MAX_ITERS) ================"
    local gemini_train_args="" gemini_env=""
    if [[ "$TRAIN_GEMINI_VAL" == "1" ]]; then
      gemini_train_args="--gemini_val_n 30 --gemini_val_every 1 --gemini_val_model $GEMINI_MODEL --gemini_val_accuracy_weight $ACC_W"
    else
      # 학습 중 Gemini val 끔(속도/비용). 실제 측정은 아래 compare_gemini 가 담당.
      gemini_env="env -u GEMINI_API_KEY -u GOOGLE_API_KEY"
    fi
    # shellcheck disable=SC2086
    $gemini_env uv run --extra tb test_performance_decoder_new_fc.py \
      --do_train --dataset "$DATASET" --model_name_or_path "$MODEL" \
      --loading_mode bf16 --context_aware_decoding_alpha 1.0 \
      --batch_size 2 --num_return_sequences 2 \
      --epoch_num 286 --warmup_step 200 --warmup_train_step 20000 \
      --max_train_iters "$MAX_ITERS" \
      --lr 5e-5 --max_grad_norm 1.0 --accumulation_steps 1 \
      --max_input_length 1800 --min_new_tokens 30 --max_new_tokens 200 \
      --do_sample --sample_top_k 50 --sample_top_p 0.9 \
      --alpha_et_hidden_size 4096 --my_fc_fp32 --dropout_rate 0.0 \
      --judge_max_new_tokens 384 --factkb_weight 0.0 \
      $extra $gemini_train_args \
      --save_checkpoint_path "$ckpt" --save_checkpoint_every 10 \
      --logging "$LOG_ROOT/train_$name.log" --id "abl_$name" \
      --tensorboard_logdir "$RUN_ROOT/$name" --device cuda:0 \
      2>&1 | tee "$LOG_ROOT/train_$name.stdout"
  fi

  # ---------------- 2) 평가 (compare_gemini -> 전체 스코어 JSON) ----------------
  if [[ "$DO_EVAL" == "1" ]]; then
    echo "================ [$name] EVAL (compare_gemini -> JSON) ================"
    # 재현성 위해 greedy 디코딩(--do_sample 없음). Gemini judge 는 seed 고정(기본 42).
    uv run compare_gemini.py \
      --dataset "$DATASET" --model_name_or_path "$MODEL" \
      --loading_mode bf16 --max_input_length 1800 \
      --save_checkpoint_path "$ckpt" --load_best 1 \
      --alpha_et_hidden_size 4096 --my_fc_fp32 \
      --min_new_tokens 30 --max_new_tokens 200 \
      --context_aware_decoding_alpha 1.0 --plausibility_alpha 0.1 \
      --gemini_model "$GEMINI_MODEL" --accuracy_weight "$ACC_W" \
      --out "$RESULT_ROOT/$name.json" \
      2>&1 | tee "$LOG_ROOT/eval_$name.stdout"
    echo "[$name] 결과 저장 -> $RESULT_ROOT/$name.json"
  fi
}

# ----------------------- 실행 루프 ------------------------------------------
for entry in "${ABLATIONS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  if [[ -n "${ONLY:-}" ]] && [[ " $ONLY " != *" $name "* ]]; then
    continue    # ONLY 로 지정된 것만 실행
  fi
  run_one "$name" "$extra"
done

# ----------------------- 요약 취합 (비교표 + combined_summary.json) -----------
if [[ "$DO_EVAL" == "1" ]]; then
  echo ""
  echo "================ ABLATION 요약 (compare_gemini summary 취합) ================"
  RESULT_ROOT="$RESULT_ROOT" python3 - <<'PY'
import glob, json, os

root = os.environ["RESULT_ROOT"]
files = sorted(glob.glob(os.path.join(root, "*.json")))
files = [f for f in files if os.path.basename(f) != "combined_summary.json"]

rows, combined = [], {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        s = json.load(open(f, encoding="utf-8")).get("summary", {})
    except Exception as e:
        print(f"  [skip] {name}: {e}"); continue
    combined[name] = s
    jw = s.get("judge_win_by_weighted_avg", {})
    dec = jw.get("llm_mlp", 0) + jw.get("llm", 0)
    winrate = 100.0 * jw.get("llm_mlp", 0) / dec if dec else float("nan")
    sm = s.get("score_llm_mlp_avg", {}); sb = s.get("score_llm_avg", {})
    rm = s.get("rouge_llm_mlp_avg", {})
    rows.append({
        "name": name,
        "judge_winrate": winrate,                       # MINE(LLM+MLP) 승률 %
        "mine_weighted": sm.get("weighted", float("nan")),
        "base_weighted": sb.get("weighted", float("nan")),
        "mine_acc": sm.get("accuracy", float("nan")),
        "mine_cov": sm.get("coverage", float("nan")),
        "mine_brev": sm.get("brevity", float("nan")),
        "mine_rougeL": rm.get("rougeLsum", float("nan")),
        "scored": s.get("judge_scored"),
    })

hdr = f'{"ablation":<12}{"win%":>7}{"MINE가중":>9}{"BASE가중":>9}{"정확성":>8}{"누락":>7}{"간결성":>8}{"RLsum":>8}{"채점":>6}'
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f'{r["name"]:<12}{r["judge_winrate"]:>7.1f}{r["mine_weighted"]:>9.2f}'
          f'{r["base_weighted"]:>9.2f}{r["mine_acc"]:>8.2f}{r["mine_cov"]:>7.2f}'
          f'{r["mine_brev"]:>8.2f}{r["mine_rougeL"]:>8.3f}{str(r["scored"]):>6}')

out = os.path.join(root, "combined_summary.json")
json.dump({"table": rows, "per_ablation_summary": combined},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n[combined] -> {out}")
PY
fi

echo ""
echo "완료. 개별 전체 스코어: $RESULT_ROOT/<name>.json  |  요약: $RESULT_ROOT/combined_summary.json"
