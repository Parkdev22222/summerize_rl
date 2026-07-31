# summarize_rl ↔ SARA 통합 노트

`summarize_rl`의 reference-free 보상 3종을 SARA의 self-critical 보상에 이식하고,
SARA가 우리 레포의 한국어 데이터로 학습·테스트할 수 있게 한 통합이다.

## 추가된 보상 (기본 가중치 0 → 켜야 동작)

| 보상 | argparse | 방향 | 입력 |
|---|---|---|---|
| Triplet 커버리지 | `--triplet_coverage_weight` | + | 데이터의 `triplets`([head,rel,tail]) |
| LLM-as-judge (로컬 EXAONE) | `--judge_weight` | + | 원문 vs 요약, 항목별 sub-score(0~15) |

> Hallucination(환각) 페널티는 보상함수에서 제거됨. (judge의 "비조작" 항목이 유사 역할을 겸함.
> 헬퍼 `ungrounded_fact_penalty`/`calculate_ungrounded_fact_penalty`는 남아 있어 필요 시 재연결 가능.)

- judge는 **이미 로드된 로컬 EXAONE 백본**으로 채점한다(API 키·네트워크 불필요).
  내부적으로 presumm/null 없이 `model.generate`를 호출 → **단일 브랜치 순수 디코딩**(정책 FC
  미개입, 순수 base 판단), `torch.no_grad`로 호출해 RL grad에 영향 없음. (원문,요약) 캐시.
- **채점 방식 = keyfacts 체크리스트 + 항목별 sub-score**(단일 0~100은 한 값에 몰려 변별력 약함).
  세 기준을 각 0~5로 매겨 합산(0~15) 후 정규화:
  1) 정확성(부대명·병력·장비수량·날짜/지명 정확), 2) 비조작(원문에 없는 것 안 지어냄),
  3) **핵심포함 = 정답 keyfacts(상위 8개)를 체크리스트로 주고, 요약이 이를 (표현이 달라도)
     의미상 정확히 반영했는지** 채점 → 어휘 커버리지(triplet)가 못 잡는 **의미적 반영**을 봄.
  출력 `정확성/비조작/핵심포함/[총점]`을 파싱(합 우선 → [총점] → 마지막 0~15 정수).
  keyfacts는 데이터의 `keyfacts`(=presumm)를 reward까지 스레딩해 자동 공급(없으면 일반 기준으로 폴백).
- 비용: 출력·프롬프트가 길어져(`--judge_max_new_tokens` 기본 48) **스텝당 더 느려짐**(단일점수의 2~3배+).
  20k step 등 긴 학습이면 `--judge_weight 0.2~0.3` 소량 권장. 첫 실패는 `[judge] ...` 로그로 노출.
- 더 강한 옵션(후보 vs greedy comparative)은 품질↑이나 배선/속도 부담이 커 현재 미채택.
- 세 가중치가 모두 0이면 SARA의 원래 보상(ROUGE+FactKB)과 **완전히 동일**하다.

## 우리 데이터로 학습·테스트

`--dataset summarize_rl_ko` 를 주면 `data/scenarios_ko.jsonl`(train 140 / eval 10)과
`data/test_scenarios_ko.jsonl`을 읽는다. 매핑: `source_text→document`,
`summary_text→summary`, `keyfacts(개행 join)→presumm`, `triplets→triplet 보상 입력`.
데이터 경로는 `$SUMMARIZE_RL_DATA`로 재정의 가능(기본: 리포의 `data/`).

예시(스모크):
```bash
cd third_party/SARA/src
python test_performance_decoder_new_fc.py \
    --do_train --dataset summarize_rl_ko \
    --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
    --num_return_sequences 2 --batch_size 2 --debug_flag --debug_num 2 \
    --triplet_coverage_weight 1.0 --hallu_weight 0.5 \
    --judge_weight 0.3 --factkb_weight 0.0
```
judge를 켜면(`--judge_weight>0`) 시작 시 `[judge] ENABLED via local EXAONE backbone`이 뜬다.

**입력에 triplet 포함해서 학습(`--input_mode`)**: 기본은 `document`(원문만, 기존과 동일).
`--input_mode document+triplets`를 주면 학습·검증 모두 main 브랜치 [보고서] 슬롯에 원문 뒤로
`[관계 정보]`로 triplet을 붙여 넣는다(`triplets`면 triplet만). train/val/test에 **일괄 적용**되어
학습·평가 입력이 일치한다. keyfacts(presumm)·null 브랜치는 그대로. triplet은 여전히 보상
(`--triplet_coverage_weight`)에도 쓰이므로, 이 옵션은 triplet을 **입력으로도** 넣는 것.
```bash
python test_performance_decoder_new_fc.py --do_train --dataset summarize_rl_ko \
    --input_mode document+triplets ...   # 원문+triplet로 학습
```
⚠️ 이렇게 학습했으면 추론/비교도 **같은 `--input_mode`**로 해야 한다(single_infer.py,
compare_gemini.py 모두 `--input_mode` 지원). 렌더 형식은 학습·추론이 `reward_extras.main_input_slot`
하나를 공유하므로 어긋나지 않는다.

## TensorBoard로 성능 보기
SARA 원본엔 TensorBoard가 없어 추가했다. `--tensorboard_logdir <경로>`를 주면 매 학습 스텝마다
아래 스칼라를 기록한다(비우면 비활성).
- `train/loss` — RL 손실(`RewardCriterion`).
- `reward/rougeL` — 샘플 롤아웃의 평균 RougeLsum(주 보상).
- `reward/triplet_coverage`, `reward/hallu`, `reward/judge` — 각 보상/페널티의 샘플 평균
  (가중치>0으로 계산될 때만 의미 있는 값; 아니면 0).
- `reward/weighted_mean` — baseline 차감 전, 가중합된 보상의 샘플 평균(학습이 요약 품질을
  실제로 올리는지 보는 핵심 곡선).

```bash
# 학습에 로그 경로 지정
python test_performance_decoder_new_fc.py ... --tensorboard_logdir ../runs/exaone_ko_500
# 다른 터미널에서 (레포 루트 기준)
tensorboard --logdir third_party/SARA/runs
```
`reward/weighted_mean`이 우상향하면 정책(FC 가중치)이 더 나은 요약을 뽑도록 학습되는 것이다.
`train/loss`는 advantage·부호 때문에 0 근처를 오갈 수 있으니 품질 추세는 `reward/*`로 본다.

## 한 건만 추론해 실제 출력 보기 (single_infer.py)
학습이 저장한 FC 가중치(`model-best_fc_layers.pth`)만 불러, train/val/test 스플릿에서
**인덱스로 한 예제만** 골라 실제 요약 텍스트(`[INPUT]/[GOLD]/[PRED]`)를 찍는다. 학습/검증과
동일한 파이프라인(`pretokenize→template→SAD generate`)이라 `val/*`가 채점하는 것과 같은 디코딩이다.
```bash
cd third_party/SARA/src
python single_infer.py \
    --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
    --dataset summarize_rl_ko \
    --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1 \
    --split test --index 0
```
특정 스텝을 보려면 `--load_best 0 --load_ckpt_num 1100`. SAD head 하이퍼파라미터
(`--alpha_et_hidden_size` 등)는 **학습 때와 동일하게** 줘야 가중치 shape가 맞는다.
`[SOURCE]`(원문 원본)·`[TRIPLETS]`·`[INPUT]`(실제 프롬프트)·`[GOLD]`·`[PRED]`를 찍는다.

**main 브랜치 입력 바꾸기(`--input_mode`)**: 학습은 항상 원문(main)+keyfacts(presumm)로
디코딩하고 triplet은 보상에만 썼다. 추론 실험용으로 main 슬롯에 무엇을 넣을지 고를 수 있다.
- `document`(기본): 원문만 — 학습과 동일(in-distribution).
- `document+triplets`: 원문 뒤에 `[관계 정보]`로 triplet을 붙여서.
- `triplets`: triplet만 (원문 없이).

`document+triplets`/`triplets`는 학습이 본 적 없는 **off-distribution** 입력이라 결과는 탐색용이다.
keyfacts(presumm) 브랜치는 그대로 두며, 빼려면 `--ablation_presumm_sequence`를 함께 준다.
```bash
python single_infer.py ... --split test --index 0 --input_mode document+triplets
python single_infer.py ... --split test --index 0 --input_mode triplets
```
요약이 중간에 끊기면 `--max_new_tokens`(기본 256)를, 원문이 잘리면 `--max_input_length`(기본
1024)를 올린다. `[PRED]`에 `HIT --max_new_tokens cap` 경고가 뜨면 출력이 잘린 것.

**숫자/글자가 `�`로 깨질 때**: SAD 결합이 `(1+gamma)*(alpha*main+beta*presumm) - gamma*null`로
null 분포를 빼는 대조 디코딩이라, EXAONE의 바이트 레벨 BPE에서 **유효하지 않은 UTF-8 바이트
연속 토큰**을 골라 깨진 문자가 나올 수 있다(특히 숫자). `_sad_generate`에 바이트 레벨
plausibility floor를 추가했다(`--plausibility_alpha`, 기본 0=off라 학습/평가 디코딩은 불변).
`--plausibility_alpha 0.1`을 주면 main(원문 기반, 항상 UTF-8 유효) 분포가 implausible하다고
보는 토큰을 마스킹해 깨진 바이트를 잘라낸다. single_infer는 `�` 감지 시 이 옵션을 안내한다.

## 순수 EXAONE vs 내 방법 승률 (compare_gemini.py)
두 요약 모두 **여기서 올리는 같은 로컬 EXAONE**에서 나온다. 차이는 학습된 MLP뿐이다.
Gemini는 요약이 아니라 **심판**만 한다. single_infer의 모델 로딩·디코딩 함수를 그대로 재사용한다.
- **MINE**: EXAONE + 학습된 SAD/FC head, 3-branch context-aware 디코딩.
- **BASE**: 순수 EXAONE — 같은 프롬프트, 단일 브랜치 일반 생성(MLP·CAD 없음). `presumm`/`null`을
  안 넣으면 `_sad_generate`가 `combined=main`으로 떨어져 FC head가 개입하지 않는다(=raw 백본).
- **판정(Gemini, 항목별 점수)**: 각 요약을 **독립적으로** 정확성/누락/간결성 **1~5점** 채점한다
  (후보를 따로 채점 → 위치 편향 없음, 두 요약 다 Gemini 것이 아니라 자기선호 편향도 없음).
  정확성은 **원자 단위 O/X 체크리스트**로 채점한다: Gemini가 원문의 검증 가능한 사실
  (부대별 보유 무기체계·수량, 병력, 사상자)을 항목별로 나열하고 요약에 정확히(이름+수량)
  반영됐는지 O/X 표시 → **적중률(O수/전체)을 코드가 1~5점으로 환산**(LLM 산술 변동 제거,
  재현성↑). 무기체계 누락·수량 생략·수치 오기입은 모두 X. 예제마다 항목별 점수를
  출력하고, **정확성에 가중치를 준 가중평균**(`--accuracy_weight`, 기본 2.0 → 정확성 ×2,
  누락·간결성 ×1)이 높은 쪽을 승자로 집계. 끝에 MINE/BASE **항목별 평균 표 + 가중평균 + 승률**을
  낸다. `--accuracy_weight 1`이면 단순 평균. (택1 방식보다 실행 간 결과가 덜 흔들린다.)
```bash
GEMINI_API_KEY=... python compare_gemini.py \
    --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
    --dataset summarize_rl_ko \
    --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1 \
    --plausibility_alpha 0.1 --out ../runs/compare_gemini.jsonl
```
`pip install google-genai`(또는 `google-generativeai`) 필요. `--limit N`으로 앞 N개만,
`--out`으로 예제별 요약·판정 JSONL 저장. 참고: MINE은 keyfacts(presumm)도 받지만 BASE는 보고서
프롬프트만 본다 → "MLP 단독"이 아니라 "전체 방법 vs 순수 백본" 비교다.

**gold-ROUGE 비교(로컬, API 불필요)**: 기본으로 각 예제의 MINE·BASE 요약을 **gold와
ROUGE-1/2/Lsum**으로 재서 평균과 승률(added=R1+R2+RLsum, 학습이 최적화한 지표)을 같이 낸다.
`--skip_judge`면 Gemini 없이 **ROUGE만** (결제/키 불필요), `--no_rouge`면 ROUGE를 뺀다.
`--out result.json`을 주면 **전체 테스트셋**(기본 `--limit 0`)을 단일 JSON으로 저장한다:
`{"summary":..., "results":[{"source"(원문), "llm_summary"(순수 LLM), "llm_mlp_summary"(LLM+MLP),
"gold", rouge_*, *_winner}, ...]}`.
```bash
# Gemini 없이 "MLP가 학습 지표(ROUGE)에서 BASE를 이기나"만 로컬로 확인
python compare_gemini.py --skip_judge \
    --model_name_or_path LGAI-EXAONE/... --loading_mode bf16 \
    --dataset summarize_rl_ko --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1
```
해석: MINE ROUGE > BASE인데 Gemini만 지면 **지표 불일치/디코딩 손해**, MINE ROUGE ≤ BASE면
**학습이 안 된 것**. `Evaluator.calculate_rouge`(torchmetrics)를 재사용해 학습·검증과 같은 계산이다.

## 백본: EXAONE 3.5 Instruct (SAD head 이식)

SARA의 context-aware 디코딩은 원래 fork가 **아키텍처별로 `*ForCausalLM`을 직접 수정**해
구현한다(`modeling_llama.py`의 `LlamaForCausalLM`: FC 3층 `my_all_f/my_all_f1/my_f` +
main/presumm/null 3브랜치 `forward` → `weight[bs,3]` 반환). 결합은 fork의
`generation/utils.py: sample()`에서:
```
alpha,beta = softmax(weight[:, :2]);  gamma = sigmoid(weight[:, 2])
logits = (1+gamma)*(alpha*main + beta*presumm) - gamma*null
```

EXAONE 3.5는 fork에 없는 자체 아키텍처(`trust_remote_code`)라 정적 modeling 파일을 고칠
수 없다. 대신 **동일한 SAD head를 로드시점에 주입**한다:

- `src/modeling_exaone_sad.py`
  - `forward_once`/`forward`를 `self.get_decoder()`·`self.get_output_embeddings()`만
    사용하도록 **모델 비의존**으로 구현(EXAONE의 base transformer/lm_head를 그대로 사용).
  - `add_sad_head(model, config)`: 로드된 `ExaoneForCausalLM`을 SAD 서브클래스로 rebless
    하고 FC 3층을 head device에 fp32로 부착.
  - `load_exaone_sad(args)`: `trust_remote_code`로 EXAONE 로드 → SAD head 부착.
- `src/utils.py: configure_model_loading*`에 `'exaone' in model_name` 분기 추가 →
  `load_exaone_sad` 호출.

사용: `--model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` (또는 2.4B) 지정.

### FC head 초기화 옵션 (`--fc_init`, 기본 `none`)
정책망(FC 3층)은 백본이 frozen이라 **유일하게 학습되는 부분**이고, 기본은 랜덤 init이다.
`--fc_init oproj`를 주면 **`my_all_f1`(hidden×hidden)만** 백본의 **마지막 디코더 블록 어텐션
출력 투영(`o_proj`/`out_proj`, hidden×hidden)** 에서 warm-start 한다(bias는 0). 세 FC층 중
shape가 백본 가중치와 정확히 일치하는 건 `my_all_f1` 뿐이라 여기만 이식 대상이다
(`my_all_f`는 입력이 3×hidden, `my_f`는 출력 3차원이라 대응 가중치 없음).
- 구현: `modeling_exaone_sad.py`의 `_find_last_output_proj`(Llama식 `o_proj`·EXAONE식
  `out_proj` 모두 지원, 최상위 레이어 선택) + `_init_linear_from_output_proj`. `load_exaone_sad`가
  `config.fc_init`로 전달, `add_sad_head`가 적용. 대상 못 찾으면 랜덤 init 유지(무해).
- 주의: 이 FC는 "브랜치 혼합 게이트"라 o_proj 이식이 학습을 개선한다는 이론적 보장은 약함
  (실험용). 적용 시 콘솔에 `[SAD] my_all_f1 warm-started ...` 로그가 뜬다.

### fork 불필요 — 자체 context-aware generate 사용
원래 SARA의 3브랜치 결합은 **fork(transformers 4.36)의 `generation/utils.py: sample()`** 에만
있었다. 하지만 EXAONE 3.5는 최신 transformers에서만 로드되므로, fork에 의존하지 않도록
**`modeling_exaone_sad.py`에 동일한 결합 로직의 자체 `generate`(`_sad_generate`)를 구현**해
SAD 모델에 바인딩했다. 덕분에 **사용자의 최신 transformers 그대로**에서 동작한다(구버전 fork
설치 불필요).

- `_sad_generate`: main/presumm/null 3브랜치를 각자 KV캐시로 스텝 생성하며 매 스텝
  `(1+γ)(α·main+β·presumm)−γ·null`로 결합. `.sequences`(프롬프트+생성)·`.scores`(스텝별
  결합 logits) 반환 — SARA 학습 루프가 기대하는 형식 그대로. top-k/top-p/temperature,
  min/max_new_tokens, EOS, `num_return_sequences` 지원. presumm/null이 없으면 단일 브랜치로
  폴백(테스트 경로 안전).
- 캐시/포지션은 base decoder의 **표준 forward 인자**(`input_ids/attention_mask/position_ids/
  past_key_values/cache_position`)만 사용 → transformers 버전에 견고. 좌측 패딩 대응.

### ⚠️ 실 하드웨어 검증 권장
- 검증된 부분(torch로 실제 PASS): `tests/test_exaone_sad.py`
  - `test_sad_head_forward_contract`: forward가 `(main,presumm,null,weight[bs,3])` 반환 + 결합 유한값.
  - `test_sad_generate_contract`: `generate`가 `.sequences[bs·nrs, prompt+gen]`·`.scores`(스텝별
    `[bs·nrs, vocab]`) 반환, 프롬프트 슬라이싱으로 생성 토큰 추출.
- 미검증: 실제 EXAONE 가중치로 GPU에서 몇 스텝 디코딩. 최신 transformers의 EXAONE base
  forward가 위 표준 인자를 그대로 받는지만 GPU에서 1회 확인하면 된다(대부분 그대로 동작).
- 보상 3종·데이터 어댑터: `tests/test_reward_extras.py` 14/14 통과.

## 검증

```bash
python third_party/SARA/tests/test_reward_extras.py     # pytest 없이도 실행
# 또는
python -m pytest third_party/SARA/tests/ -q             # torch 불필요(stdlib만)
```
