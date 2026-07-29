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

## 강화학습 알고리즘: SCST(기본) / GRPO (`--rl_algo`)

`--rl_algo scst`(기본)는 SARA 원본 self-critical: advantage = `sample − greedy`, REINFORCE
(`RewardCriterion`). `--rl_algo grpo`로 **정식 GRPO** 사용:
- **advantage**: greedy 대신 **그룹**(같은 프롬프트의 `--num_return_sequences` 롤아웃) 표준화
  `(r − group_mean)/(group_std + eps)` (`--grpo_adv_eps` 기본 1e-4).
- **loss**: PPO clipped surrogate `-min(ratio·A, clip(ratio,1±ε)·A)` (`--grpo_clip_eps` 0.2) +
  **KL(reference)** (`--grpo_kl_beta` 0.04, k3 추정자 `exp(Δ)−Δ−1 ≥ 0`).
- **reference = frozen base LM**(context-aware FC 미적용). `_sad_base_forward`로 prompt+rollout을
  teacher-forcing 1회 스코어링(detached).
- **완전한 multi-epoch PPO (`--grpo_mu`, 기본 4) — clip 실작동**: 롤아웃을 **업데이트된 정책으로
  재스코어링**해 매 inner epoch마다 `new_logp`를 다시 구한다. 그래서 epoch가 진행되며 π_θ가
  π_old에서 벌어지고 **clip이 실제로 문**(`reward/clipfrac` 곡선으로 확인). epoch 0(및 μ=1)에선
  π_θ==π_old → ratio=1 → clip 무효(= 그룹-advantage PG + KL, DeepSeekMath 1-iteration).
- **재스코어링(핵심)**: SARA 정책의 결합 logit은 원래 `_sad_generate`가 **토큰 단위로만** 만들어
  임의 시퀀스 재점수화가 불가능했음. 해결: 세 브랜치(main/presumm/null)를 `[prompt;response]`로
  teacher-forcing forward → `_weight_from_hidden`(Linear이라 `[N,L,H]`에 broadcast)로 **per-position
  weight `[N,L,3]`** → `(1+γ)(α·m+β·p)−γ·n` 결합 → per-token logprob(미분가능). **백본이 frozen**이라
  세 브랜치 backbone forward는 **배치당 1회(no_grad) 캐시**하고 inner epoch에선 값싼 FC 재결합만 →
  μ>1이 거의 공짜. FC dropout은 재스코어 중 off(결정적 ratio). 구현: `src/modeling_exaone_sad.py`의
  `sad_branch_features`·`sad_logprobs_from_features`.
- **미니배치(`--grpo_minibatch_size`, 기본 0=full-batch)**: >0이면 매 epoch 롤아웃 N행을 섞어
  청크로 나눠 업데이트(표준 PPO). GRPO는 자체 optimizer 루프를 소유하고 scheduler는 **배치당 1회**
  step(SCST와 동일한 스케줄).
- 구현: `src/grpo_extras.py`(`group_normalized_advantage`·`GRPOLoss`(clip+KL, `.last_kl`·
  `.last_clipfrac`)·`reference_logprobs`). `--rl_algo scst`면 기존 경로와 **바이트 동일**(무영향).
  TB에 `reward/kl`·`reward/clipfrac` 로깅.
- ⚠️ 재스코어링·grad 흐름은 **실 하드웨어(GPU+EXAONE) 스모크 권장**: epoch 0에서 old≈new(ratio≈1)
  확인 → epoch>0에서 ratio 이탈 + `reward/clipfrac`↑ 확인. 단위테스트는 advantage 표준화·loss grad·
  KL≥0·ref 정렬·**clip 활성(clipfrac, 클립영역 grad=0)**·**재스코어 수식/정렬/FC-only grad**를 커버.

예: `--rl_algo grpo --num_return_sequences 4 --grpo_mu 4 --grpo_kl_beta 0.04 --grpo_clip_eps 0.2`
(그룹이 의미 있으려면 `num_return_sequences ≥ 2`, 클수록 baseline 안정. `--grpo_mu>1`이 clip을 활성화).

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
