# SARA (본 코드) 방법론 & 구현 완전 해설

> 군사 도메인 한국어 요약을 위한 **SARA (Salience-Aware Reinforced Adaptive decoding)** 구현 전체를,
> 수식 · 코드 위치 · 한국어 예시와 함께 빠짐없이 설명합니다.
> 대상 파일: `third_party/SARA/src/` 아래 `modeling_exaone_sad.py`, `test_performance_decoder_new_fc.py`,
> `reward_extras.py`, `single_infer.py`, `compare_gemini.py`, `utils.py`.

---

## 0. 한 장 요약 (큰 그림)

- **백본 LLM은 통째로 얼립니다(frozen).** 학습하는 건 아주 작은 **MLP 하나(SAD 헤드)** 뿐입니다.
- 같은 LLM에 입력을 **세 가지로 바꿔 3번** 돌립니다: **원문(main) / 핵심사실(presumm) / 빈입력(null)**.
- MLP이 세 실행의 hidden state를 보고 **조합 가중치 $(\alpha,\beta,\gamma)$** 를 매 토큰마다 뱉고, 세 로짓을 대비(contrastive)해서 다음 토큰 분포를 만듭니다.
- 정답 요약이 필요 없는 **강화학습(SCST)** 으로, **ROUGE + triplet 커버리지 + LLM 심사(judge)** 를 보상으로 삼아 MLP만 업데이트합니다.
- MLP은 백본에 독립적이라 **EXAONE / Llama / Qwen 어디에나** 붙습니다.

```
                 ┌───────────── 얼어있는 LLM (학습 안 함) ──────────────┐
   원문 ────────▶│ main 브랜치   ─▶ hidden_main,  logits_main          │
   핵심사실 ─────▶│ presumm 브랜치─▶ hidden_pre,   logits_pre           │
   빈입력 ───────▶│ null 브랜치   ─▶ hidden_null,  logits_null          │
                 └──────────────────────────────────────────────────┘
                          │ (세 hidden 이어붙임)
                          ▼
                 ┌── 학습되는 작은 MLP (SAD 헤드) ──┐
                 │  weight = [wα, wβ, wγ] (3차원)   │
                 └────────────────────────────────┘
                          │
        α,β = softmax(wα,wβ),   γ = sigmoid(wγ)
                          ▼
   combined = (1+γ)·(α·logits_main + β·logits_pre) − γ·logits_null
                          ▼
        plausibility floor(깨진 바이트 컷) ▶ 다음 토큰 선택
```

---

## 1. 문제 설정 & 데이터

### 1.1 태스크
긴 **군사 상황보고 원문**을 짧고 정확한 **요약**으로 만든다. 정확성(부대·무기체계·수량·지명·시간)이 최우선.

### 1.2 데이터 한 줄(row)의 구조
`reward_extras.py:_read_ours` / `utils.py:load_dataset` 기준, 한 예시는 리스트로 표현됩니다:

```
row = [ document,      summary,        presumm,         triplets ]
        row[0]=원문     row[1]=정답요약  row[2]=핵심사실    row[3]=KB 트리플
```

- **document(원문)**: 요약 대상 보고서 전체.
- **summary(정답요약)**: 학습에서 강화학습 보상의 ROUGE 계산 기준(정답). *생성에 직접 넣지 않음.*
- **presumm(핵심사실 = keyfacts)**: 원문에서 뽑은 핵심 사실 문장들(줄바꿈으로 연결). presumm 브랜치 입력과 judge 체크리스트로 쓰임.
- **triplets(KB 트리플)**: `[head, relation, tail]` 목록. triplet-coverage 보상과 (옵션) 입력에 쓰임.

> ⚠️ 이 데이터는 **템플릿 기반 합성 데이터**입니다 (`data/gen_test_scenarios.py`). 원문·요약·keyfacts·triplets가
> **같은 난수 슬롯**에서 생성돼서, keyfacts는 "추출된" 것이 아니라 애초에 정답으로 주어진 값입니다.

### 1.3 예시 (개념용)
```
원문:    "아군 제3기계화보병대대는 대전차미사일 5기, 박격포 10문을 보유하며 병력은 1,200명이다. ...(중략)..."
정답요약: "제3기보대대(병력 1,200명)는 대전차미사일 5기·박격포 10문 보유, 방어태세 유지."
핵심사실: ["제3기계화보병대대 병력 1,200명", "대전차미사일 5기", "박격포 10문"]
트리플:   [["제3기계화보병대대","보유","대전차미사일 5기"], ["제3기계화보병대대","보유","박격포 10문"]]
```

---

## 2. 전체 구조: 얼어있는 LLM + 작은 MLP

핵심 아이디어: **큰 LLM은 그대로 두고(파라미터 고정), 아주 작은 MLP만 강화학습으로 학습**합니다.
그 MLP이 "세 가지 관점(원문/핵심/빈칸)의 다음-토큰 예측을 어떻게 섞을지"를 매 토큰마다 결정합니다.

- 학습되는 파라미터: `my_all_f`, `my_all_f1`, `my_f` (아래 3개 Linear) — 전부 합쳐 수백만 개 수준.
- 학습 안 되는 것: 백본 LLM(수십억 개) 전부.

코드: `modeling_exaone_sad.py:add_sad_head` 가 로드된 `*ForCausalLM` 을 `*_SAD` 서브클래스로 바꾸고 위 3개 Linear를 붙입니다.

---

## 3. 세 브랜치 (main / presumm / null)

"브랜치"는 **같은 얼어있는 LLM에 서로 다른 입력을 넣어 돌린 실행**입니다. `utils.py`의 세 함수가 입력 텍스트를 만듭니다.

| 브랜치 | 만드는 함수 | 슬롯에 들어가는 것 | 의미 |
|---|---|---|---|
| **main** | `template_input_decoder` (row[0]) | 원문 전체 | 진짜 근거. 가장 믿을 만함 |
| **presumm** | `presumm_input_decoder` (row[2]) | 핵심사실만 | 중요한 것에 집중하도록 유도 |
| **null** | `get_null_input_decoder` (빈칸) | 아무 내용 없음 | "원문 안 봐도 나오는 뻔한 말"을 대표(대조용 음성) |

실제 프롬프트(한국어 데이터셋 `summarize_rl_ko`):

```
main:     [보고서]\n{원문}\n\n위 보고서를 군사 표준용어를 사용하여 간결히 요약하시오.\n\n[요약]\n
presumm:  [보고서]\n{핵심사실}\n\n위 보고서를 ... 요약하시오.\n\n[요약]\n
null:     [보고서]\n\n위 보고서를 ... 요약하시오.\n\n[요약]\n     ← 보고서 자리가 비어 있음
```

각 브랜치를 LLM에 통과시키면 다음이 나옵니다:
- **logits**: 다음 토큰 점수 벡터 (어휘 크기만큼).
- **마지막 토큰 hidden state**: 그 시점 LLM 내부 표현 벡터 (`hidden_size` 차원).

---

## 4. SAD 헤드(MLP): 조합 가중치를 만드는 부분

코드: `modeling_exaone_sad.py:_weight_from_hidden` (매 디코딩 스텝 호출).

세 브랜치의 마지막 hidden state $h_m, h_p, h_n \in \mathbb{R}^{d}$ (여기서 $d=$ `hidden_size`)를 이어붙여 MLP에 넣습니다:

$$
x_0 = [\,h_m \,\Vert\, h_p \,\Vert\, h_n\,] \in \mathbb{R}^{3d}
$$

$$
x_1 = \mathrm{ReLU}\!\Big(\tfrac{1}{s_1}\,W_1 x_0 + b_1\Big),\qquad
x_2 = \mathrm{ReLU}\!\Big(\tfrac{1}{s_2}\,W_2 x_1 + b_2\Big),\qquad
\text{weight} = W_3 x_2 + b_3 \in \mathbb{R}^{3}
$$

- $W_1$=`my_all_f` $(3d \to a)$, $W_2$=`my_all_f1` $(a \to a)$, $W_3$=`my_f` $(a \to 3)$. 여기 $a$=`--alpha_et_hidden_size`.
- $s_1, s_2$: `sqrt_dimension` 옵션이 켜지면 $\sqrt{\text{차원}}$ 으로 나누는 스케일링(값 폭주 방지). 기본 on.
- 출력 `weight`는 3개 스칼라 $[\,w_\alpha,\,w_\beta,\,w_\gamma\,]$.

이 3개에서 조합 계수를 만듭니다 (`_sad_generate` 내부):

$$
(\alpha,\ \beta) = \mathrm{softmax}(w_\alpha,\ w_\beta)\quad(\alpha+\beta=1,\ \alpha,\beta>0),\qquad
\gamma = \sigma(w_\gamma)\in(0,1)
$$

- $\alpha$: main(원문) 비중, $\beta$: presumm(핵심) 비중 — 둘이 합쳐 1.
- $\gamma$: null(뻔한 말)을 **얼마나 빼낼지**의 세기.

> **학습되는 건 정확히 이 MLP의 $W_1,b_1,W_2,b_2,W_3,b_3$ 뿐입니다.** $(\alpha,\beta,\gamma)$는 이들의 출력이고,
> 토큰마다 값이 달라집니다(동적).

---

## 5. 조합 공식 (contrastive combination)

코드: `modeling_exaone_sad.py:381`.

$$
\boxed{\ \text{combined} = (1+\gamma)\,\big(\alpha\cdot \text{logits}_{\text{main}} + \beta\cdot \text{logits}_{\text{presumm}}\big) - \gamma\cdot \text{logits}_{\text{null}}\ }
$$

의미: 원문·핵심을 가중합한 뒤, "빈입력에서도 나오는 뻔한 분포"를 $\gamma$만큼 빼서 **원문에 실제로 근거한 토큰을 부각**시킵니다(Contrastive Decoding, Li et al. 2022 계열).

### 예시 (숫자)
어휘가 `{"5", "3", "보유", "�"}` 4개뿐이라 하고, 어떤 스텝에서 각 브랜치 logits와 MLP 출력이 이렇다고 합시다:

| 토큰 | main | presumm | null |
|---|---|---|---|
| `5` | 3.0 | 2.0 | 0.5 |
| `3` | 1.0 | 0.5 | 0.4 |
| `보유` | 2.0 | 2.5 | 2.0 |
| `�` | 0.2 | 0.1 | 0.1 |

MLP이 $w=[1.0, 0.0, w_\gamma]$ 를 내서 $\alpha=\mathrm{softmax}(1,0)_0=0.73,\ \beta=0.27$, $\gamma=\sigma(w_\gamma)=0.5$ 라면:

`5` 토큰: $(1+0.5)\,(0.73\cdot3.0 + 0.27\cdot2.0) - 0.5\cdot0.5 = 1.5\cdot(2.19+0.54) - 0.25 = 1.5\cdot2.73 - 0.25 = 3.845$

`보유` 토큰: $1.5\,(0.73\cdot2.0 + 0.27\cdot2.5) - 0.5\cdot2.0 = 1.5\cdot(1.46+0.675) - 1.0 = 1.5\cdot2.135 - 1.0 = 2.20$

→ null에서 흔한 `보유`는 $-\gamma\cdot\text{null}$ 때문에 상대적으로 눌리고, 원문 근거가 강한 `5`가 더 부각됩니다.

---

## 6. Plausibility floor (깨진 바이트 컷) — 본 구현의 추가 장치

코드: `modeling_exaone_sad.py:385–396`. 명령줄 옵션 `--plausibility_alpha`(=$\alpha_{\text{pl}}$, 기본 0=끔).

**문제**: 위의 $-\gamma\cdot\text{null}$ 대비가 EXAONE/Llama/Qwen의 **바이트 단위 BPE** 어휘에서 가끔 UTF-8로 깨진 조각의 점수를 튀게 해 숫자가 `�`로 깨집니다.

**해결**: main 브랜치(원문 본 모델)가 "말도 안 된다"고 보는 토큰을 후보에서 제거:

$$
p_{\text{main}}(t)=\mathrm{softmax}(\text{logits}_{\text{main}})[t],\qquad
p_{\text{main}}(t) < \alpha_{\text{pl}}\cdot \max_{t'} p_{\text{main}}(t') \ \Rightarrow\ \text{combined}[t] \leftarrow -\infty
$$

(코드는 로그공간에서 `base_logp < max(base_logp) + log(α_pl)` 로 동일 계산.)

- $\alpha_{\text{pl}}=0.1$: "1등 확률의 10% 이상"인 토큰만 남김 → 깨진 바이트만 컷, 대비 효과 유지. **권장.**
- $\alpha_{\text{pl}}=1$: 1등 토큰만 남김 → 대비 무시, **사실상 main greedy** (방법을 끄는 셈).
- $\alpha_{\text{pl}}=0$(기본): 아무것도 안 함(원래 동작).

> 위 5절 예시에서 $\max p_{\text{main}}$의 10% 미만인 `�`(main logit 0.2)은 제거되고, `5/3/보유`만 combined 점수로 경쟁합니다.

이것은 **원래 SARA에 없던, 본 구현이 추가한 추론 시점 안전장치**이며 **학습되지 않습니다**(gradient 없음).

---

## 7. 강화학습: Self-Critical Sequence Training (SCST)

정답 요약과 토큰을 맞추는 지도학습이 아니라, **생성한 요약의 "질(=보상)"을 직접 올리도록** MLP을 학습합니다.
정답 요약이 없어도 되는 **reference-free** 보상을 씁니다.

### 7.1 보상 $R(\hat y)$ — 한 요약에 대한 점수
코드: `test_performance_decoder_new_fc.py:get_self_critical_reward` (232–234행).

$$
R(\hat y) = w_L\,\text{RougeL} + w_1\,\text{Rouge1} + w_2\,\text{Rouge2} + w_{fk}\,\text{FactKB} + w_{\text{trip}}\,\text{TripCov} + w_{\text{judge}}\,\text{Judge}
$$

각 가중치는 명령줄 옵션(`--RougeL_reward_weight` 등). 본 실험 기본값: $w_L=1,\ w_{\text{trip}}=1,\ w_{\text{judge}}=0.3,\ w_{fk}=0$.

### 7.2 Self-critical baseline (핵심 아이디어)
매 스텝 두 종류를 생성합니다:
- **greedy** 요약 $\hat y^{g}$ (baseline, 안정적인 기준선)
- **sample** 요약 $\hat y^{s}$ (탐색용, `--num_return_sequences`개)

**어드밴티지(advantage)** = 샘플이 baseline보다 얼마나 더 좋았나:

$$
A(\hat y^{s}) = R(\hat y^{s}) - R(\hat y^{g})
$$

코드 236행: `scores = scores[:samples].reshape(B, k) - scores[-B:][:,None]` (샘플별 보상 − 그 입력의 greedy 보상).

- $A>0$: 샘플이 baseline보다 좋음 → 그 토큰들이 더 자주 나오게 밀어줌.
- $A<0$: 더 나쁨 → 억제.
- 이렇게 하면 "평균보다 잘한 만큼만" 학습해 분산이 줄고 안정적입니다.

### 7.3 정책 경사 손실 (loss)
코드: `RewardCriterion` (252–271행).

$$
\mathcal{L} = -\,\frac{\sum_{s}\sum_{t} \log \pi_\theta(\hat y^{s}_t)\;\cdot\;A(\hat y^{s})\;\cdot\;\text{mask}_t}{\sum_{s}\sum_t \text{mask}_t}
$$

- $\log\pi_\theta(\hat y^{s}_t)$: 우리 조합분포(combined에서 나온)가 그 토큰에 부여한 로그확률(`sample_logprobs.gather(...)`).
- `mask`: 패딩/EOS 이후 토큰 제외.
- $\theta$: **MLP 파라미터만.** 백본은 얼어 있어 gradient가 MLP까지만 흐릅니다.

직관: **좋았던(A>0) 샘플에서 실제로 고른 토큰들의 확률을 높이고, 나빴던 샘플의 토큰 확률을 낮춘다.**

### 7.4 예시
입력 1개에 샘플 2개, greedy 보상 $R^g=0.50$ 이라 하면:
- 샘플A $R=0.62 \Rightarrow A=+0.12$ → A의 토큰 확률 ↑
- 샘플B $R=0.44 \Rightarrow A=-0.06$ → B의 토큰 확률 ↓

---

## 8. 보상 구성요소 상세

### 8.1 ROUGE-1 / 2 / Lsum
정답요약 `row[1]` 과의 표면적 n-그램/최장공통부분수열 F-measure. `Evaluator.calculate_rouge`(torchmetrics). 값 $\in[0,1]$.

### 8.2 triplet_coverage — KB 트리플 커버리지
코드: `reward_extras.py:triplet_coverage`. 각 트리플의 **head·tail 엔티티**의 내용 토큰이 요약에 얼마나 들어갔나(부분 점수).

엔티티 $e$의 내용 토큰 집합 $T(e)$ (2글자 이상·순수숫자 아님), 요약 $S$에 대해:

$$
\text{cov}(e) = \frac{|\{\,t\in T(e): t \in S\ (\text{공백무시 포함})\,\}|}{|T(e)|},\qquad
\text{TripCov} = \frac{1}{|E|}\sum_{e\in E}\text{cov}(e)\ \in[0,1]
$$

엔티티가 없으면 vacuously 1.0. 공백에 둔감(`_nospace`)해서 "블루포스" vs "블루 포스" 차이를 흡수.

**예시**: 트리플 tail `"대전차미사일 5기"` → 내용토큰 `{"대전차미사일"}`("5기"는 숫자 성분이라 제외될 수 있음). 요약에 "대전차미사일" 있으면 cov=1.0, 없으면 0.0.

### 8.3 LLM-as-judge (심사관) — 두 종류
> 학습 중 보상에 쓰는 judge와, 평가용 Gemini judge는 **별개**입니다.

#### (a) 학습 보상용: `BackboneJudge` (로컬 백본이 스스로 채점)
코드: `reward_extras.py:BackboneJudge`. **이미 로드된 얼어있는 백본**에게 채점을 시킵니다(API·네트워크 불필요, `torch.no_grad`라 gradient 무관, presumm/null 없이 단일 브랜치로 생성).

프롬프트(`judge_prompt_subscore`)는 **O/X 체크리스트 방식**:
1. 원문의 "검증 가능한 사실 항목"(무기체계+수량, 병력, 사상자 등)을 한 줄씩 나열.
2. 각 항목이 요약에 **정확히(이름+수량 일치)** 반영됐으면 O, 누락·수량생략·수치오류면 X.
3. 그 외 `비조작(0-5)`, `핵심포함(0-5)` 채점.

출력 파싱(`_parse_subscore`)에서 **정확성 점수는 코드가 계산**(LLM 산술 편차 제거):

$$
\text{정확성} = 5\cdot\frac{h}{t}\quad(h=\text{O 개수},\ t=\text{전체 항목수}),\qquad
\text{Judge} = \frac{\text{정확성} + \text{비조작} + \text{핵심포함}}{15}\ \in[0,1]
$$

**예시**: 원문에 검증항목 12개, 요약이 9개 정확 반영(O) → 정확성 $=5\cdot9/12=3.75$. 비조작 4, 핵심포함 4 →
$\text{Judge}=(3.75+4+4)/15=0.783$.

> "대전차미사일 5기"를 요약이 "대전차미사일 보유"로만 적어 **수량을 빠뜨리면 X** → 정확성 하락. 이게 이 프롬프트의 핵심 의도.

#### (b) 평가/모니터링용: Gemini judge
`compare_gemini.py`(테스트셋 전체 채점)와 학습 중 val 로깅에 쓰는 **외부 Gemini API** 심사관. 항목별(정확성/누락/간결성) 1~5점, 정확성은 O/X 적중률로 코드 계산($1+4\cdot h/t$), seed 고정으로 재현성 확보. (자세히는 12절.)

### 8.4 FactKB
영어 전용 사실성 분류기(RoBERTa). **한국어에선 0**이 나오므로 본 실험은 $w_{fk}=0$ 로 끕니다.

### 8.5 ungrounded_fact_penalty (제공되나 기본 미사용)
요약이 만들어낸(원문에 없는) 부대·수치의 비율. 현재 보상 합에는 직접 안 들어가고, judge의 "비조작" 축이 유사 역할.

---

## 9. 학습 루프 한 스텝 (조립)

코드: `test_performance_decoder_new_fc.py:900–1071`. 배치 하나에서:

1. 배치의 원문/핵심/빈입력을 토크나이즈.
2. `model.generate(..., baseline_generation_config)` 로 **greedy** 요약 생성 → baseline.
3. `model.generate(..., sample_generation_config, num_return_sequences=k)` 로 **sample** 요약 생성 + 토큰 로그확률.
4. `get_self_critical_reward(...)` 로 보상 계산 → 어드밴티지 $A$.
5. `RewardCriterion` 로 손실 $\mathcal{L}$ → `loss.backward()` (MLP만 grad).
6. `clip_grad_norm_(max_grad_norm)` → `optimizer.step()`(accumulation_steps마다) → `scheduler.step()`.
7. `iteration += 1`, TensorBoard에 `train/loss`, `reward/*`, `reward_ema/*` 기록.

---

## 10. 무엇이 저장되나 & 체크포인트 선택

- **저장되는 가중치는 MLP 3개뿐**: `save_checkpoint` 가 `model-<tag>_fc_layers.pth` 에 `my_all_f/my_all_f1/my_f` state_dict만 저장(백본은 원본을 그대로 다시 로드하면 됨).
- **best 선택 기준** (`test_performance_decoder_new_fc.py:1107`):

$$
\text{added\_results} = \text{rouge1} + \text{rouge2} + \text{rougeLsum} + w^{\text{test}}_{fk}\cdot\text{factkb}
$$

`--save_checkpoint_every` 마다 테스트셋으로 greedy 디코딩→ROUGE 측정→이 값이 최고면 `model-best_fc_layers.pth` 저장.

---

## 11. 정확히 N번만 학습: `--max_train_iters`

코드: `test_performance_decoder_new_fc.py`(추가됨). `accumulation_steps=1`이면 **iteration 1증가 = 옵티마이저 업데이트 1번**.

$$
\text{train\_step\_cap} = \min(\text{warmup\_train\_step},\ \text{max\_train\_iters})
$$

iteration이 이 값에 도달하면 **배치 루프와 epoch 루프를 모두** 빠져나와 정확히 그 횟수만 학습합니다.
(원래 코드는 배치 루프만 break해서 epoch마다 1스텝씩 초과되는 문제가 있었음 → 그래서 122 같은 값이 나옴.)
`--max_train_iters 120` → 딱 120번.

---

## 12. 추론 · 평가

### 12.1 단일 예시 추론 — `single_infer.py`
`--split test --index i` 로 한 예시를 뽑아 원문/정답/예측 출력. `--input_mode`(document / document+triplets / triplets), `--plausibility_alpha`, `--context_aware_decoding_alpha` 등 지원. `generate_summary(..., pure=True)` 는 presumm/null 없이 **순수 단일 브랜치**(=SAD 끔) 디코딩.

### 12.2 방법 vs 순수 LLM 비교 — `compare_gemini.py`
테스트셋 전체에서 **MINE(LLM+MLP)** 와 **BASE(순수 LLM)** 요약을 만들고:
- **로컬 gold-ROUGE**: 정답요약과의 ROUGE(예시별+평균).
- **Gemini judge(항목별 1~5)**: 정확성/누락/간결성.
  - 정확성 = O/X 적중률로 코드 계산: $\text{acc}=1+4\cdot h/t$ (항목 없으면 5).
  - 승자 = 정확성 가중 평균 $\text{wavg}=\dfrac{w_{\text{acc}}\text{acc}+\text{cov}+\text{brev}}{w_{\text{acc}}+2}$ 비교 ($w_{\text{acc}}$=`--accuracy_weight`).
- **재시도**: 503 등 일시적 오류는 지수 백오프로 끈질기게 재시도(스킵 방지).
- `--out result.json` 로 **모든 스코어**를 `{summary, results:[...]}` 하나에 저장.

**예시 승자 판정**: MINE(acc 4.2, cov 3.8, brev 4.0), BASE(acc 3.0, cov 3.6, brev 4.1), $w_{\text{acc}}=2$:
- MINE wavg $=(2\cdot4.2+3.8+4.0)/4=4.05$, BASE $=(2\cdot3.0+3.6+4.1)/4=3.43$ → **MINE 승**.

### 12.3 학습 중 Gemini 승률 로깅
`--gemini_val_n / --gemini_val_every` 로 `save_checkpoint_every` 시점마다 테스트셋 일부(또는 전체 30개)를 Gemini로 채점해 `val/gemini_winrate`, 항목별 점수, "MINE이 진 인덱스"를 TensorBoard에 남김.

---

## 13. 다중 백본 (Llama / Qwen) — model-agnostic

SAD 헤드는 백본에 독립적입니다:
- `modeling_exaone_sad.py:load_exaone_sad` → `AutoModelForCausalLM.from_pretrained(...)` + `add_sad_head`. 이름과 달리 **아무 causal LM**에 동작.
- `add_sad_head` 는 로드된 클래스를 `<원클래스>_SAD` 로 리블레싱하고 `config.hidden_size`, `get_decoder()`, `get_output_embeddings()`, `o_proj/out_proj` 같은 **표준 인터페이스만** 사용.

**유일하게 필요했던 수정**(`utils.py:_sad_backbone`): `configure_model_loading` 이 `exaone/llama/qwen` 이름을 **SAD 로더로 라우팅**하도록. (이전엔 exaone만 라우팅돼 Llama/Qwen은 SAD 헤드 없는 순수 모델로 로드 → 방법이 안 돌았음.)

### 13.1 oproj 워밍업과 hidden_size
`--fc_init oproj` 는 `my_all_f1`$(a\times a)$ 을 백본 마지막 블록의 어텐션 출력투영 $o\_proj$$(d\times d)$ 으로 초기화합니다. 조건은 **$a=d$(=hidden_size) 그리고 $o\_proj$가 정사각($d\times d$)**. 그래서 스크립트는 모델별로 `--alpha_et_hidden_size = hidden_size` 로 맞춥니다.

| 모델 | hidden_size($d$) | oproj 정사각? |
|---|---|---|
| EXAONE-3.5-7.8B-Instruct | 4096 | ✅ |
| Llama-3.1-8B-Instruct | 4096 | ✅ |
| Qwen3-8B | 4096 | ✅ |
| Qwen2.5-7B-Instruct | 3584 | ✅ |
| Llama-3.2-3B-Instruct | 3072 | ✅ |

(맞지 않으면 워밍업은 조용히 랜덤 초기화로 폴백 — 치명적이지 않음.)

---

## 14. Ablation study (구성요소 제거 실험)

"각 요소가 정말 기여하는가"를 보려면 하나씩 끄고 **같은 조건**으로 재학습·평가합니다.
스크립트: `run_ablation.sh`(단일 백본), `run_ablation_multimodel.sh`(모델×ablation 매트릭스).

| ablation | 바꾸는 것 | 검증 대상 |
|---|---|---|
| **full** | — | 방법 전체(기준) |
| **nojudge** | `--judge_weight 0` | LLM judge 보상의 기여 |
| **notriplet** | `--triplet_coverage_weight 0` | triplet 커버리지 보상의 기여 |
| **rougeonly** | judge·triplet 둘 다 0 | 순수 SCST 대비 총 이득 |
| **norouge** | `--RougeL_reward_weight 0` | ROUGE의 필요성 |
| **fcnone** | `--fc_init none` | oproj 워밍업의 효과 |

> 공정성: 모든 셀을 `--max_train_iters` 동일, greedy 평가·Gemini seed 고정으로 맞춰, 차이가 "학습량"이 아니라 "그 구성요소" 때문이 되게 함.

⚠️ 참고: `--ablation_main_sequence / _presumm_sequence / _null_sequence` 플래그는 현재 GenerationConfig에 값만 실리고 `modeling_exaone_sad.py`가 읽지 않아 **동작하지 않습니다(no-op)**. 브랜치 자체를 끄는 ablation이 필요하면 별도 코드 수정이 필요.

---

## 15. 주요 하이퍼파라미터 / 플래그

| 플래그 | 의미 | 본 실험 값 |
|---|---|---|
| `--model_name_or_path` | 백본 | EXAONE / Llama-3.1-8B / Qwen3-8B |
| `--context_aware_decoding_alpha` | ≥0이면 presumm/null 브랜치 구성(3-branch on) | 1.0 |
| `--alpha_et_hidden_size` | MLP 내부 폭 $a$ (=hidden이면 oproj 워밍업 동작) | 4096 |
| `--fc_init` | MLP 초기화 (none/oproj) | oproj |
| `--RougeL/ triplet_coverage / judge / factkb _weight` | 보상 가중치 | 1 / 1 / 0.3 / 0 |
| `--judge_max_new_tokens` | 로컬 judge 생성 길이(O/X 체크리스트라 길게) | 384 |
| `--plausibility_alpha` | 깨진 바이트 컷 문턱 (추론) | 0(학습)/0.1(평가) |
| `--num_return_sequences` | SCST 샘플 수 | 2 |
| `--max_train_iters` | 본 학습 스텝 상한 | 120 |
| `--save_checkpoint_every` | val/체크포인트 주기 | 10 |
| `--device` | 단일 GPU 지정 | cuda:0 |

---

## 16. 파일별 역할 지도

| 파일 | 역할 |
|---|---|
| `modeling_exaone_sad.py` | SAD 헤드 정의, 3-branch forward, 조합공식, plausibility floor, 모델-불문 로더 |
| `test_performance_decoder_new_fc.py` | 학습 메인(SCST 루프), 보상 계산, best 선택, val/Gemini 로깅, `--max_train_iters` |
| `reward_extras.py` | triplet_coverage, ungrounded penalty, BackboneJudge/GeminiJudge, judge 프롬프트·파싱 |
| `utils.py` | 데이터 로딩, 브랜치 입력 템플릿, `configure_model_loading`/`_sad_backbone`, Evaluator |
| `single_infer.py` | 단일 예시 추론(FC 가중치만 로드) |
| `compare_gemini.py` | 테스트셋 전체 MINE vs BASE 비교(ROUGE+Gemini 항목별), JSON 저장, 503 재시도 |
| `run_ablation.sh` / `run_ablation_multimodel.sh` | ablation(+다중 백본) 자동 학습·평가·요약 |

---

## 부록: 자주 헷갈리는 두 "알파"

| | $\alpha$ (조합계수) | $\alpha_{\text{pl}}$ (`--plausibility_alpha`) |
|---|---|---|
| 정체 | main 브랜치 가중치 | 깨진 바이트 컷 문턱 |
| 출처 | MLP이 매 토큰 출력 | 사람이 명령줄 상수 |
| 학습되나 | **예(유일한 학습 대상 중 일부)** | 아니오(gradient 없음) |
| 원래 SARA에 | 있음(핵심) | 없음(본 구현 추가) |
