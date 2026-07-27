# PMI 결합 가중치만 학습하는 참조 없는(Reference-Free) 군사 문서 요약

**― 동결된 LLM 위에서 4-브랜치 PMI 디코딩의 결합 가중치 `[a, b, c, d]`만 강화학습으로 학습한다**

---

## 초록 (Abstract)

우리는 정답 요약(gold summary) 없이, **동결된(frozen) 대형 언어모델(LLM)** 을 그대로 두고
그 위에 얹은 **작은 가중치 정책망(weight policy, ~수백만 파라미터의 MLP)** 만 강화학습으로
학습하여 군사 도메인 요약을 생성하는 방법을 제시한다. 매 디코딩 스텝에서 LLM은 네 개의
서로 다른 조건(원문, 핵심 triplet, 표준용어, 순수 지시문)으로 **네 벌의 다음-토큰 로짓**을
내놓고, 정책망은 이 네 벌을 섞는 **점별 상호정보(pointwise mutual information, PMI) 결합
가중치** `a, b, c, d`를 토큰마다 산출한다. 학습 신호는 **참조 없는 보상(reference-free
reward)** — 원문 충실도, triplet 커버리지, 표준용어 사용, 핵심문장 반영, PMI 대비(contrast),
길이·복사 페널티 — 로만 구성된다. 두 가지 강화학습 알고리즘(자기비판 시퀀스 학습 SCST,
그룹 상대 정책 최적화 GRPO)을 동일한 디코딩·보상 위에서 교체 가능하게 구현했다. 본 문서는
현재 코드베이스에 실제로 반영된 수식과 알고리즘을 빠짐없이, 그러나 직관적으로 서술한다.

---

## 1. 서론과 문제 설정

일반적인 요약 강화학습은 LLM 전체를 미세조정하며, 흔히 정답 요약이나 보상모델을 요구한다.
본 방법의 핵심 제약과 선택은 다음과 같다.

- **LLM은 동결.** 역전파가 LLM에 닿지 않는다(로짓은 디코더 내부에서 `detach`된다). 따라서
  학습 대상은 오직 정책망 파라미터 $\theta$ 뿐이다. 이는 (i) 학습 비용이 매우 작고,
  (ii) 백본 교체가 자유로우며, (iii) 도메인 안전성(백본 지식 훼손 없음)을 준다.
- **정답 요약 불필요.** 보상은 원문·구조화 정보(triplet)·용어사전으로부터 규칙적으로
  계산되는 참조 없는 신호다.
- **학습되는 것은 "무엇을 얼마나 섞을지"뿐.** 정책망은 네 조건부 분포의 **결합 비율**만
  결정한다. 새로운 내용을 창작하지 않으며, 백본이 각 조건에서 낼 수 있는 분포를 재조합할 뿐이다.

기호 요약: 어휘집합 크기 $V$, LLM 은닉차원 $D$, 디코딩 스텝 $t$, 생성 토큰 $o_{t}$.

---

## 2. 전체 구조 (Overview)

$$
\underbrace{\text{4-브랜치 프롬프트}}_{\text{XQ, SQ, GQ, Q}}
\;\xrightarrow{\text{동결 LLM}}\;
\underbrace{\{\ell_{XQ}, \ell_{SQ}, \ell_{GQ}, \ell_{Q}\}}_{\text{네 벌의 로짓 }[4,V]},\;
\underbrace{\{h_{XQ}, h_{SQ}, h_{GQ}, h_{Q}\}}_{\text{네 벌의 은닉상태 }[4,D]}
$$

$$
\{h_\bullet\}\;\xrightarrow{\text{정책망 }\pi_\theta}\;(a,b,c,d)
\;\xrightarrow{\text{PMI 결합}}\;\ell_c \;\xrightarrow{\text{샘플링}}\; o_t
$$

- 정책망은 **은닉상태**로 가중치를 만들고, 그 가중치로 **로짓**을 결합한다.
- 한 요약(rollout)을 끝까지 생성한 뒤 **참조 없는 보상** $R$ 을 매기고, 그 $R$ 로 정책망을
  갱신한다.

---

## 3. 네 브랜치 구성 (`branches.py`)

하나의 입력 예시는 원문 $X$, triplet 목록 $S$, 질의(지시문) $Q_{\text{text}}$ 로 이루어진다.
여기서 네 개의 **조건 텍스트**를 만든다.

| 브랜치 | 조건 내용 | 프롬프트 구성 |
|---|---|---|
| **XQ** | 원문 + 지시 | `[원문]{X}\n\n[지시]{Q}` |
| **SQ** | 핵심 triplet + 지시 | `[핵심정보]{S}\n\n[지시]{Q}` |
| **GQ** | 게이팅된 표준용어 + 지시 | `[표준용어]{G}\n\n[지시]{Q}` |
| **Q** | 지시만 (사전분포/prior) | `[지시]{Q}` |

**Triplet 직렬화 $S$** (`serialize_triplets`): 같은 head를 묶어
`head: relation tail; relation tail` 형태로 만든다.

**용어사전 게이팅 $G$** (`glossary.py`): 표준용어사전은 `{표준용어: [트리거들]}` 구조다.
원문에 트리거가 실제로 등장한 표준용어만 **활성(active)** 으로 표시하여 $G$ 에 넣는다.

$$
G = \{\, \tau \;:\; \exists\, g \in \text{triggers}(\tau),\; g \subseteq X \,\}
$$

이 게이팅은 **근거 없는 용어 주입(환각)** 을 막는다: 원문에 근거가 없는 표준용어는 GQ 브랜치에
아예 들어가지 못한다. (표준용어 자신도 자기 트리거의 하나로 취급된다.)

---

## 4. PMI 결합 디코딩 (`decoder.py`)

### 4.1 결합 공식

각 스텝에서 네 브랜치 로짓 $\ell_{XQ}, \ell_{SQ}, \ell_{GQ}, \ell_{Q} \in \mathbb{R}^{V}$ (상수로
취급, `detach`)을 정책 가중치로 결합한다:

$$
\boxed{\;\ell_c \;=\; (1+a)\,\big(b\,\ell_{XQ} + c\,\ell_{SQ} + d\,\ell_{GQ}\big) \;-\; a\,\ell_{Q}\;}
$$

여기서
$$
a \in (0,1), \qquad b + c + d = 1,\quad b,c,d \ge 0 .
$$

**직관.** 괄호 안 $b\ell_{XQ}+c\ell_{SQ}+d\ell_{GQ}$ 은 "원문·핵심정보·표준용어"라는 **근거 있는
분포**의 볼록결합이다. 여기서 $a\,\ell_Q$ (지시만 본 **사전분포/prior**)를 빼면, 근거와 무관하게
그냥 그럴듯한 토큰(백본의 일반 성향)을 **억제**한다. 이것이 PMI(점별 상호정보)의 형태다:
"근거가 주어졌을 때 이 토큰이 얼마나 더 그럴듯해지는가"를 키우는 방향이다. 앞의 $(1+a)$ 배율은
prior를 빼면서 줄어든 스케일을 보정한다.

- $a$: **prior 제거 강도.** 클수록 "일반적으로 그럴듯한" 토큰을 강하게 눌러 원문 특유의 내용을
  부각한다.
- $b, c, d$: **원문 / 핵심정보 / 표준용어** 사이의 균형.

배치 버전 `combine_logits_batch` 는 로짓이 $[N,4,V]$ 일 때 동일 연산을 롤아웃 축에 대해
벡터화한 것이다.

### 4.2 샘플링과 로그확률

결합 로짓을 온도 $T$ 로 나눈 뒤:
$$
\tilde\ell = \ell_c / \max(T,\,10^{-6})
$$
- **최소 길이 강제:** $t < \text{min\_new\_tokens}$ 이면 EOS 로짓을 $-\infty$ 로 마스킹.
- **샘플링:** nucleus(top-$p$) 마스크 후 재정규화하여 다항분포에서 $o_t$ 를 뽑는다(학습 시).
  추론 시에는 greedy(argmax).
- **로그확률:** $\log \pi_\theta(o_t) = \big[\log \mathrm{softmax}(\tilde\ell_{\text{filtered}})\big]_{o_t}$.
  이 값만이 그래디언트를 나른다: $\log\pi \to \ell_c \to (a,b,c,d) \to \theta$.

### 4.3 PMI 대비 항 (contrast) — 보상 전용, 그래디언트 없음

$a$ 를 학습시키려면 "prior를 얼마나 뺐을 때 좋은가"라는 신호가 필요하다. 이를 위해 정책과
**무관한** 대비 스칼라를 토큰마다 계산한다:

$$
\mathrm{pmi}(o_t) \;=\; \tanh\!\Big(\big[\log \mathrm{softmax}(\bar\ell_{+})\big]_{o_t}
\;-\; \big[\log \mathrm{softmax}(\ell_Q)\big]_{o_t}\Big),
\qquad
\bar\ell_{+} = \tfrac{1}{3}(\ell_{XQ}+\ell_{SQ}+\ell_{GQ})
$$

- 근거 분포(세 브랜치 균등 평균)에서 이 토큰이 prior $Q$ 보다 얼마나 더 그럴듯한지를 잰다.
  양수면 "원문 근거 덕분에 뽑힌 토큰".
- **$\tanh$ 로 $(-1,1)$ 로 유계화**한 것이 핵심이다. 원래 로그비는 무한대로 커질 수 있어서,
  이를 그대로 보상에 쓰면 정책이 $a$ 를 경계(0 또는 1)로 몰아 **보상만 부풀리는 붕괴(reward
  hacking)** 가 일어났다. $\tanh$ 가 이를 막는다.
- 한 롤아웃의 대비 값은 토큰 평균 $\overline{\mathrm{pmi}} = \frac{1}{L}\sum_t \mathrm{pmi}(o_t)$
  으로 보상에 들어간다(그래디언트 없음, 순수 스칼라).

---

## 5. 가중치 정책망 (`policy.py`)

### 5.1 입력 특징

스텝 $t$ 의 네 브랜치 은닉상태로 특징 벡터를 만든다:

$$
x_t = \big[\,h_{XQ};\,h_{SQ};\,h_{GQ};\,h_{Q}\,\big]
\;\big(\;+\;[\,h_{XQ}-h_Q;\;h_{SQ}-h_Q;\;h_{GQ}-h_Q\,]\;\big)
$$

괄호 안의 **대비 특징(contrast features)** 은 옵션(`use_contrast_features=True`)이며, "근거 대비
prior의 차이"를 정책에 직접 제공한다. 입력 차원은 $4D$ (대비 포함 시 $7D$).

### 5.2 신경망과 출력 제약

$$
u = \mathrm{Head}\big(\mathrm{MLP}(\mathrm{LayerNorm}(x_t))\big) \in \mathbb{R}^4,
\qquad u = (u_a, u_b, u_c, u_d)
$$
MLP: `Linear→GELU→Dropout→Linear→GELU→Dropout`, Head: `Linear(h,4)`.

출력 제약:
$$
a = \sigma(u_a) \in (0,1), \qquad (b,c,d) = \mathrm{softmax}(u_b, u_c, u_d).
$$

- `learn_a=True`(기본): $a$ 를 토큰마다 학습.
- `learn_a=False`: $a \equiv a_{\text{fixed}}$ 상수로 고정(그래디언트 없음), 행동공간을
  $(b,c,d)$ 심플렉스로 축소.

**웜스타트 초기화.** Head 가중치 $\sim \mathcal{N}(0, \text{init\_std}^2)$, 편향 $=0$. 그러면
초기 출력 $u \approx 0$ 이므로 $a = \sigma(0) = 0.5$, $b=c=d=\mathrm{softmax}(0,0,0)=1/3$ 인
**중립 시작점**이 된다.

### 5.3 엔트로피(탐색/붕괴 방지)

$$
H(a,b,c,d) \;=\; \underbrace{-\!\!\sum_{k\in\{b,c,d\}} k\log k}_{\text{범주형 }(b,c,d)}
\;+\; \underbrace{-\big(a\log a + (1-a)\log(1-a)\big)}_{\text{베르누이 }a}
$$

핵심은 **$a$ 까지 엔트로피에 포함**한 것이다. 범주형 엔트로피만 쓰면 $b,c,d$ 붕괴는 막아도
$a$ 가 경계로 포화되는 것은 못 막는다. $H$ 는 중립 시작점 $(a{=}0.5, b{=}c{=}d{=}1/3)$ 에서
최대가 된다.

---

## 6. 참조 없는 보상 (`rewards.py`)

한 롤아웃(요약 텍스트 $y$)의 보상은 다음 항들의 가중합이다. 각 항은 해석 가능한 유계 범위를
가진다.

### 6.1 개별 항목

**(a) 충실도 (Faithfulness), $F \in [0,1]$.** 요약의 **내용 토큰**(2자 이상) 중 원문에
부분문자열로 근거되는 비율. 1자 토큰(한국어 조사·조각)은 아무 문서에나 사소하게 걸리므로
제외한다. (NLI/FactKB 모델을 `FaithfulnessModel` 인터페이스로 교체 가능.)

$$
F = \frac{\#\{\text{원문에 등장하는 요약 내용토큰}\}}{\#\{\text{요약 내용토큰}\}}
$$

**(b) Triplet 커버리지 (Coverage), $\mathrm{Cov}\in[0,1]$.** head/tail 엔티티가 요약에 얼마나
반영됐는지. **최근 수정된 정의**는 다음과 같다(구식 전부-일치 방식이 실데이터에서 항상 0에
가까워 학습 신호를 죽였기 때문):

$$
\mathrm{Cov} = \frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}}
\frac{\#\{\text{요약에 등장하는 } e \text{의 내용토큰}\}}{\#\{e \text{의 내용토큰}\}}
$$

- $\mathcal{E}$: 중복 제거한 head/tail 엔티티 집합.
- 엔티티별 **부분 점수**(토큰 비율)를 매긴 뒤 평균 — 전부-일치(all-or-nothing)가 아니다.
- **공백 무시 매칭**: 사전형 `블루포스` ↔ 요약형 `블루 포스` 를 같게 본다.
- 1자·순수숫자 토큰은 제외, 점수화 불가한 엔티티는 건너뜀.

> **왜 바꿨나 (설계 노트).** 실데이터의 tail은 `영토 방어 및 지역 안정` 같은 긴 서술구라,
> 전부-일치로는 사람이 쓴 정답 요약조차 $\mathrm{Cov}\approx 0.1\text{–}0.3$ 이었다. 그러면
> 그룹 내 모든 롤아웃에서 $\mathrm{Cov}$ 가 평평 → 상대 이득으로 상쇄 → **학습 신호 0** 이 되어,
> 게임 가능한 다른 항들만 학습을 지배하고 "군사적이지만 엉뚱한" 요약이 최적해가 되었다.
> 부분 점수+공백무시로 바꾸자 정답 요약이 $\sim 0.4\text{–}0.56$ 로 오르고(포화 아님 → 롤아웃
> 간 분산 발생 → **살아있는 신호**), 엉뚱한 요약은 여전히 $\sim 0$ 로 남았다.

**(c) 표준용어 사용 (Term usage), $\mathrm{Term}\in[0,1]$.** 활성 표준용어 중 **원문에 아직 없는**
것(=용어 표준화가 실제로 필요한 것)만 분모로 삼아, 요약에 등장한 비율. 원문에 이미 있는 용어는
그냥 복사해도 얻어지므로 크레딧을 주지 않는다(복사 유인 차단).

**(d) 핵심문장 반영 (Key-sentence coverage), $\mathrm{KS}\in[0,1]$.** 동결 LLM이 **원문에서 직접
고른** 핵심 문장 $\{s_j\}$ 각각에 대해, 그 내용토큰이 요약에 반영된 비율의 평균:

$$
\mathrm{KS} = \frac{1}{|\{s_j\}|}\sum_j
\frac{\#\{\text{요약에 등장하는 } s_j \text{의 내용토큰}\}}{\#\{s_j \text{의 내용토큰}\}}
$$

핵심문장은 원문에만 의존하므로 문서당 **한 번 추출해 캐시**한다(`keysent.py`). 프롬프트는
"가장 중요한 핵심 문장 $n$개를 원문에서 골라 출력하시오". 이 항은 구조화된 엔티티(커버리지)가
놓치는 **원문의 서술적 핵심**을 잡는 리콜 앵커다.

**(e) PMI 대비 (Contrast), $\mathrm{Ctr}\in(-1,1)$.** §4.3의 토큰 평균 $\overline{\mathrm{pmi}}$.
**유일하게 $a$ 에 학습 신호를 주는 항.**

**(f) 길이 페널티, $\mathrm{Lpen}\ge 0$.** 과다 길이 + $n$-그램 반복:
$$
\mathrm{Lpen} = \frac{\max(0,\, L - L^\*)}{L^\*} \;+\; \Big(1 - \frac{\#\{\text{서로 다른 }n\text{-gram}\}}{\#\{n\text{-gram}\}}\Big)
$$
($L$: 요약 토큰 수, $L^\*$: 목표 길이 `target_length`.)

**(g) 복사 페널티 (Extractive copy), $\mathrm{Copy}\in[0,1]$.** 요약 $n$-그램 중 원문과 그대로
겹치는 비율. 원문을 통째로 베끼는 퇴화해(=원문 브랜치로 $b\to 1$ 몰기)를 벌한다.

### 6.2 총 보상 (기본: 가산형)

$$
\boxed{\;R \;=\; w_F\,F + w_C\,\mathrm{Cov} + w_T\,\mathrm{Term} + w_{KS}\,\mathrm{KS}
+ w_{Ct}\,\mathrm{Ctr} \;-\; w_L\,\mathrm{Lpen} \;-\; w_{Cp}\,\mathrm{Copy}\;}
$$

### 6.3 내용 게이트 (옵션: `balance_content=True`)

"유창하지만 엉뚱한" 실패를 직접 막기 위한 **선택적** 변형. 게임 가능한 유창성 항
(충실도·용어·대비)을 **실제로 담은 내용량**으로 스케일한다:

$$
\text{fluency} = w_F F + w_T \mathrm{Term} + w_{Ct}\mathrm{Ctr},\qquad
\text{gate} = \max\!\big(\text{gate\_floor},\; \mathrm{mean}(\mathrm{Cov}, \mathrm{KS})\big)
$$
$$
R = \text{gate}\cdot\text{fluency} \;+\; w_C\,\mathrm{Cov} + w_{KS}\,\mathrm{KS}
\;-\; w_L\,\mathrm{Lpen} - w_{Cp}\,\mathrm{Copy}
$$

내용을 안 담으면($\mathrm{Cov},\mathrm{KS}\to 0$) 유창성 점수가 `gate_floor`(기본 0.1)까지
눌려 학습을 지배하지 못한다. 내용 앵커(Cov, KS)는 가산으로 남아 항상 내용 쪽으로 당긴다.
기본값은 off라 가산형과 완전히 동일하다.

### 6.4 그룹 내 보상 표준화

$$
\tilde R_i = \frac{R_i - \mu}{\sigma + \varepsilon},\qquad
\mu = \frac{1}{N}\sum_i R_i,\quad \sigma = \sqrt{\tfrac{1}{N}\sum_i (R_i-\mu)^2}
$$
(SCST에서 사용; GRPO는 이 형태를 이득으로 직접 쓴다 — §8.)

---

## 7. 학습 알고리즘 A — 자기비판 시퀀스 학습 (SCST, `train.py`)

한 입력에 대해:

1. **$N$개 확률적 롤아웃** 생성(배치 forward, 배치 크기 $N\times4$).
2. 롤아웃별 참조 없는 보상 $R_i$, 그룹 표준화 $\tilde R_i$ (§6.4).
3. **자기비판 베이스라인** $b = \frac{1}{N}\sum_i \tilde R_i$ (또는 greedy 롤아웃 보상).
4. **이득** $A_i = \tilde R_i - b$.
5. **손실**:

$$
\boxed{\;
\mathcal{L}_{\text{SCST}} = -\frac{1}{N}\sum_{i=1}^{N} A_i \cdot \frac{L_{\text{ref}}}{L_i}
\cdot \sum_{t=1}^{L_i} \log \pi_\theta(o_{i,t}) \;-\; \beta\, \bar H
\;}
$$

여기서 $L_{\text{ref}} = \frac{1}{N}\sum_i L_i$ (그룹 평균 길이), $\bar H$ 는 평균 엔트로피,
$\beta = \text{entropy\_beta}$.

**길이 정규화의 요점.** 각 롤아웃의 시퀀스 로그확률을 자기 길이로 **나누는** 대신
$(L_{\text{ref}}/L_i)$ 로 **스케일**한다. 긴 롤아웃이 지배하는 것(길이 정규화의 목적)은 막되,
전체 그래디언트 크기를 $\sim L$ 배 줄여버리지 않는다. (자기 길이로 나눴을 때 그래디언트가
약 $L(\approx 24)$배 작아져 `grad_clip` 아래로 떨어지고 정책 가중치가 얼어붙는 문제가 있었다.)

6. **최적화:** AdamW + 그래디언트 클리핑 + 누적. 학습률은 선형 워밍업 후 코사인 감쇠:
$$
\eta(s) = \begin{cases}
\eta_0 \cdot s / s_{\text{warm}} & s < s_{\text{warm}}\\[4pt]
\eta_0 \cdot \tfrac{1}{2}\big(1 + \cos(\pi\, p)\big),\; p=\frac{s - s_{\text{warm}}}{s_{\text{total}} - s_{\text{warm}}} & \text{그 외}
\end{cases}
$$

---

## 8. 학습 알고리즘 B — 그룹 상대 정책 최적화 (GRPO, `grpo.py`)

SCST와 디코딩·보상·정책·PMI 대비까지 전부 동일하고, **RL 목적함수만** 다르다.

1. 한 프롬프트에서 **$G$개 롤아웃**(그룹)을 배치로 생성.
2. 롤아웃별 보상 $R_i$.
3. **그룹 상대 이득**(critic 없음, greedy 베이스라인 없음), 롤아웃 $i$ 의 모든 토큰에 브로드캐스트:
$$
A_i = \frac{R_i - \mu_G}{\sigma_G + \varepsilon},\qquad
\mu_G = \tfrac{1}{G}\sum_i R_i,\;\; \sigma_G = \sqrt{\tfrac{1}{G}\sum_i (R_i - \mu_G)^2}
$$
4. **중요도비와 PPO 클리핑 대리손실.** 같은 샘플 그룹으로 `inner_epochs`번 갱신:
$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t})}{\pi_{\theta_{\text{old}}}(o_{i,t})}
= \exp\!\big(\log\pi_\theta(o_{i,t}) - \log\pi_{\theta_{\text{old}}}(o_{i,t})\big)
$$
$$
L^{\text{clip}}_{i} = \frac{1}{L_i}\sum_{t}
\min\!\Big(\rho_{i,t} A_i,\;\; \mathrm{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\, A_i\Big)
$$
5. **참조 정책에 대한 토큰별 KL** (k3 추정자, 항상 $\ge 0$):
$$
\mathrm{KL}_{i,t} = \exp(r_{i,t}) - r_{i,t} - 1,\qquad
r_{i,t} = \log\pi_{\text{ref}}(o_{i,t}) - \log\pi_\theta(o_{i,t})
$$
참조 정책 $\pi_{\text{ref}}$ 는 **초기(웜스타트) 정책의 동결 복사본**이다.

6. **총 손실** (엔트로피 보너스 포함):
$$
\boxed{\;
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{L_i}\sum_{t}
\Big[\, \min(\rho_{i,t}A_i,\, \mathrm{clip}(\rho_{i,t})A_i) \;-\; \beta_{\text{KL}}\,\mathrm{KL}_{i,t}
\;+\; \beta_{H}\, H_{i,t} \,\Big]
\;}
$$

**정확한 비율을 위한 장치.** GRPO 동안 **드롭아웃을 끈다**(정책을 결정적으로). 그래야 여러 inner
epoch에 걸쳐 $\pi_{\theta_{\text{old}}}$ 가 잘 정의되고 $\rho$ 가 정확하다. 또한 old/new/ref
로그확률이 **같은 분포**를 공유하도록 nucleus 절단 없이(`top_p=1.0`) **전체 softmax**로 재점수한다.
`score_tokens_batch` 가 고정 토큰열을 교사강요(teacher-forcing)로 재점수한다.

---

## 9. SCST vs GRPO 요약 비교

| 항목 | SCST | GRPO |
|---|---|---|
| 베이스라인 | 그룹 평균(또는 greedy) 보상 | 그룹 표준화 이득 $(R-\mu)/(\sigma+\varepsilon)$ |
| 이득 부여 | 시퀀스 단위 | **토큰 단위**(브로드캐스트) |
| 비율/클리핑 | 없음(on-policy REINFORCE) | PPO 중요도비 + 클립 $\epsilon$ |
| 표본 재사용 | 1회 | `inner_epochs`회 |
| 정규화 | 그룹 표준화 + 길이 스케일 | 그룹 표준화 |
| 안정화 | 엔트로피 보너스 | 엔트로피 + **참조 KL(k3)** |
| 드롭아웃 | 켬 | **끔**(비율 정확성) |

두 알고리즘은 `--rl {scst,grpo}` 로 교체된다(체크포인트 payload 형식은 동일).

---

## 10. 추론과 순수 LLM 비교 (`infer.py`, `serve.py`)

- **추론:** 학습된 정책 체크포인트를 로드하고, 네 브랜치를 구성해 **greedy PMI 디코딩** 1회.
  요약 텍스트와 함께 토큰 평균 가중치 $(\bar a, \bar b, \bar c, \bar d)$ 를 반환한다.
- **순수 LLM 베이스라인:** 같은 XQ 프롬프트(원문+지시)를 백본에 **그대로** 넣어 얻은 요약.
  PMI 결합도 정책도 없는, 정책이 개선하려는 기준선이다. 서버는 `/summarize` 응답에 `baseline`
  필드로 함께 반환하여(기본 on) 정책의 효과를 나란히 확인할 수 있다.

---

## 11. 구현 세부

- **배치 롤아웃 생성.** 같은 프롬프트의 $N$개 롤아웃을 **한 번의 배치 forward**(배치 $N\times4$)로
  처리(`generate_batch`). EOS를 친 롤아웃은 pad 토큰을 먹이며 기록을 멈추고, 나머지는 계속 생성.
- **배치 재점수.** `score_tokens_batch` 가 고정 토큰열들을 최장 길이까지 배치 교사강요로
  재점수(전체 softmax, 동일한 min-new-tokens EOS 마스킹).
- **동결 보장.** `assert_llm_frozen` 이 매 스텝 백본 파라미터가 `requires_grad=False` 임을 확인.
  정책망은 수치 안정성을 위해 **fp32** 유지(백본이 bf16이어도).
- **핵심문장 캐시.** 문서당 1회 추출(`KeySentenceExtractor._cache`).

---

## 12. 기본 하이퍼파라미터 (`config.py`)

| 그룹 | 파라미터 | 값 |
|---|---|---|
| 정책 | `hidden_dim` / `dropout` / `init_std` | 512 / 0.1 / 0.01 |
| | `use_contrast_features` / `learn_a` / `fixed_a` | True / True / 0.5 |
| 디코딩 | `temperature` / `top_p` | 1.0 / 0.95 |
| | `max_new_tokens` / `min_new_tokens` | 120 / 20 |
| 보상 가중치 | $w_F, w_C, w_T, w_{KS}, w_{Ct}, w_L, w_{Cp}$ | 1.0, 2.0, 0.25, 1.0, 0.5, 0.2, 1.0 |
| 보상 기타 | `target_length` / `repeat_ngram` / `copy_ngram` | 120 / 3 / 4 |
| | `keysent_n` / `balance_content` / `gate_floor` | 3 / False / 0.1 |
| 학습(SCST) | `num_samples` / `lr` / `weight_decay` | 5 / 3e-4 / 0.01 |
| | `warmup_ratio` / `total_steps` / `grad_clip` | 0.07 / 2000 / 1.0 |
| | `entropy_beta` / `reward_norm` / `baseline` | 0.01 / True / mean |
| GRPO | `group_size` / `clip_eps` / `kl_beta` | 8 / 0.2 / 0.04 |
| | `inner_epochs` / `adv_eps` / `entropy_beta` | 2 / 1e-6 / 0.01 |

---

## 부록 A. 기호표

| 기호 | 의미 |
|---|---|
| $\ell_{XQ},\ell_{SQ},\ell_{GQ},\ell_Q$ | 네 브랜치의 다음-토큰 로짓 ($\mathbb{R}^V$) |
| $h_{XQ},\dots,h_Q$ | 네 브랜치의 마지막층 은닉상태 ($\mathbb{R}^D$) |
| $a,b,c,d$ | PMI 결합 가중치; $a\in(0,1)$, $b{+}c{+}d{=}1$ |
| $\ell_c$ | 결합 로짓 |
| $\pi_\theta$ | 가중치 정책(유일한 학습 대상) |
| $R,\tilde R,A$ | 보상 / 그룹 표준화 보상 / 이득 |
| $F,\mathrm{Cov},\mathrm{Term},\mathrm{KS},\mathrm{Ctr}$ | 충실도/커버리지/용어/핵심문장/대비 |
| $\rho_{i,t}$ | GRPO 중요도비 |
| $\mathrm{KL}_{i,t}$ | 참조 정책 KL(k3 추정자) |
| $L_i, L_{\text{ref}}$ | 롤아웃 길이 / 그룹 평균 길이 |
| $\varepsilon,\epsilon$ | 표준화 안정항 / PPO 클립 폭 |

---

*본 문서는 현재 코드베이스(`summarize_rl/`)에 구현된 수식과 알고리즘을 그대로 서술한 것이며,
구식(전부-일치 커버리지 등)은 최신 정의로 갱신되어 있다.*
