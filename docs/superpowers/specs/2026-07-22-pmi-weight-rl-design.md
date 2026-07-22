# 군사 도메인 PMI 가중치 RL 학습 — 설계 문서 (핵심 알고리즘 라이브러리)

**작성일:** 2026-07-22
**범위:** 작업계획서 Phase 2~4의 핵심 알고리즘을 데이터/백본 선정과 독립적인 자기완결적 Python 패키지로 구현.

---

## 1. 목표와 범위

### 1.1 이번 구현 목표
얼려둔(frozen) LLM 위에서 PMI 결합 가중치 `[a,b,c,d]`만 강화학습으로 학습하는
핵심 알고리즘을 구현한다. 실제 한국어 7~13B 백본, 실제 원문·triplet 코퍼스,
실제 군사 용어사전은 **인터페이스로 분리**하여 나중에 교체 가능하게 한다.

### 1.2 포함
- 4갈래(XQ/SQ/GQ/Q) 프롬프트 구성 및 triplet 직렬화
- 다갈래 PMI 결합식 디코더 (정책망 가중치 주입)
- 가중치 생성 정책망 MLP (제약 + warm-start, fp32)
- 용어사전 게이팅 (표준용어 ← 트리거 매칭)
- 보상 모듈 (Faithfulness / TripletCoverage / TermUsage / LengthPenalty)
- SCST 학습 루프 (N 롤아웃, self-critical baseline, 마스크 손실, AdamW, clip, accum, ckpt)
- LLM 백엔드 래퍼 (frozen, hidden state + logit 추출)

### 1.3 제외 (의존성 때문에 이번 범위 밖)
- 실제 백본 LLM 선정/다운로드
- 실제 원문·triplet 코퍼스, 실제 군사 용어사전 구축
- vLLM 통합 (초기엔 HF 기반 자기회귀 생성 + KV 캐시)
- 대규모 학습 실행, 무거운 NLI 충실도 모델 배포 (인터페이스 + lexical fallback 제공)

---

## 2. 아키텍처

### 2.1 모듈 구성

```
summarize_rl/
  __init__.py
  config.py            # 데이터클래스 설정 (디코딩/학습/보상 하이퍼파라미터)
  llm_backend.py       # LLM 래퍼: frozen, 4갈래 logit + 마지막층 hidden 추출
  branches.py          # 4갈래 프롬프트 구성 + triplet 직렬화
  glossary.py          # 용어사전 게이팅: 표준용어 ← 트리거 매칭
  policy.py            # 정책망 MLP: x_t → [a,b,c,d], sigmoid/softmax 제약
  decoder.py           # 다갈래 PMI 결합식 + 샘플링/greedy 디코딩
  rewards.py           # 보상 4종 + 가중합 + 정규화
  train.py             # SCST 학습 루프
tests/                 # 각 모듈 유닛테스트 + 작은 모델 end-to-end smoke test
```

### 2.2 데이터 흐름

```
입력 예제 (X 원문, triplets, Q 질의)
  │
  ├─ glossary.gate(X)                → 활성 표준용어 G
  ├─ branches.build(X, triplets, G, Q) → 4갈래 텍스트 (XQ, SQ, GQ, Q)
  │
  └─ decoder.generate(...)  ── 매 토큰 t ──▶
         llm_backend.step(4갈래)  → logit_{XQ,SQ,GQ,Q}, hidden_{XQ,SQ,GQ,Q}
         policy(features)         → a_t, b_t, c_t, d_t
         combine:  (1+a)[b·XQ + c·SQ + d·GQ] − a·Q
         sample / greedy          → ŷ_t, logπ_t
       ──▶ 생성 요약 Ŷ, logπ 시퀀스, 마스크
  │
  └─ rewards.compute(Ŷ, X, triplets, active_terms) → R
  │
  └─ train: N 롤아웃 → baseline b → advantage → 마스크 손실 → θ만 갱신
```

### 2.3 핵심 인터페이스 (경계)

- **LLMBackend** — `step_logits_hidden(branch_ids, past_kv) -> (logits[V], last_hidden[D], new_kv)`
  실제 백본을 감싸는 유일한 지점. 테스트는 이 인터페이스를 mock으로 대체.
- **Policy** — `forward(features[B, F]) -> (a, b, c, d)`. 순수 torch 모듈, LLM 무관.
- **RewardModel** — `faithfulness(summary, source) -> float`. 교체 가능(lexical fallback → NLI).
- **Glossary** — `gate(text) -> List[ActiveTerm]`. 문자열/임베딩 매칭 전략 주입 가능.

---

## 3. 상세 명세

### 3.1 정책망 (policy.py)
- 입력 특징: `x_t = [h_XQ; h_SQ; h_GQ; h_Q]` (+ 선택적 대조 특징 `h_* − h_Q`).
- 구조: `LayerNorm → Linear(F→H) → GELU → Dropout(0.1) → Linear(H→H) → GELU → Dropout(0.1) → Linear(H→4)`.
- 출력 제약: `a = sigmoid(u_a) ∈ (0,1)`, `[b,c,d] = softmax([u_b,u_c,u_d])`, 합=1.
- 초기화: 마지막 Linear weight `N(0, 0.01)`, bias 0 → warm-start (`a≈0.5, b≈c≈d≈1/3`).
- 정밀도: fp32.

### 3.2 PMI 결합식 (decoder.py)
```
logit_combined = (1 + a) · (b·logit_XQ + c·logit_SQ + d·logit_GQ) − a·logit_Q
P = softmax(logit_combined / T)          # nucleus(top-p) 필터 후 샘플
```
- greedy 모드: `argmax(logit_combined)` (baseline·평가용).
- 그래디언트 경로: 토큰 logπ → 결합 로짓 → (a,b,c,d) → θ. LLM 로짓은 상수(detach).
- 길이 제어: `max_new_tokens`, `min_new_tokens`(EOS 마스킹), EOS 종료.

### 3.3 용어사전 게이팅 (glossary.py)
- 용어사전 포맷: `{표준용어: [트리거1, 트리거2, ...]}`.
- `gate(text)`: 원문에 트리거가 등장한 표준용어만 활성으로 선별.
- 매칭 전략: 문자열 매칭(기본) + 임베딩 유사도(선택, 인터페이스만).
- 활성 용어 → GQ 갈래 텍스트 구성 및 (선택) 로짓 부스트에 사용.

### 3.4 보상 (rewards.py) — reference-free
- **Faithfulness [0,1]**: NLI/FactKB 인터페이스 + lexical(토큰 겹침) fallback.
- **TripletCoverage [0,1]**: 요약에 등장한 head/tail 개체 비율. 문자열 매칭(+임베딩 선택).
- **TermUsage [0,1]**: 활성 표준용어 중 요약이 사용한 비율.
- **LengthPenalty**: 목표 길이 초과분 + n-gram 반복 페널티.
- `R = w1·Faith + w2·Cov + w3·Term − w4·LenPen` (기본 1.0/1.0/0.5/0.2).
- 정규화: `R̃ = (R − μ)/(σ + ε)`.

### 3.5 SCST 학습 (train.py)
- N개 롤아웃 → self-critical baseline `b = mean_i R̃_i`.
- advantage `A_i = R̃_i − b`.
- 손실 `L = −(1/N) Σ_i A_i · Σ_t (mask_t · logπ_t)` (− β·entropy 선택).
- AdamW(lr 1e-5~5e-5, wd 0.01), warmup→cosine, grad clip 1.0, grad accum 4~8.
- frozen 보장: `llm.requires_grad_(False)` + θ 외 grad None assert.
- 로깅: 평균 보상·구성요소·a/b/c/d 분포·엔트로피·grad 노름·lr.
- 체크포인트: save_every + best 보관 (opt/sched 상태 포함).

---

## 4. 테스트 전략 (TDD)
- 각 수식/제약을 유닛테스트로 먼저 작성:
  - 정책망 출력 제약 (`0<a<1`, `b+c+d=1`), warm-start 근사값.
  - 결합식이 특정 가중치에서 기대 로짓을 내는지.
  - 게이팅이 트리거 등장/미등장을 정확히 구분.
  - coverage/term/length 보상 경계값.
  - frozen assert: 학습 스텝 후 LLM 파라미터 grad가 None.
- **MockBackend**: 결정적 logit/hidden을 내는 가짜 백엔드로 CPU에서 디코더·학습 루프 end-to-end smoke test.
- 실제 HF 모델 연결은 별도 통합 테스트(선택, 무거우면 skip 마크).

---

## 5. 완료 기준
- 모든 유닛테스트 통과 (CPU, mock 백엔드).
- MockBackend로 SCST 1스텝이 실행되고 정책망 θ만 갱신됨(LLM grad None) 확인.
- 실제 백본/데이터/용어사전은 config·인터페이스 교체만으로 연결 가능.
