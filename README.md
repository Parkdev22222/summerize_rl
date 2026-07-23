# summarize_rl — 군사 도메인 PMI 가중치 RL

얼려둔(frozen) LLM 위에서 **PMI 결합 가중치 `[a, b, c, d]`만** 강화학습(SCST)으로
학습하는 핵심 알고리즘 라이브러리. 정답 요약 없이(reference-free) 학습하며, 근거 없는
표준용어 주입을 막는 용어사전 게이팅을 포함한다.

작업계획서 Phase 2~4의 핵심 알고리즘을 데이터/백본 선정과 독립적으로 구현한 것이다.
실제 한국어 7~13B 백본, 실제 원문·triplet 코퍼스, 실제 군사 용어사전은 인터페이스로
분리되어 나중에 교체할 수 있다.

## 핵심 아이디어

매 토큰 t에서 소형 MLP 정책망이 4개 가중치를 내고, 이 가중치로 4갈래 로짓을 결합한다:

```
logit_c = (1 + a) · (b·logit_XQ + c·logit_SQ + d·logit_GQ) − a·logit_Q
          a ∈ (0,1),   b + c + d = 1
```

- `XQ` 원문+질의 · `SQ` triplet+질의 · `GQ` 게이팅 용어사전+질의 · `Q` 질의(prior)
- **LLM은 frozen** — 로짓/hidden만 제공. 그래디언트는 정책망 θ에만 흐른다.

## 설치

```bash
pip install torch            # 핵심 라이브러리 + MockBackend + 테스트
pip install transformers     # 실제 백본(HFBackend) 사용 시에만
```

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `config.py` | 디코딩/학습/보상 하이퍼파라미터 (작업계획서 5절) |
| `policy.py` | 가중치 정책망 MLP: 특징 → `[a,b,c,d]`, sigmoid/softmax 제약, fp32, warm-start |
| `glossary.py` | 용어사전 게이팅: `표준용어 ← 트리거`, 원문 등장 시만 활성 |
| `branches.py` | 4갈래 프롬프트 구성 + triplet 직렬화 |
| `triplets.py` | 구조화 보고서(`### 섹션` / `- **필드:** 값`)에서 triplet 규칙 추출 |
| `data.py` | JSONL 코퍼스 로더: 레코드 → `Example`(triplet 추출·train/eval 분할) |
| `llm_backend.py` | frozen LLM 래퍼. `MockBackend`(테스트용) / `HFBackend`(실모델) |
| `decoder.py` | 다갈래 PMI 결합식 + 샘플링/greedy 디코딩 |
| `rewards.py` | Faithfulness · TripletCoverage · TermUsage · LengthPenalty + 정규화 |
| `train.py` | SCST 학습 루프 + `fit`/`evaluate` 드라이버 (N 롤아웃, self-critical baseline, 마스크 손실, AdamW, clip, accum, ckpt) |
| `logging_utils.py` | TensorBoard 로거 (미설치 시 no-op 폴백) |

## 데모 (모델 다운로드 불필요)

```bash
python -m examples.run_demo --steps 20
```

`MockBackend`로 전체 파이프라인(게이팅 → 4갈래 → PMI 디코딩 → 보상 → advantage →
정책 갱신)을 CPU에서 실행하고 스텝별 보상·가중치·grad 노름을 출력한다.

## 데이터 (군사 시나리오 코퍼스)

`data/military_scenarios.jsonl` (150건, `train` 140 / `eval` 10). 각 레코드:

| 필드 | 의미 |
|---|---|
| `source_text` | 원문 보고서 X (구조화된 영어 시나리오) |
| `summary_text` | 정답 요약(한국어) — **학습에는 안 씀**(reference-free), 평가용으로만 보관 |
| `keyfacts` | 원자 사실 리스트(한국어) — 평가/분석용 |
| `split` | `train` / `eval` |

정답 triplet은 코퍼스에 없으므로 `triplets.py`가 `source_text`의 구조
(`### 섹션` → `#### 하위섹션` → `- **필드:** 값` / 번호목록 / 일반 불릿)를 훑어
규칙 기반으로 추출한다. 결정적·무의존(LLM/파서 불필요)이라 같은 원문은 항상 같은
triplet을 낸다. 이렇게 뽑은 triplet이 SQ(핵심정보) 갈래를 채운다.

```python
from summarize_rl.data import load_examples
corpus = load_examples()                 # data/military_scenarios.jsonl 기본
print(len(corpus.train), len(corpus.eval))   # 140 10
ex = corpus.train[0]
print(ex.triplets)      # source_text에서 추출된 triplet
print(ex.reference)     # summary_text (평가용, 보상엔 미사용)
```

> 참고: 원문은 영어, 요약은 한국어다. 현재 lexical 보상(`Faithfulness`/`Coverage`)은
> 문자열 겹침 기반이라 교차언어에서는 약하다. 실전에서는 다국어 NLI/FactKB를
> `FaithfulnessModel`로 주입하거나 `keyfacts`(한국어) 기반 커버리지로 교체하는 것을
> 권장한다.

## 실제 백본 학습 (EXAONE) + TensorBoard

CUDA 머신(H200 등)에서 EXAONE 백본을 받아 학습하고 실시간 모니터링:

```bash
python -m examples.train_exaone \
    --model LGAI-EXAONE/EXAONE-Deep-7.8B \
    --steps 2000 --log-dir runs/exaone --device cuda

# 다른 터미널에서 실시간 관찰
tensorboard --logdir runs/exaone
```

### GPU 개수 제어

```bash
# 단일 GPU (7.8B는 H200 1장에 충분)
python -m examples.train_exaone --num-gpus 1

# 여러 장에 백본 샤딩 (더 큰 백본/헤드룸용). H200이면 장당 ~120GiB
python -m examples.train_exaone --num-gpus 2 --max-memory-per-gpu 120GiB

# 특정 GPU만 지정 (CUDA_VISIBLE_DEVICES 자동 설정)
python -m examples.train_exaone --gpu-ids "0,3"
```

- `--num-gpus 1`: `cuda:0` 단일 로드
- `--num-gpus N>1`: `device_map="auto"` + `max_memory`로 정확히 N장에만 샤딩
- `--gpu-ids "0,3"`: 그 GPU들만 보이게 고정(`--num-gpus`보다 우선). torch가 CUDA를
  초기화하기 **전에** `CUDA_VISIBLE_DEVICES`를 설정하므로 정확히 적용됨
- 미지정 시: `--device` 값을 그대로 사용(기존 동작)

> 참고: 정책망은 항상 fp32 단일 장치이고 아주 작다. 멀티 GPU는 **얼린 백본 샤딩**을
> 위한 것이므로 7.8B에는 보통 1장이면 된다.

EXAONE 관련:
- 커스텀 모델링 코드를 쓰므로 `trust_remote_code=True` (기본 적용)
- instruct/reasoning 튜닝 모델이라 각 갈래에 chat template 적용 (`--no-chat-template`로 해제)
- **EXAONE-Deep은 추론 모델**(긴 `<thought>` 생성)이라 요약엔 `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`가 더 자연스러움 — `--model`로 교체

### TensorBoard에 기록되는 것 (실시간 성능)

| 그룹 | 스칼라 |
|---|---|
| `reward/` | mean, faithfulness, coverage, term_usage, length_penalty |
| `weights/` | a, b, c, d (정책망 가중치 평균) |
| `train/` | loss, grad_norm, lr, entropy |
| `gen/` | mean_len |
| `eval/` | 검증셋 greedy 요약의 위 지표 (`--eval-every`마다) |

### 코드로 직접 (`fit` 드라이버)

```python
from summarize_rl.config import Config
from summarize_rl.llm_backend import HFBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from summarize_rl.glossary import Glossary
from summarize_rl.logging_utils import TensorBoardLogger

backend = HFBackend(
    "LGAI-EXAONE/EXAONE-Deep-7.8B", device="cuda", dtype="bfloat16",
    trust_remote_code=True, use_chat_template=True,
)
cfg = Config()
cfg.policy.llm_hidden_size = backend.hidden_size   # 정책망 입력 차원 자동 맞춤
cfg.decode.eos_token_id = backend.eos_token_id

policy = WeightPolicy(cfg.policy)                  # fp32, LLM과 무관
glossary = Glossary({"기동": ["이동", "전진"]})
trainer = SCSTTrainer(policy, backend, cfg, glossary=glossary)

logger = TensorBoardLogger("runs/exp1")
trainer.fit(
    train_examples,                # list[Example]
    logger=logger,
    val_dataset=val_examples,
    eval_every=100,
)
logger.close()
```

`fit`이 롤아웃·보상·advantage·정책 갱신·로깅·체크포인트·주기 검증을 모두 처리한다.
보상의 `Faithfulness`는 기본적으로 lexical fallback을 쓰며, NLI/FactKB 모델을
`FaithfulnessModel` 인터페이스로 주입해 교체할 수 있다.

## 설계상 보장

- **LLM frozen**: `combine_logits`가 LLM 로짓을 detach → 그래디언트가 백본에 흐르지
  않음. `train_step`이 매 스텝 `assert_llm_frozen`으로 확인. (테스트로 검증됨)
- **제약 항상 만족**: `a ∈ (0,1)`, `b+c+d=1` (sigmoid/softmax).
- **정책망 fp32**, LLM은 bf16 가능.

## 테스트

```bash
python -m pytest tests/ -q
```

`tests/test_hf_backend_integration.py`는 로컬에서 소형 GPT-2를 만들어(다운로드 없음)
실제 transformers forward·KV캐시·hidden 추출 경로를 검증한다.

## 이번 범위 밖 (의존성)

실제 백본 다운로드/대규모 학습 실행, 실전 군사 용어사전 구축, 다국어 NLI/FactKB
보상, vLLM 롤아웃 통합. (군사 시나리오 코퍼스는 `data/`에 포함되어 연결됨.)
설계 문서: `docs/superpowers/specs/2026-07-22-pmi-weight-rl-design.md`.
