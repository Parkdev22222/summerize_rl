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

## 실제 백본 학습 (EXAONE) + TensorBoard

CUDA 머신(H200 등)에서 EXAONE 백본을 받아 학습하고 실시간 모니터링:

```bash
python -m examples.train_exaone \
    --model LGAI-EXAONE/EXAONE-Deep-7.8B \
    --steps 2000 --log-dir runs/exaone --device cuda

# 다른 터미널에서 실시간 관찰
tensorboard --logdir runs/exaone
```

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

실제 백본 선정/다운로드, 실제 코퍼스·용어사전 구축, vLLM 롤아웃 통합, 대규모 학습 실행.
설계 문서: `docs/superpowers/specs/2026-07-22-pmi-weight-rl-design.md`.
