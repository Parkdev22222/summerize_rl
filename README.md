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

### uv (권장)

```bash
uv sync                      # 가상환경(.venv) 생성 + 핵심 deps(torch) + dev(pytest,numpy)
uv sync --extra hf           # 실제 백본(HFBackend, transformers) 사용 시 추가
```

이후 모든 명령은 `uv run` 앞에 붙여 실행한다 (venv 자동 활성화):

```bash
uv run python -m examples.run_demo --steps 20
uv run pytest
```

의존성은 `pyproject.toml`에 선언되어 있고 `uv.lock`으로 고정된다.

### pip (대안)

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
| `train.py` | SCST 학습 루프 (N 롤아웃, self-critical baseline, 마스크 손실, AdamW, clip, accum, ckpt) |

## 데모 (모델 다운로드 불필요)

```bash
python -m examples.run_demo --steps 20
```

`MockBackend`로 전체 파이프라인(게이팅 → 4갈래 → PMI 디코딩 → 보상 → advantage →
정책 갱신)을 CPU에서 실행하고 스텝별 보상·가중치·grad 노름을 출력한다.

## 실제 백본 연결

```python
from summarize_rl.config import Config
from summarize_rl.llm_backend import HFBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from summarize_rl.glossary import Glossary

backend = HFBackend("<korean-7B-model>", device="cuda", dtype="bfloat16")
cfg = Config()
cfg.policy.llm_hidden_size = backend.hidden_size
cfg.decode.eos_token_id = backend.eos_token_id

policy = WeightPolicy(cfg.policy)          # fp32, LLM과 무관
glossary = Glossary({"기동": ["이동", "전진"], ...})
trainer = SCSTTrainer(policy, backend, cfg, glossary=glossary)

for step in range(cfg.train.total_steps):
    metrics = trainer.train_step(micro_batch)   # micro_batch: list[Example]
    if trainer.maybe_update_best(metrics.mean_reward):
        trainer.save_checkpoint("checkpoints/ckpt.pt", is_best=True)
```

보상의 `Faithfulness`는 기본적으로 lexical fallback을 쓰며, NLI/FactKB 모델을
`FaithfulnessModel` 인터페이스로 주입해 교체할 수 있다.

## 실제 학습 실행 (단일 GPU, CLI)

`examples/train_real.py`가 frozen `HFBackend` + triplet 코퍼스(JSONL) + `SCSTTrainer`를
CLI로 묶는다. 정책망(fp32)과 샘플링 generator를 백본과 같은 device로 올려 GPU 실행 시
device 불일치가 없다. 7~13B bf16 백본은 H100 80GB **한 장**에 올라간다.

```bash
uv sync --extra hf                          # transformers 포함
CUDA_VISIBLE_DEVICES=0 uv run python -m examples.train_real \
    --model <korean-7B-model> --dtype bfloat16 \
    --data data/scenarios_ko.jsonl \
    --steps 2000 --num-samples 5 --grad-accum 4 --ckpt-dir checkpoints
```

스모크 테스트: `--limit 4 --steps 5` 를 덧붙인다. 코퍼스는 `data/build_dataset.py`가
`data/parts/*.jsonl`(원문 한국어 번역 + triplet)을 병합해 `data/scenarios_ko.jsonl`로
만든다. 라이브러리는 모델 샤딩/데이터 병렬 롤아웃을 구현하지 않으므로 두 번째 H100은
현재 활용되지 않는다(더 큰 백본 샤딩은 `HFBackend`에 `device_map` 지원 추가 필요).

## 학습된 가중치로 요약 (추론 REPL)

학습이 `checkpoints/`에 저장한 정책망 가중치(`best.pt` 등)를 **붙여서** 요약을 낸다.
LLM은 frozen, 학습된 소형 정책망만 로드해 **greedy** PMI 디코딩한다. `--ckpt` 기본값은
`checkpoints/best.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m examples.summarize \
    --model <korean-7B-model> --dtype bfloat16 \
    --ckpt checkpoints/best.pt
```

원문을 붙여넣고 **빈 줄**로 입력을 끝내면 요약이 나온다. "쿼리"는 요약 **지시문**(`--query`,
기본 "…군사 표준용어를 사용하여 요약하시오.")이며 REPL에서 바꿀 수 있다.

| REPL 명령어 | 동작 |
|---|---|
| *(원문 + 빈 줄)* | 요약 출력 (+ 활성 표준용어, 평균 가중치 `[a,b,c,d]`) |
| `:query <지시문>` | 요약 지시문 변경 |
| `:ckpt <경로>` | 다른 체크포인트 즉시 재로드 |
| `:help` | 도움말 |
| `:q` / `:quit` / `:exit` | 종료 |

프로그램에서 재사용하려면 `summarize_rl.infer.Summarizer`를 직접 쓴다 (백본 비의존):

```python
from summarize_rl.infer import Summarizer
s = Summarizer(backend, policy, cfg, glossary=glossary)
s.load_checkpoint("checkpoints/best.pt")
result = s.summarize("적 부대가 이동 중이며 고지를 점령했다.")
print(result.text, result.active_terms, result.mean_weights)
```

## 상주 서버 + 대화형 클라이언트 (답만 확인)

무거운 백본을 매번 로드하지 않고, **GPU 1장에 상주시킨 서버**에 대화형 클라이언트로
질의해 **답(요약)만 확인**한다. `CUDA_VISIBLE_DEVICES`로 카드 1장 고정.

```bash
# 1) 서버: GPU 3번에 백본+체크포인트 상주
CUDA_VISIBLE_DEVICES=3 uv run python -m examples.serve \
    --model <korean-7B-model> --dtype bfloat16 \
    --ckpt checkpoints/best.pt --host 127.0.0.1 --port 8000

# 2) 클라이언트: 터미널 REPL (모델 없음, 서버에 POST만)
uv run python -m examples.summarize_client --server http://127.0.0.1:8000
```

> **왜 vLLM이 아닌가:** 이 요약은 4갈래 PMI *정책망* 디코딩이라 매 스텝 4브랜치의 logits
> 전체 + hidden state + 대조 결합이 필요하다. 표준 vLLM은 이를 노출하지 않아, 올려도
> **학습된 가중치가 적용된 요약이 나오지 않는다.** 그래서 `Summarizer`를 감싼 경량 HTTP
> 서버(stdlib, 추가 의존성 없음)로 상주시킨다.

엔드포인트: `GET /health`, `POST /summarize {source, query?}`, `POST /reload {ckpt}`.
클라이언트 명령어는 REPL과 동일(`:query`, `:ckpt`, `:help`, `:q`).

### 추론 속도 — 어텐션 커널

vLLM 엔진은 이 4갈래 PMI 디코딩을 못 돌리지만, 융합 어텐션 커널은 그대로 쓸 수 있다.
`serve.py`/`summarize.py`의 `--attn`으로 선택한다(수치 동일, 품질 영향 없음):

- `sdpa` (기본): torch SDPA, 어디서나 안전하게 빠름.
- `flash_attention_2`: 가장 빠름. `flash-attn` 설치 필요(없으면 자동으로 `sdpa`→`eager` 폴백).
- `eager`: 폴백/디버그용.

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m examples.serve \
    --model <model> --ckpt checkpoints/best.pt --attn flash_attention_2
```

### 추론 속도 — torch.compile + StaticCache (CUDA graph)

`--compile`을 주면 디코드를 **고정 크기 `StaticCache` + `torch.compile`된 모델**로 돌린다.
매 토큰 스텝의 shape가 고정되어 **CUDA graph로 캡처**되므로 스텝당 파이썬/런치 오버헤드가
크게 줄어든다(요청 간에도 같은 graph 재사용). 프리필은 길이가 가변이라 eager로 둔다.

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m examples.serve \
    --model <llama/qwen-계열> --ckpt checkpoints/best.pt \
    --attn flash_attention_2 --compile --max-seq-len 2048
```

- **StaticCache 호환 아키텍처**(Llama/Qwen/Mistral 등)가 필요하다.
- **첫 요청은 컴파일 때문에 느리고**, 이후부터 빨라진다.
- `--max-seq-len`은 (프롬프트+생성) 상한이자 스텝당 어텐션 폭이다. 실제 최대에 가깝게
  잡아라(너무 크면 스텝마다 낭비 어텐션이 늘어 오히려 느려질 수 있다).
- 출력은 기본(DynamicCache) 경로와 **토큰 단위로 동일**함이 테스트로 검증돼 있다
  (`tests/test_compile_static_integration.py`, 소형 Llama).

> 더 큰 가속으로 양자화(4-bit/AWQ/FP8, 메모리 대역폭↓)와 요청 연속 배칭도 가능하다.
> 특히 양자화는 정책망이 bf16 백본으로 학습됐으므로 요약이 달라질 수 있어 A/B 검증이 필요하다.

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
