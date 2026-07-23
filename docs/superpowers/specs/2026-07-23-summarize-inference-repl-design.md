# 체크포인트 로드 요약 추론 REPL — 설계

## 배경

학습(`examples/train_real.py` → `SCSTTrainer.save_checkpoint`)은 `checkpoints/` 아래에
`best.pt`, `step<N>.pt` 형태로 체크포인트를 저장한다. 체크포인트 payload는

```python
{"policy": state_dict, "optimizer": ..., "scheduler": ..., "step": int, "best_reward": float}
```

이고, **학습되는 것은 소형 가중치 정책망(`WeightPolicy` MLP)뿐**이다. LLM 백본은 frozen이다.

목표: `checkpoints/`에 생성된 가중치를 **붙여서(load)**, 원문이 들어오면 요약을 내주는
추론 코드를 만든다. 인터페이스는 **대화형 REPL**, 백본은 **실제 `HFBackend`**, 체크포인트는
**기본 `checkpoints/best.pt` + `--ckpt`로 덮어쓰기**.

## 핵심 관찰

이 아키텍처에서 요약하려면 **원문 텍스트 X**가 필요하다. 사용자가 말한 "쿼리"는 요약
**지시문(query Q, 예: "군사 표준용어를 사용하여 요약하시오")** 이며 기본값을 두고 바꿀 수 있다.
추론 = 정책망 가중치 로드 → 4갈래 프롬프트 구성 → **greedy** PMI 디코딩.

## 구성 (격리 + 테스트 가능성)

### 1. `summarize_rl/infer.py` — 백본 비의존 `Summarizer` 코어

- `SummaryResult` 데이터클래스: `text: str`, `active_terms: list[str]`,
  `mean_weights: tuple[float, float, float, float]` (평균 a,b,c,d).
- `Summarizer(backend, policy, config, *, glossary=None)`
  - 생성 시 `freeze_llm(backend)` 호출, `policy.eval()`로 dropout 비활성(결정론적 추론).
- `load_checkpoint(path) -> int` (반환: step)
  - `torch.load(path, map_location=..., weights_only=False)`.
  - `payload["policy"]`를 정책망에 로드. payload에 `"policy"` 키가 없고 그 자체가
    state_dict처럼 보이면(값이 텐서) bare state_dict로 간주해 로드. 그 외에는 명확한 에러.
  - 파일 부재 시 `FileNotFoundError`에 안내 메시지.
- `summarize(source, *, query=None, triplets=None) -> SummaryResult`
  - `query`가 None이면 `Example`의 기본 지시문 사용.
  - 용어사전이 있으면 `glossary.gate(source)`로 활성 용어 산출, 없으면 빈 리스트.
  - `build_branches` → `torch.no_grad()` 안에서 `generate(greedy=True)` → 결과 조립.
  - `mean_weights`는 `rollout.weight_trace` 평균(빈 경우 0으로).

정책망은 백본과 같은 device에 있어야 한다(호출자 책임). CLI가 맞춰서 올린다.

### 2. `examples/summarize.py` — REPL CLI (`python -m examples.summarize`)

인자:
- `--model` (필수): HF 백본 이름/경로.
- `--ckpt` (기본 `checkpoints/best.pt`): 로드할 정책망 체크포인트.
- `--glossary` (기본 None → 데모 `MILITARY_GLOSSARY`): `{표준용어: [트리거...]}` JSON.
- `--device` (기본 `cuda`), `--dtype` (기본 `bfloat16`).
- `--query` (기본: `Example`의 기본 지시문): 요약 지시문.
- `--max-new-tokens`, `--min-new-tokens`: 디코딩 오버라이드(선택).

흐름:
1. `HFBackend` 생성(frozen).
2. `Config()` 생성 후 백본에 동기화: `policy.llm_hidden_size = backend.hidden_size`,
   `decode.eos_token_id`, `decode.pad_token_id`. (학습과 동일한 기본 `PolicyConfig`를
   써야 체크포인트 아키텍처가 맞는다 — 문서에 명시.)
3. `WeightPolicy(cfg.policy).to(device)` 생성 → `Summarizer` 구성 → `load_checkpoint`.
4. **REPL 루프**: 원문을 여러 줄 붙여넣고 **빈 줄**로 입력 종료 → 요약 + 활성 표준용어 +
   평균 `[a,b,c,d]` 출력.

REPL 명령어(`:` 접두):
| 명령어 | 동작 |
|---|---|
| *(원문 + 빈 줄)* | 요약 실행 후 결과 출력 |
| `:query <지시문>` | 요약 지시문 변경 |
| `:ckpt <경로>` | 다른 체크포인트 즉시 재로드 |
| `:help` | 도움말 |
| `:q` / `:quit` / `:exit` | 종료 |

EOF(Ctrl-D)/`KeyboardInterrupt`도 정상 종료로 처리.

### 3. `tests/test_infer.py` — `MockBackend` 기반

- 학습→저장→로드→요약 왕복: `SCSTTrainer.save_checkpoint`로 저장한 payload를
  `Summarizer.load_checkpoint`가 읽어 요약을 낸다(비어 있지 않은 text).
- bare state_dict 로드 경로 동작.
- 없는 파일 경로 → `FileNotFoundError`.
- greedy 결정론: 같은 입력 두 번 → 같은 요약.

CLI는 HFBackend 기본이지만 `Summarizer`가 백본 비의존이라 `MockBackend`로 검증한다.

## 범위 밖 (YAGNI)

- REPL에서 트리플릿 입력(코어는 인자로 받되 REPL은 원문만). 배치. 샘플링(greedy만).
- 모델 샤딩/멀티 GPU.
