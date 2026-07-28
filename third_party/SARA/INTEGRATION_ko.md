# summarize_rl ↔ SARA 통합 노트

`summarize_rl`의 reference-free 보상 3종을 SARA의 self-critical 보상에 이식하고,
SARA가 우리 레포의 한국어 데이터로 학습·테스트할 수 있게 한 통합이다.

## 추가된 보상 (기본 가중치 0 → 켜야 동작)

| 보상 | argparse | 방향 | 입력 |
|---|---|---|---|
| Triplet 커버리지 | `--triplet_coverage_weight` | + | 데이터의 `triplets`([head,rel,tail]) |
| Hallucination 페널티 | `--hallu_weight` | − (감산) | 요약 vs 원문(부대·수치 정규식) |
| LLM-as-judge (Gemini) | `--judge_weight` | + | 원문 vs 요약, 0~100 정확도 |

- judge는 **Gemini API** 사용: `--judge_gemini_model`(기본 `gemini-2.5-flash`),
  키는 `GEMINI_API_KEY`(또는 `GOOGLE_API_KEY`) 환경변수. 키가 없으면 자동 비활성(0.0).
- judge는 (원문,요약) 캐시로 중복 호출을 줄이지만, RL 루프에서 rollout마다 API를
  호출하므로 대규모 학습에는 비용·rate limit 부담이 있다. 실험적으로만 켤 것.
- 세 가중치가 모두 0이면 SARA의 원래 보상(ROUGE+FactKB)과 **완전히 동일**하다.

## 우리 데이터로 학습·테스트

`--dataset summarize_rl_ko` 를 주면 `data/scenarios_ko.jsonl`(train 140 / eval 10)과
`data/test_scenarios_ko.jsonl`을 읽는다. 매핑: `source_text→document`,
`summary_text→summary`, `keyfacts(개행 join)→presumm`, `triplets→triplet 보상 입력`.
데이터 경로는 `$SUMMARIZE_RL_DATA`로 재정의 가능(기본: 리포의 `data/`).

예시(스모크):
```bash
cd third_party/SARA/src
GEMINI_API_KEY=... python test_performance_decoder_new_fc.py \
    --do_train --dataset summarize_rl_ko \
    --model_name_or_path <backbone> --loading_mode bf16 \
    --num_return_sequences 2 --batch_size 2 --debug_flag --debug_num 2 \
    --triplet_coverage_weight 1.0 --hallu_weight 0.5 \
    --judge_weight 0.0 --factkb_weight 0.0
```

## ⚠️ 알려진 제약: 백본(EXAONE 3.5 Instruct)

SARA의 핵심(context-aware FC 디코딩: presumm/null 브랜치 결합)은 **수정된 transformers
fork의 아키텍처별 클래스**(`LlamaForCausalLM_sft`, `OPTForCausalLM_sft`,
`GPTNeoForCausalLM_sft`, `MistralForCausalLM_sft`)에만 구현돼 있다
(`src/utils.py: configure_model_loading*`).

- **EXAONE 3.5 Instruct는 자체 아키텍처**(`trust_remote_code`)라 이 목록에 없다. 그대로
  로드하면 FC 결합/`presumm_input` 커널이 없어 context-aware 학습이 동작하지 않는다.
- 옵션:
  1. **Llama 계열 한국어 instruct 모델** 사용 → SARA의 `_sft` 경로를 그대로 활용(권장, 즉시 가능).
  2. SARA의 `_sft` 수정(FC 레이어 + generate의 presumm/null 결합)을 **EXAONE modeling
     클래스로 이식** → 별도의 큰 작업.

이번 통합의 **보상 3종 + 데이터 어댑터는 백본과 무관**하며 단위 테스트로 검증돼 있다
(`tests/test_reward_extras.py`, 14/14 통과). 백본만 위 옵션 중 하나로 정하면 된다.

## 검증

```bash
python third_party/SARA/tests/test_reward_extras.py     # pytest 없이도 실행
# 또는
python -m pytest third_party/SARA/tests/ -q             # torch 불필요(stdlib만)
```
