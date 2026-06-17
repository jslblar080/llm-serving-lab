# llm-bench

Qwen2.5-7B를 양자화·서빙 스택별로 프로파일링하며 LLM 추론 병목을 이해하는 실습 레포.

목표: prefill/decode 구조 이해 → 양자화 트레이드오프 정량화 → vLLM 연속 배칭 효과 측정.

## 환경

- **GPU:** NVIDIA L4 (VRAM 24 GB)
- **CUDA:** 12.4 (cu124) / Driver 580.159.03
- **Python:** 3.12 / PyTorch 2.6.0+cu124 / transformers 5.12.1

## 실행

```bash
uv sync
```

## 진행 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 0 | prefill/decode 프로파일링 | 완료 |
| 1 | 양자화 방법별 벤치마크 + eval | 완료 |
| 2 | vLLM 서빙 스택 동시 부하 벤치마크 | 진행 중 |
| 3 | 프로덕션 인프라 + 모니터링 | 대기 |

---

## 0단계 — prefill/decode 프로파일링

`past_key_values`를 직접 전달해 prefill(KV 캐시 구축)과 decode(토큰 1개씩 생성)를 분리 측정.

```bash
uv run python bench_stage0.py --batch-sizes 1 4 8 --prompt-lens 128 512 1024
```

### 결과 (Qwen2.5-7B-Instruct NF4, NVIDIA L4)

| batch | prompt_len | TTFT (ms) | decode p50 (ms/tok) | tok/s | peak VRAM (MB) |
|------:|-----------:|----------:|--------------------:|------:|---------------:|
|     1 |        128 |     144.5 |               70.43 |  14.2 |          5,460 |
|     1 |        512 |     208.9 |               70.58 |  14.2 |          5,521 |
|     1 |      1,024 |     334.0 |               70.73 |  14.1 |          5,674 |
|     4 |        128 |     208.8 |              130.11 |  30.7 |          5,622 |
|     4 |        512 |     622.1 |              131.07 |  30.5 |          6,153 |
|     4 |      1,024 |   1,246.9 |              132.73 |  30.1 |          6,857 |
|     8 |        128 |     358.1 |              131.37 |  60.9 |          5,800 |
|     8 |        512 |   1,262.2 |              134.09 |  59.7 |          6,858 |
|     8 |      1,024 |   2,588.1 |              136.43 |  58.6 |          8,270 |

### 해석

**batch=1:** decode 70ms/tok. L4 이론 하한(3.5 GB 가중치 / 300 GB/s ≈ 11.7ms)의 약 6배로, memory-bandwidth bound에 NF4 dequantize 오버헤드가 얹힌 수준.

**batch=1 → 4:** 130ms/tok으로 1.85배 증가. bitsandbytes NF4 커널이 GEMV → GEMM으로 전환되는 지점. 가중치를 매 스텝 fp16으로 dequantize하는 비용이 병목이 된다.

**batch=4 → 8:** 131ms/tok으로 거의 변하지 않음. GEMM 영역에 진입한 이후 추가 배치는 GPU 병렬성으로 흡수.

---

## 1단계 — 양자화 방법별 벤치마크 + eval

NF4 / AWQ / HQQ 세 방법을 동일 모델에 적용해 속도·VRAM·품질을 비교.

```bash
uv run python bench_stage1.py --methods nf4 awq hqq
```

### 결과 (Qwen2.5-7B-Instruct, batch=1, prompt_len=512, NVIDIA L4)

| 방법 | tok/s | VRAM (MB) | Perplexity ↓ | ARC-Easy acc_norm ↑ |
|------|------:|----------:|-------------:|--------------------:|
| NF4  |  14.1 |     5,521 |         9.71 |               73.0% |
| AWQ  |  13.9 |     5,502 |         10.0 |               75.5% |
| HQQ  |   2.5 |     6,156 |         9.58 |               79.5% |

### 해석

**속도:** NF4/AWQ 모두 ~14 tok/s. HQQ는 순수 PyTorch 구현으로 CUDA 커널이 없어 5.7배 느림 — CUDA 커널 유무가 decode 속도의 결정적 변수.

**PPL vs Task accuracy 불일치:** PPL 순위(HQQ < NF4 < AWQ)와 ARC-Easy 순위(NF4 < AWQ < HQQ)가 역전됨. Perplexity는 downstream task accuracy의 완벽한 proxy가 아님.

**AWQ 선택 근거:** 속도·메모리·정확도 세 축에서 균형이 가장 좋음 → 서빙 스택(2단계)의 기본 모델로 채택.

### 환경 특이사항

- CUDA 13.0 드라이버로 pre-built GPTQ wheel 설치 불가 → HQQ(pure PyTorch)로 대체
- transformers 5.x에서 AWQ 백엔드가 gptqmodel로 변경됨 → autoawq 직접 API로 우회

---

## 2단계 — vLLM 서빙 스택 동시 부하 벤치마크 (진행 중)

**핵심 질문:** 동시 요청이 쌓이면 TTFT와 처리량이 어떻게 달라지는가?

| 구분 | HF 서버 (`serve_hf.py`) | vLLM 서버 |
|------|------------------------|-----------|
| 배칭 방식 | 요청 직렬화 (GPU 락) | continuous batching |
| 메모리 관리 | 정적 할당 | PagedAttention |
| API | FastAPI + SSE 스트리밍 | OpenAI-compatible |

```bash
uv run python bench_stage2.py   # 서버 자동 실행·측정·종료
```

**측정 지표:** mean/p95 TTFT, e2e latency, 전체 처리량(tok/s)
**동시 사용자 수:** [1, 4, 8, 16, 32]
