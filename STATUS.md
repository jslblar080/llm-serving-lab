# 진행 상태

## 현재 단계
**1단계 (양자화 + eval)** — 진행 중

## 환경
- GPU: NVIDIA L4 (VRAM 24GB)
- CUDA: 13.0 / Driver 580.159.03
- Python: 3.12 / transformers: 5.12.1

## 1단계 환경 특이사항
- CUDA 13.0으로 인해 auto-gptq / gptqmodel / llama-cpp-python(cu124) 설치 불가
- transformers 5.x AWQ 백엔드가 autoawq → gptqmodel로 변경됨
- 해결: AWQ는 autoawq 직접 API, GPTQ 대신 HQQ(pure PyTorch) 사용

## 1단계 사용 방법
| 방법 | 라이브러리 | 비고 |
|------|-----------|------|
| NF4 | bitsandbytes | 통계적 가정, 캘리브레이션 없음 |
| AWQ | autoawq 직접 API | pre-quantized (Qwen/Qwen2.5-7B-Instruct-AWQ) |
| HQQ | hqq | on-the-fly 양자화, 캘리브레이션 없음, pure PyTorch |

## 단계별 완료 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 0 | Qwen 7B 프로파일링 — TTFT/tokens·s/VRAM 측정표 | 완료 |
| 1 | 양자화 방법별 벤치마크 + eval | 진행 중 |
| 2 | vLLM 서빙 스택 튜닝 | 대기 |
| 3 | 프로덕션 인프라 + 모니터링 | 대기 |
| 4 | CUDA 읽기/기여 (상시) | 대기 |
