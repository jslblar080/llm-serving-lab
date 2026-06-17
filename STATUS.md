# 진행 상태

## 현재 단계
**2단계 (vLLM 서빙 스택)** — 진행 중

## 환경
- GPU: NVIDIA L4 (VRAM 24GB)
- CUDA: 12.4 (cu124) / Driver 580.159.03
- Python: 3.12 / PyTorch 2.6.0+cu124 / transformers 5.12.1

## 단계별 완료 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 0 | Qwen 7B 프로파일링 — TTFT/tokens·s/VRAM 측정표 | 완료 |
| 1 | 양자화 방법별 벤치마크 + eval | 완료 |
| 2 | vLLM 서빙 스택 동시 부하 벤치마크 | 진행 중 |
| 3 | 프로덕션 인프라 + 모니터링 | 대기 |
| 4 | CUDA 읽기/기여 (상시) | 대기 |

## 1단계 환경 특이사항
- CUDA 13.0 드라이버로 인해 auto-gptq / gptqmodel / llama-cpp-python(cu124) pre-built wheel 설치 불가
- transformers 5.x AWQ 백엔드가 autoawq → gptqmodel로 변경됨
- 해결: AWQ는 autoawq 직접 API, GPTQ 대신 HQQ(pure PyTorch) 사용

## 2단계 구조
- `serve_hf.py`: FastAPI + AWQ SSE 스트리밍 서버 (GPU 락으로 직렬화)
- `bench_stage2.py`: 서버 생명주기 자동 관리 + asyncio 동시 요청 클라이언트
- 비교 모델: Qwen/Qwen2.5-7B-Instruct-AWQ (양쪽 동일)
- autoawq 공식 deprecated 공지 (마지막 테스트 환경 torch 2.6.0 — 현재 환경과 일치)
