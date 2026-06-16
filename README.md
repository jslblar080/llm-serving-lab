# llm-bench

Qwen2.5-7B NF4를 직접 프로파일링하며 prefill/decode 병목을 이해하기 위한 실습 레포.

## 환경

- GPU: Tesla T4 (VRAM 15GB)
- CUDA 12.1 / Driver 580.159.03
- Python 3.12 / PyTorch 2.5.1+cu121
- 모델: Qwen/Qwen2.5-7B-Instruct (bitsandbytes NF4, double quant)

## 0단계 — prefill/decode 프로파일링

`past_key_values`를 직접 전달해 prefill(KV 캐시 구축)과 decode(토큰 1개씩 생성)를 분리 측정.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python bench_stage0.py --batch-sizes 1 4 8 --prompt-lens 128 512 1024
```

### 결과

| batch | prompt_len | TTFT (ms) | decode p50 (ms/tok) | tok/s | peak VRAM (MB) |
|------:|-----------:|----------:|--------------------:|------:|---------------:|
| 1 | 128 | 714.6 | 65.52 | 15.3 | 5,460 |
| 1 | 512 | 2,549.1 | 70.59 | 14.2 | 5,521 |
| 1 | 1,024 | 5,106.4 | 76.11 | 13.1 | 5,823 |
| 4 | 128 | 2,543.1 | 446.03 | 9.0 | 5,622 |
| 4 | 512 | 10,047.5 | 467.11 | 8.6 | 6,156 |
| 4 | 1,024 | 20,643.2 | 492.53 | 8.1 | 7,348 |
| 8 | 128 | 5,093.9 | 464.26 | 17.2 | 5,800 |
| 8 | 512 | 20,355.9 | 499.72 | 16.0 | 6,909 |
| 8 | 1,024 | 41,621.0 | 547.98 | 14.6 | 9,380 |

### 해석

**batch=1**: decode가 65ms/tok. T4 이론 하한(3.5GB 가중치 / 300 GB/s ≈ 11.7ms)의 약 5.6배로, memory-bandwidth bound에 NF4 dequantize 오버헤드가 얹힌 수준.

**batch=1 → 4**: decode step이 446ms/tok으로 6.8배 폭증. bitsandbytes NF4 커널이 GEMV에서 GEMM으로 전환되는 지점이고, 가중치를 매 스텝 fp16으로 dequantize하는 비용이 병목이 된다.

**batch=4 → 8**: 464ms/tok으로 거의 변하지 않음. dequantize 비용은 배치 크기가 아니라 가중치 크기에 비례하므로, GEMM 영역에 들어선 이후 추가 배치는 GPU 병렬성으로 흡수된다.
