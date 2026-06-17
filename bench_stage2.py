#!/usr/bin/env python3
"""
Stage 2: HF 직렬 서빙 vs vLLM 연속 배칭 — 동시 부하 벤치마크

핵심 질문: 동시 요청이 늘어날 때 TTFT·처리량이 어떻게 달라지는가?

측정 지표:
  - mean / p95 TTFT (Time To First Token): 요청 전송 → 첫 토큰 수신까지
  - mean e2e latency: 요청 전송 → 마지막 토큰 수신까지
  - 전체 처리량(tok/s): concurrency 개의 요청이 동시에 진행되는 시간 기준

서버 생명주기를 자동 관리 (vLLM → HF 순서로 하나씩 실행, GPU 경합 없음).

실행:
    uv run python bench_stage2.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ── 설정 ──────────────────────────────────────────────────────────────────

MODEL_AWQ = "Qwen/Qwen2.5-7B-Instruct-AWQ"
VLLM_PORT = 8001
HF_PORT = 8002
MAX_TOKENS = 100
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
RESULTS_FILE = Path("results/stage2_comparison.json")

VLLM_CMD = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_AWQ,
    "--quantization", "awq",
    "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "2048",
    "--max-num-seqs", "256",
    "--enable-prefix-caching",
]
VLLM_ENV = {**os.environ, "VLLM_ATTENTION_BACKEND": "XFORMERS"}

HF_CMD = [sys.executable, "serve_hf.py", "--port", str(HF_PORT)]

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Describe photosynthesis including light-dependent and independent reactions.",
    "Write a haiku about the ocean.",
    "What is machine learning?",
    "Explain the theory of relativity briefly.",
    "Name three tropical fruits.",
    "Describe the fall of the Roman Empire.",
    "What is 2 + 2?",
    "Explain supervised vs unsupervised learning.",
    "What is the speed of light?",
    "Describe neural network backpropagation.",
    "Write a short poem about spring.",
    "How does the internet work?",
    "What is DNA?",
    "Explain Adam Smith's economic theories.",
]


# ── 서버 관리 ──────────────────────────────────────────────────────────────

class ManagedServer:
    def __init__(self, name: str, cmd: list[str], health_url: str, log_path: Path, env: dict | None = None):
        self.name = name
        self.cmd = cmd
        self.health_url = health_url
        self.log_path = log_path
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self, timeout: int = 200) -> None:
        print(f"\n[{self.name}] 서버 시작 중 (로그 → {self.log_path})")
        self.log_path.parent.mkdir(exist_ok=True)
        self._log_fh = self.log_path.open("w")
        self._proc = subprocess.Popen(
            self.cmd,
            stdout=self._log_fh,
            stderr=self._log_fh,
            preexec_fn=os.setsid,
            env=self.env,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._log_fh.flush()
                raise RuntimeError(
                    f"[{self.name}] 프로세스가 종료됨 — 로그 확인: {self.log_path}"
                )
            try:
                r = httpx.get(self.health_url, timeout=3)
                if r.status_code == 200:
                    print(f"[{self.name}] 준비 완료 (PID {self._proc.pid})")
                    return
            except Exception:
                pass
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"[{self.name}] 시작 타임아웃 ({timeout}s) — 로그: {self.log_path}")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=20)
            except Exception:
                self._proc.kill()
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        print(f"[{self.name}] 종료")


# ── 벤치마크 클라이언트 ────────────────────────────────────────────────────

async def _single(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    """SSE 스트림을 파싱해 TTFT / e2e latency / 생성 토큰 수를 반환."""
    t0 = time.perf_counter()
    ttft: float | None = None
    token_count = 0

    async with client.stream(
        "POST", url,
        json={
            "model": MODEL_AWQ,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
        },
        timeout=360.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            text = chunk["choices"][0]["text"]
            finish = chunk["choices"][0].get("finish_reason")
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                token_count += 1
            if finish == "stop":
                break

    e2e = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else e2e, "latency": e2e, "tokens": token_count}


async def _warmup(url: str, name: str) -> None:
    print(f"  [{name}] 웜업 ...", end=" ", flush=True)
    async with httpx.AsyncClient() as client:
        await _single(client, url, PROMPTS[0])
    print("완료")


async def _run_concurrency(url: str, concurrency: int) -> dict:
    prompts = (PROMPTS * ((concurrency // len(PROMPTS)) + 1))[:concurrency]
    async with httpx.AsyncClient() as client:
        t_wall = time.perf_counter()
        results = await asyncio.gather(
            *[_single(client, url, p) for p in prompts],
            return_exceptions=True,
        )
        wall = time.perf_counter() - t_wall
    rows = [r for r in results if isinstance(r, dict)]
    if not rows:
        raise RuntimeError(f"모든 요청 실패 (concurrency={concurrency})")
    if len(rows) < concurrency:
        print(f"    ({len(rows)}/{concurrency} 성공)", end=" ")

    ttfts = sorted(r["ttft"] for r in rows)
    latencies = [r["latency"] for r in rows]
    total_tokens = sum(r["tokens"] for r in rows)
    p95 = ttfts[max(0, int(len(ttfts) * 0.95) - 1)]

    return {
        "concurrency": concurrency,
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3),
        "p95_ttft_s": round(p95, 3),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3),
        "throughput_tok_s": round(total_tokens / wall, 1),
        "total_tokens": total_tokens,
    }


async def benchmark(name: str, url: str) -> list[dict]:
    print(f"\n[{name}] 벤치마크 시작")
    await _warmup(url, name)
    rows = []
    for c in CONCURRENCY_LEVELS:
        print(f"  concurrency={c:2d} ...", end=" ", flush=True)
        row = await _run_concurrency(url, c)
        rows.append(row)
        print(
            f"mean_ttft={row['mean_ttft_s']:.2f}s  "
            f"p95_ttft={row['p95_ttft_s']:.2f}s  "
            f"{row['throughput_tok_s']:.1f} tok/s"
        )
    return rows


# ── 결과 출력 ──────────────────────────────────────────────────────────────

def print_table(hf: list[dict], vllm: list[dict]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  HF (AWQ, 직렬)  vs  vLLM (AWQ, 연속 배칭)  |  max_tokens={MAX_TOKENS}")
    print("  * TTFT: 요청 전송 → 첫 토큰 수신 (낮을수록 좋음)")
    print("  * tok/s: N개 동시 요청의 전체 처리량 (높을수록 좋음)")
    print(f"{'conc':>5}  {'HF TTFT':>9}  {'vLLM TTFT':>10}  {'HF tok/s':>9}  {'vLLM tok/s':>11}  {'배율':>6}")
    print("-" * 80)
    hm = {r["concurrency"]: r for r in hf}
    vm = {r["concurrency"]: r for r in vllm}
    for c in CONCURRENCY_LEVELS:
        h, v = hm[c], vm[c]
        ratio = v["throughput_tok_s"] / h["throughput_tok_s"] if h["throughput_tok_s"] else 0
        print(
            f"  {c:>3}  "
            f"{h['mean_ttft_s']:>8.2f}s  "
            f"{v['mean_ttft_s']:>9.2f}s  "
            f"{h['throughput_tok_s']:>8.1f}  "
            f"{v['throughput_tok_s']:>10.1f}  "
            f"{ratio:>5.2f}x"
        )
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"모델: {MODEL_AWQ}")
    print(f"max_tokens={MAX_TOKENS}  concurrency={CONCURRENCY_LEVELS}")

    logs = Path("results")
    vllm_server = ManagedServer(
        name="vLLM",
        cmd=VLLM_CMD,
        health_url=f"http://localhost:{VLLM_PORT}/health",
        log_path=logs / "stage2_vllm_server.log",
        env=VLLM_ENV,
    )
    hf_server = ManagedServer(
        name="HF",
        cmd=HF_CMD,
        health_url=f"http://localhost:{HF_PORT}/health",
        log_path=logs / "stage2_hf_server.log",
    )

    vllm_rows: list[dict] = []
    hf_rows: list[dict] = []

    try:
        # ① vLLM — 먼저 실행하고 측정 후 종료
        vllm_server.start(timeout=300)
        vllm_rows = await benchmark("vLLM", f"http://localhost:{VLLM_PORT}/v1/completions")
        vllm_server.stop()

        # ② HF — GPU 경합 없이 단독 실행
        hf_server.start(timeout=120)
        hf_rows = await benchmark("HF", f"http://localhost:{HF_PORT}/v1/completions")
        hf_server.stop()

    except KeyboardInterrupt:
        print("\n중단됨")
    except Exception as e:
        print(f"\n오류: {e}")
    finally:
        vllm_server.stop()
        hf_server.stop()

    if not (vllm_rows and hf_rows):
        print("결과 없음 — 종료")
        return

    result = {
        "model": MODEL_AWQ,
        "max_tokens": MAX_TOKENS,
        "vllm_continuous": vllm_rows,
        "hf_serial": hf_rows,
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {RESULTS_FILE}")
    print_table(hf_rows, vllm_rows)


if __name__ == "__main__":
    asyncio.run(main())
