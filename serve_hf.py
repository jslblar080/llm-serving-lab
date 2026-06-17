#!/usr/bin/env python3
"""
HF AWQ 서빙 서버 — /v1/completions (SSE 스트리밍)
vLLM OpenAI-compatible API와 동일한 응답 포맷.

실행:
    uv run python serve_hf.py [--port 8002]
"""
import asyncio
import json
import threading
import warnings
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_PORT = 8002

_model = None
_tokenizer = None
_gpu_lock = asyncio.Lock()  # HF는 GPU를 점유하므로 요청을 직렬화


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[HF-server] 모델 로드: {MODEL_ID}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoAWQForCausalLM.from_quantized(
        MODEL_ID, fuse_layers=False, trust_remote_code=True
    )
    _model.eval()
    print("[HF-server] 준비 완료")
    yield


app = FastAPI(lifespan=lifespan)


async def _sse_tokens(prompt: str, max_tokens: int):
    """TextIteratorStreamer를 asyncio.Queue로 브릿지해서 SSE 스트리밍."""
    from transformers import TextIteratorStreamer

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str | None] = asyncio.Queue()

    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )

    def _generate():
        _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
            streamer=streamer,
        )

    def _forward():
        for text in streamer:
            loop.call_soon_threadsafe(q.put_nowait, text)
        loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=_generate, daemon=True).start()
    threading.Thread(target=_forward, daemon=True).start()

    while True:
        token = await q.get()
        if token is None:
            break
        yield f"data: {json.dumps({'choices': [{'text': token, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    max_tokens = body.get("max_tokens", 100)

    async def _locked():
        # 락을 잡는 동안 스트리밍 — 이 동작이 HF의 직렬화 한계를 그대로 드러냄
        async with _gpu_lock:
            async for chunk in _sse_tokens(prompt, max_tokens):
                yield chunk

    return StreamingResponse(_locked(), media_type="text/event-stream")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
