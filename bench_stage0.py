"""
Stage 0: Qwen 7B 4-bit profiling
측정 방법: model() 직접 호출 + past_key_values로 prefill/decode 분리
- Prefill: model(prompt_tokens) → KV 캐시 구축, 첫 토큰 생성
- Decode: model(token_1개, past_key_values=...) × N번 → 후속 토큰 생성
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DECODE_STEPS = 64   # decode 스텝 수 (TTFT 이후 생성할 토큰 수)
WARMUP_RUNS = 2


def load_model(model_id: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def make_input(prompt_len: int, batch_size: int, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
    ids = torch.tensor([[token_id] * prompt_len], dtype=torch.long, device="cuda:0")
    ids = ids.repeat(batch_size, 1)
    mask = torch.ones_like(ids)
    return ids, mask


def measure(tokenizer, model, batch_size: int, prompt_len: int) -> dict:
    input_ids, attention_mask = make_input(prompt_len, batch_size, tokenizer)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ── Phase 1: Prefill ──────────────────────────────────────────────
    # 프롬프트 전체를 한 번에 forward → KV 캐시 구축 + 첫 출력 토큰
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill_start) * 1000

    # 첫 토큰: 마지막 포지션 logits에서 argmax
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)

    # ── Phase 2: Decode ───────────────────────────────────────────────
    # 토큰 하나씩 forward, KV 캐시를 이어받아 이전 토큰 재계산 없음
    decode_times_ms: list[float] = []

    # attention_mask를 decode 스텝마다 1씩 늘려줘야 함
    cur_mask = attention_mask

    for _ in range(DECODE_STEPS):
        cur_mask = torch.cat(
            [cur_mask, torch.ones((batch_size, 1), dtype=torch.long, device="cuda:0")],
            dim=1,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        decode_times_ms.append((time.perf_counter() - t0) * 1000)

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # 첫 1~2 스텝은 GPU 스케줄링 잡음이 있으므로 스텝 3 이후로 통계
    stable = decode_times_ms[2:]
    inter_token_ms = statistics.median(stable)
    # throughput: 한 decode 스텝에서 batch_size개 토큰이 동시에 나옴
    tokens_per_sec = batch_size / (inter_token_ms / 1000)

    # TTFT = prefill 시간 (prefill 끝나는 순간 첫 토큰이 나옴)
    ttft_ms = prefill_ms

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "ttft_ms": round(ttft_ms, 1),
        "inter_token_ms": round(inter_token_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "decode_p50_ms": round(statistics.median(decode_times_ms), 2),
        "decode_p95_ms": round(sorted(decode_times_ms)[int(len(decode_times_ms) * 0.95)], 2),
    }


def warmup(tokenizer, model):
    input_ids, mask = make_input(128, 1, tokenizer)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur_mask = mask
        for _ in range(4):
            cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=torch.long, device="cuda:0")], dim=1)
            out = model(input_ids=next_tok, attention_mask=cur_mask, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--out", default="results/stage0_bench.json")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer, model = load_model(args.model)

    print(f"Warming up ({WARMUP_RUNS}x) ...")
    for _ in range(WARMUP_RUNS):
        warmup(tokenizer, model)

    results = []
    total = len(args.batch_sizes) * len(args.prompt_lens)
    done = 0

    for bs in args.batch_sizes:
        for pl in args.prompt_lens:
            done += 1
            print(f"[{done}/{total}] batch={bs}  prompt_len={pl} ...", flush=True)
            try:
                r = measure(tokenizer, model, bs, pl)
                results.append(r)
                print(
                    f"  TTFT(prefill)={r['ttft_ms']}ms"
                    f"  decode_median={r['inter_token_ms']}ms/tok"
                    f"  {r['tokens_per_sec']} tok/s"
                    f"  VRAM={r['peak_vram_mb']}MB"
                )
            except torch.cuda.OutOfMemoryError:
                print("  OOM — skipped")
                torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\n=== 측정 결과 ===")
    print(df.to_string(index=False))

    # 완료 기준 체크: compute vs memory bound 판단
    print("\n=== Roofline 분석 ===")
    print("T4 이론 decode 하한(NF4 7B, batch=1): ~11.7ms/token  (3.5GB / 300GB/s)")
    b1 = [r for r in results if r["batch_size"] == 1]
    if b1:
        avg_decode = statistics.mean(r["inter_token_ms"] for r in b1)
        ratio = avg_decode / 11.7
        print(f"실측 decode median(batch=1 평균): {round(avg_decode,2)}ms/token")
        print(f"이론 대비: {round(ratio,2)}x  (1.0에 가까울수록 bandwidth-bound에 근접)")

    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
