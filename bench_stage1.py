"""
Stage 1: 양자화 방법별 벤치마크 (NF4 / AWQ / HQQ)
지표: TTFT, tokens/s, peak VRAM, Perplexity(WikiText-2)

사용법:
  .venv/bin/python bench_stage1.py --methods nf4 awq hqq
  .venv/bin/python bench_stage1.py --methods nf4 --skip-ppl   # 속도만
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AWQ_MODEL  = "Qwen/Qwen2.5-7B-Instruct-AWQ"

DECODE_STEPS = 64
WARMUP_RUNS  = 2

# perplexity: WikiText-2 테스트셋에서 사용할 최대 토큰 수
PPL_MAX_TOKENS = 4096


# ── 모델 로더 ──────────────────────────────────────────────────────────

def load_nf4(model_id: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_awq(model_id: str):
    # transformers 5.x는 AWQ 백엔드로 gptqmodel을 요구하므로 autoawq 직접 API 사용
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
    from awq import AutoAWQForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_id,
        fuse_layers=False,  # 호환성 우선
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_hqq(model_id: str):
    from hqq.core.quantize import BaseQuantizeConfig
    from hqq.models.hf.base import AutoHQQHFModel

    quant_cfg = BaseQuantizeConfig(nbits=4, group_size=64)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # FP16으로 로드 후 HQQ 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    AutoHQQHFModel.quantize_model(model, quant_config=quant_cfg, compute_dtype=torch.float16)
    model.eval()
    return tok, model


LOADERS = {
    "nf4": (load_nf4, BASE_MODEL),
    "awq": (load_awq, AWQ_MODEL),
    "hqq": (load_hqq, BASE_MODEL),
}


# ── 벤치마크 측정 ──────────────────────────────────────────────────────

def make_input(prompt_len: int, batch_size: int, tokenizer, device: str = "cuda:0"):
    token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
    ids  = torch.tensor([[token_id] * prompt_len], dtype=torch.long, device=device)
    ids  = ids.repeat(batch_size, 1)
    mask = torch.ones_like(ids)
    return ids, mask


def measure(tokenizer, model, batch_size: int, prompt_len: int) -> dict:
    device = next(model.parameters()).device
    input_ids, attention_mask = make_input(prompt_len, batch_size, tokenizer, str(device))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cur_mask = attention_mask

    # Decode
    decode_times: list[float] = []
    for _ in range(DECODE_STEPS):
        cur_mask = torch.cat(
            [cur_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
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
        decode_times.append((time.perf_counter() - t0) * 1000)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    stable = decode_times[2:]
    inter_ms = statistics.median(stable)

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "ttft_ms": round(ttft_ms, 1),
        "inter_token_ms": round(inter_ms, 2),
        "tokens_per_sec": round(batch_size / (inter_ms / 1000), 1),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "decode_p50_ms": round(statistics.median(decode_times), 2),
        "decode_p95_ms": round(sorted(decode_times)[int(len(decode_times) * 0.95)], 2),
    }


def warmup(tokenizer, model):
    device = str(next(model.parameters()).device)
    ids, mask = make_input(128, 1, tokenizer, device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur_mask = mask
        for _ in range(4):
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
            )
            out = model(
                input_ids=next_tok,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()


# ── Perplexity ──────────────────────────────────────────────────────────

def compute_perplexity(tokenizer, model, max_tokens: int = PPL_MAX_TOKENS) -> float:
    """WikiText-2 test set perplexity (최대 max_tokens 토큰 사용)."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0, :max_tokens].to(next(model.parameters()).device)

    stride     = 512
    seq_len    = input_ids.size(0)
    nlls: list[float] = []

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end        = min(begin + stride, seq_len)
        target_len = end - prev_end  # 새로 예측할 토큰 수

        chunk_input  = input_ids[begin:end].unsqueeze(0)
        chunk_target = chunk_input.clone()
        # 이전 컨텍스트 부분은 loss 계산에서 제외
        chunk_target[:, :-target_len] = -100

        with torch.no_grad():
            loss = model(chunk_input, labels=chunk_target).loss
        nlls.append(loss.item() * target_len)
        prev_end = end

    ppl = math.exp(sum(nlls) / seq_len)
    return round(ppl, 2)


# ── 메인 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+", default=["nf4", "awq", "hqq"],
        choices=["nf4", "awq", "hqq"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--prompt-lens",  nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--skip-ppl", action="store_true", help="Perplexity 측정 건너뜀")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    all_results: dict[str, dict] = {}

    for method in args.methods:
        loader_fn, model_id = LOADERS[method]
        print(f"\n{'='*60}")
        print(f"  방법: {method.upper()}  모델: {model_id}")
        print(f"{'='*60}")

        print("Loading model ...")
        tok, model = loader_fn(model_id)
        torch.cuda.empty_cache()

        print(f"Warmup ({WARMUP_RUNS}x) ...")
        for _ in range(WARMUP_RUNS):
            warmup(tok, model)

        bench_rows: list[dict] = []
        total = len(args.batch_sizes) * len(args.prompt_lens)
        done  = 0
        for bs in args.batch_sizes:
            for pl in args.prompt_lens:
                done += 1
                print(f"  [{done}/{total}] batch={bs} prompt={pl} ...", end=" ", flush=True)
                try:
                    r = measure(tok, model, bs, pl)
                    bench_rows.append(r)
                    print(
                        f"TTFT={r['ttft_ms']}ms  "
                        f"decode={r['inter_token_ms']}ms/tok  "
                        f"{r['tokens_per_sec']} tok/s  "
                        f"VRAM={r['peak_vram_mb']}MB"
                    )
                except torch.cuda.OutOfMemoryError:
                    print("OOM — skipped")
                    torch.cuda.empty_cache()

        ppl = None
        if not args.skip_ppl:
            print("  Perplexity (WikiText-2) 계산 중 ...")
            try:
                ppl = compute_perplexity(tok, model)
                print(f"  Perplexity: {ppl}")
            except Exception as e:
                print(f"  Perplexity 실패: {e}")

        all_results[method] = {"bench": bench_rows, "perplexity": ppl}

        # 메모리 해제
        del model
        torch.cuda.empty_cache()

    # 저장
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage1_bench.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n결과 저장: {out_path}")

    # 요약 표
    print("\n=== 방법별 요약 (batch=1, prompt=512) ===")
    header = f"{'method':<6}  {'tok/s':>8}  {'VRAM(MB)':>10}  {'PPL':>8}"
    print(header)
    print("-" * len(header))
    for method, data in all_results.items():
        row = next(
            (r for r in data["bench"] if r["batch_size"] == 1 and r["prompt_len"] == 512),
            None,
        )
        tps  = row["tokens_per_sec"] if row else "—"
        vram = row["peak_vram_mb"]   if row else "—"
        ppl  = data["perplexity"] or "—"
        print(f"{method:<6}  {str(tps):>8}  {str(vram):>10}  {str(ppl):>8}")


if __name__ == "__main__":
    main()
