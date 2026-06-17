"""
Stage 1 Perplexity 측정 (bench_stage1.py 결과에 PPL 추가)
결과를 stage1_bench.json에 덮어씁니다.

사용법:
  .venv/bin/python eval_ppl.py --methods nf4 awq hqq
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AWQ_MODEL  = "Qwen/Qwen2.5-7B-Instruct-AWQ"
PPL_MAX_TOKENS = 4096


def load_nf4(model_id):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_awq(model_id):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
    from awq import AutoAWQForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=False, trust_remote_code=True)
    model.eval()
    return tok, model


def load_hqq(model_id):
    from hqq.core.quantize import BaseQuantizeConfig
    from hqq.models.hf.base import AutoHQQHFModel
    quant_cfg = BaseQuantizeConfig(nbits=4, group_size=64)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    AutoHQQHFModel.quantize_model(model, quant_config=quant_cfg, compute_dtype=torch.float16)
    model.eval()
    return tok, model


LOADERS = {
    "nf4": (load_nf4, BASE_MODEL),
    "awq": (load_awq, AWQ_MODEL),
    "hqq": (load_hqq, BASE_MODEL),
}


def compute_perplexity(tokenizer, model, max_tokens: int = PPL_MAX_TOKENS) -> float:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0, :max_tokens].to(next(model.parameters()).device)

    stride  = 512
    seq_len = input_ids.size(0)
    nlls: list[float] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end        = min(begin + stride, seq_len)
        target_len = end - prev_end

        chunk_input  = input_ids[begin:end].unsqueeze(0)
        chunk_target = chunk_input.clone()
        chunk_target[:, :-target_len] = -100

        with torch.no_grad():
            loss = model(chunk_input, labels=chunk_target).loss
        nlls.append(loss.item() * target_len)
        prev_end = end

    return round(math.exp(sum(nlls) / seq_len), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["nf4", "awq", "hqq"],
                        choices=["nf4", "awq", "hqq"])
    parser.add_argument("--out", default="results/stage1_bench.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    for method in args.methods:
        loader_fn, model_id = LOADERS[method]
        print(f"\n[{method.upper()}] Loading model ...")
        tok, model = loader_fn(model_id)
        torch.cuda.empty_cache()

        print(f"[{method.upper()}] Perplexity 계산 중 (WikiText-2, {PPL_MAX_TOKENS} tokens) ...")
        ppl = compute_perplexity(tok, model)
        print(f"[{method.upper()}] Perplexity = {ppl}")

        if method not in existing:
            existing[method] = {}
        existing[method]["perplexity"] = ppl

        del model
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\n결과 저장: {out_path}")

    print("\n=== PPL 비교 ===")
    for method in args.methods:
        ppl = existing.get(method, {}).get("perplexity", "—")
        print(f"  {method:<6}: {ppl}")


if __name__ == "__main__":
    main()
