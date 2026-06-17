"""
Stage 1 Task Eval — ARC-Easy (0-shot, limit=200)
lm-evaluation-harness로 각 양자화 방법의 정확도를 측정합니다.
결과를 stage1_bench.json의 "arc_easy_acc" 필드에 씁니다.

사용법:
  .venv/bin/python eval_task.py --methods nf4 awq hqq
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AWQ_MODEL  = "Qwen/Qwen2.5-7B-Instruct-AWQ"

TASK       = "arc_easy"
NUM_FEWSHOT = 0
LIMIT      = 200  # 전체 2350개 중 200개 — 방법별 ~3분


# ── 모델 로더 ──────────────────────────────────────────────────────────

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
    awq_model = AutoAWQForCausalLM.from_quantized(
        model_id, fuse_layers=False, trust_remote_code=True,
    )
    awq_model.eval()
    # lm_eval은 표준 transformers 모델을 기대하므로 내부 모델을 꺼냄
    inner = awq_model.model
    inner.eval()
    return tok, inner


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


# ── 평가 ──────────────────────────────────────────────────────────────

def run_eval(tokenizer, model) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=4,
    )
    results = simple_evaluate(
        model=lm,
        tasks=[TASK],
        num_fewshot=NUM_FEWSHOT,
        limit=LIMIT,
        log_samples=False,
    )
    # arc_easy metric: acc_norm (normalized accuracy)
    task_res = results["results"][TASK]
    return {
        "acc":      round(task_res.get("acc,none",      0), 4),
        "acc_norm": round(task_res.get("acc_norm,none", 0), 4),
    }


# ── 메인 ───────────────────────────────────────────────────────────────

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
        print(f"\n{'='*50}")
        print(f"  [{method.upper()}]  모델: {model_id}")
        print(f"{'='*50}")
        print("Loading model ...")
        tok, model = loader_fn(model_id)
        torch.cuda.empty_cache()

        print(f"Running {TASK} (0-shot, limit={LIMIT}) ...")
        task_scores = run_eval(tok, model)
        print(f"  acc      = {task_scores['acc']:.4f} ({task_scores['acc']*100:.1f}%)")
        print(f"  acc_norm = {task_scores['acc_norm']:.4f} ({task_scores['acc_norm']*100:.1f}%)")

        if method not in existing:
            existing[method] = {}
        existing[method][TASK] = task_scores

        del model
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\n결과 저장: {out_path}")

    print(f"\n=== ARC-Easy 0-shot (limit={LIMIT}) 비교 ===")
    print(f"{'method':<6}  {'acc':>8}  {'acc_norm':>10}")
    print("-" * 30)
    for method in args.methods:
        scores = existing.get(method, {}).get(TASK, {})
        acc  = f"{scores.get('acc', 0)*100:.1f}%" if scores else "—"
        norm = f"{scores.get('acc_norm', 0)*100:.1f}%" if scores else "—"
        print(f"{method:<6}  {acc:>8}  {norm:>10}")


if __name__ == "__main__":
    main()
