"""Evaluate the QLoRA-fine-tuned proposer on a suite, reusing the project's harness.

Generation runs through transformers (base + LoRA adapter); the proposed spans are
parsed and verified by the SAME code paths as the Ollama eval (parse_span_json ->
blind_text -> verify_safe_text), so numbers are directly comparable to the stock model.
Run on Colab GPU right after training, or locally on CPU (slow). Fixture-off only.

  python training/eval_adapter.py --adapter training/adapter \
      --suite evals/private/contextual_heldout_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from grader.llm_pass import STRICT_SPAN_V1_PROMPT, parse_span_json  # noqa: E402
from grader.policy import load_policy  # noqa: E402
from grader.transform import blind_text  # noqa: E402
from grader.verify import verify_safe_text  # noqa: E402

BASE = "ibm-granite/granite-3.3-2b-instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "training/adapter"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--policy", default=str(ROOT / "policies/client_strategy.json"))
    ap.add_argument("--max-cases", type=int, default=0)   # 0 = all
    ap.add_argument("--max-new", type=int, default=192)
    args = ap.parse_args()

    policy = load_policy(Path(args.policy))
    tok = AutoTokenizer.from_pretrained(args.adapter)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows = [json.loads(l) for l in open(args.suite) if l.strip()]
    if args.max_cases:
        rows = rows[:args.max_cases]
    import tempfile
    passed = caused_known = 0
    lat = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(rows):
            prompt = STRICT_SPAN_V1_PROMPT + case["raw"]
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            lat.append((time.time() - t0) * 1000)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            spans = parse_span_json(gen, case["raw"]).spans
            blind = blind_text(case["raw"], policy, artifact_id=f"ft_{i}", base_dir=tmp, llm_spans=spans)
            v = verify_safe_text(blind.safe_text, policy, artifact_id=blind.artifact_id, base_dir=tmp,
                                 canaries=case.get("canaries", []), canary_field_types=case.get("expected_field_types", []))
            if v.passed:
                passed += 1
    import statistics
    n = len(rows)
    print(json.dumps({
        "suite": Path(args.suite).name, "adapter": args.adapter, "cases": n,
        "cases_passed": passed, "pass_rate": round(passed / n, 3) if n else 0,
        "p50_gen_ms": round(statistics.median(lat), 1) if lat else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
