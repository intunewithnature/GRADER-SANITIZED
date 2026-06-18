"""CPU LoRA fine-tune of the span proposer (granite-3.3-2b-instruct).

Transparent manual loop (no Trainer) so the RAM/precision choices are explicit:
  - base weights in bf16 for STORAGE (~5GB; fp32 would be ~10GB, over budget) — this
    CPU has no bf16 *acceleration*, so compute is unaccelerated but correct; notes are
    short so it stays tractable.
  - only LoRA adapter params train (kept fp32); gradient checkpointing + use_cache=False.
NB: this is LoRA on CPU, not literal 4-bit QLoRA (bitsandbytes 4-bit needs CUDA). The
same script runs as true QLoRA on a GPU by loading 4-bit; see README.

Usage:
  python training/train_lora.py --smoke              # a few steps, prove it fits + runs
  python training/train_lora.py --epochs 2 --subset 0  # full run (subset 0 = all)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = "ibm-granite/granite-3.3-2b-instruct"
DATA = HERE / "data" / "train.jsonl"
OUT = HERE / "adapter"


def load_examples(max_len: int, tok, subset: int):
    rows = [json.loads(l) for l in DATA.open() if l.strip()]
    if subset:
        rows = rows[:subset]
    examples = []
    for r in rows:
        # tokenize via the string form (robust across transformers versions)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": r["input"]}], add_generation_prompt=True, tokenize=False
        )
        full_text = tok.apply_chat_template(
            [{"role": "user", "content": r["input"]}, {"role": "assistant", "content": r["target"]}],
            tokenize=False,
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_len]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        if len(labels) < len(full_ids):  # prompt got truncated away
            continue
        examples.append((torch.tensor([full_ids]), torch.tensor([labels])))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--subset", type=int, default=0)       # 0 = all
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--qlora", action="store_true", help="true 4-bit QLoRA (needs CUDA + bitsandbytes; use on Colab)")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (saves RAM, ~25%% slower)")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"qlora={args.qlora} dtype={args.dtype} ckpt={args.ckpt}  loading...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # build data first (fast) so any data bug fails before the model load
    examples = load_examples(args.max_len, tok, args.subset)
    print(f"examples: {len(examples)}  (building before model load)", flush=True)
    t0 = time.time()
    if args.qlora:
        # true 4-bit QLoRA — the GPU path (Colab). nf4 + double-quant, bf16 compute.
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, low_cpu_mem_usage=True)
        model.config.use_cache = False
        if args.ckpt:
            model.gradient_checkpointing_enable()
    print(f"  model loaded in {time.time()-t0:.0f}s", flush=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    max_steps = 6 if args.smoke else None
    total = len(examples)
    step = 0
    opt.zero_grad()
    running = 0.0
    t_start = time.time()
    n_epochs = 1 if args.smoke else args.epochs
    done = False
    for epoch in range(int(n_epochs) + 1):
        if done:
            break
        frac = n_epochs - epoch
        if frac <= 0:
            break
        n = total if frac >= 1 else int(total * frac)
        for i in range(n):
            ids, labels = examples[i]
            ids, labels = ids.to(device), labels.to(device)
            out = model(input_ids=ids, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                avg = running / args.grad_accum; running = 0.0
                rate = (time.time() - t_start) / max(1, ((epoch * total) + i + 1))
                print(f"  epoch{epoch} ex{i+1}/{n} optstep{step} loss={avg:.3f} {rate:.1f}s/ex", flush=True)
                if max_steps and step >= max_steps:
                    done = True; break

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
        print(f"SAVED adapter -> {OUT}", flush=True)
    print(f"DONE in {time.time()-t_start:.0f}s (smoke={args.smoke})", flush=True)


if __name__ == "__main__":
    main()
