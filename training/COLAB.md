# Training the proposer on Colab (true 4-bit QLoRA)

The proposer is fine-tuned on a **GPU** (Colab T4 is enough) — true 4-bit QLoRA via
bitsandbytes. The same `train_lora.py` runs as plain LoRA on CPU for a local smoke; add
`--qlora` for the real run. Granite-2B + ~950 short SFT pairs × 3 epochs is ~10–20 min on a T4.

Paste these into a Colab notebook (Runtime → GPU):

```python
# 1. get the code (public, self-contained — synthetic data only)
!git clone https://github.com/intunewithnature/GRADER-SANITIZED grader-repo
%cd grader-repo

# 2. deps
!pip -q install "transformers>=4.44" peft bitsandbytes datasets accelerate

# 3. build the span-extraction SFT data from the synthetic corpus
!python training/build_sft.py

# 4. true 4-bit QLoRA (nf4 + double-quant, bf16 compute) — saves training/adapter
!python training/train_lora.py --qlora --epochs 3

# 5. evaluate the adapter the SAME way as the stock model (fixture-off, reuses the harness)
!python training/eval_adapter.py --adapter training/adapter \
    --suite evals/private/contextual_heldout_100.jsonl
!python training/eval_adapter.py --adapter training/adapter \
    --suite evals/private/independent_150.jsonl

# 6. download the adapter (small — just the LoRA weights)
!cd training && zip -r adapter.zip adapter && echo "download training/adapter.zip"
```

## Using the fine-tuned model back in the main pipeline (optional)
To run the full eval harness (propose/grade, all metrics) against the fine-tuned model
locally, merge + convert to GGUF and register it with Ollama:

```bash
# on a machine with llama.cpp:
python -m peft ...            # merge_and_unload, save merged HF model
python llama.cpp/convert_hf_to_gguf.py merged/ --outfile granite-ft.gguf
ollama create granite-ft -f Modelfile        # FROM ./granite-ft.gguf
# then the existing harness treats it as any model:
python -m grader eval --suite evals/private/contextual_heldout_100.jsonl \
  --policy policies/client_strategy.json --out-dir evals/results --fixture-transform off \
  --llm ollama --llm-model granite-ft --llm-strategy grade --grader-verdicts triage
```

## Why this is QLoRA on GPU, LoRA on CPU
4-bit quantization (bitsandbytes nf4) needs CUDA; on a CPU-only box the script falls back
to bf16/fp32 LoRA (no `--qlora`). The adapter format is identical either way, so the GPU
run and a local smoke produce interchangeable adapters.
