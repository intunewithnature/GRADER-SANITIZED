#!/usr/bin/env bash
# Run the detector+Granite fixture-off lanes SEQUENTIALLY (CPU inference; never
# run these concurrently or they thrash the single Ollama worker). Bind is
# localhost-only via OllamaClient. Writes a DONE marker when finished.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=granite3.3:2b
TO=60
run () { # suite max-cases
  echo ">>> $(date +%H:%M:%S)  $1  (max=$2)  detector+Granite, fixture-off"
  python3 -m grader eval --suite "evals/private/$1" \
    --policy policies/client_strategy.json --out-dir evals/results \
    --fixture-transform off --llm ollama --llm-model "$MODEL" \
    --llm-max-cases "$2" --llm-timeout-seconds "$TO" >/dev/null 2>&1
  echo "    done $(date +%H:%M:%S)"
}

# 1) held-out (mine) — KEY: Granite marginal value on hardest semantic categories
run contextual_heldout_100.jsonl 100
# 2) dev probe (post-strengthening) — for dev-vs-held-out Granite gap
run contextual_llm_probe_100.jsonl 100
# 3) independent held-out (syntactic+status), capped — Granite generalization
run holdout_250.jsonl 100

touch evals/results/.granite_comparisons_done
echo ">>> ALL GRANITE LANES COMPLETE $(date +%H:%M:%S)"
