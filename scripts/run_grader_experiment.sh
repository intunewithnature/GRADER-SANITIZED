#!/usr/bin/env bash
# Grader A/B + bug-fix measurement, fixture-off, SEQUENTIAL (CPU; never parallel).
# All localhost-only via OllamaClient. Writes per-lane markers + a final DONE.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
M=granite3.3:2b; TO=60; POL=policies/client_strategy.json; OUT=evals/results

run () { # suite extra-args label
  echo ">>> $(date +%H:%M:%S)  $3"
  python3 -m grader eval --suite "evals/private/$1" --policy "$POL" --out-dir "$OUT" \
    --fixture-transform off --llm ollama --llm-model "$M" --llm-max-cases 100 --llm-timeout-seconds "$TO" \
    $2 >/dev/null 2>&1
  echo "    done $(date +%H:%M:%S)"
}

# held-out (the fair test): bug-fixed propose, then the grader A/B
run contextual_heldout_100.jsonl ""                                          "held-out: propose (post union-merge fix)"
run contextual_heldout_100.jsonl "--llm-strategy grade --grader-verdicts binary"  "held-out: GRADE binary"
run contextual_heldout_100.jsonl "--llm-strategy grade --grader-verdicts triage"  "held-out: GRADE triage"
# dev probe with the grader, for the dev-vs-held-out generalization gap
run contextual_llm_probe_100.jsonl "--llm-strategy grade --grader-verdicts binary" "dev: GRADE binary"

touch "$OUT/.grader_exp_done"
echo ">>> GRADER EXPERIMENT COMPLETE $(date +%H:%M:%S)"
