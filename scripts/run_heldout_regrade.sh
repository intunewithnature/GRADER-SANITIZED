#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
for v in binary triage; do
  python3 -m grader eval --suite evals/private/contextual_heldout_100.jsonl \
    --policy policies/client_strategy.json --out-dir evals/results --fixture-transform off \
    --llm ollama --llm-model granite3.3:2b --llm-strategy grade --grader-verdicts "$v" \
    --llm-max-cases 100 --llm-timeout-seconds 60 >/dev/null 2>&1
  echo "done heldout grade $v $(date +%H:%M:%S)"
done
touch evals/results/.heldout_regrade_done
