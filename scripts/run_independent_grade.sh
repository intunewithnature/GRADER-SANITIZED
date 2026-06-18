#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
M=granite3.3:2b; S=evals/private/independent_150.jsonl; P=policies/client_strategy.json; O=evals/results
for v in binary triage; do
  echo ">>> $(date +%H:%M:%S) GRADE $v"
  python3 -m grader eval --suite "$S" --policy "$P" --out-dir "$O" --fixture-transform off \
    --llm ollama --llm-model "$M" --llm-strategy grade --grader-verdicts "$v" \
    --llm-max-cases 150 --llm-timeout-seconds 60 >/dev/null 2>&1
  echo "    done $(date +%H:%M:%S) GRADE $v"
done
touch "$O/.independent_grade_done"
echo ">>> INDEPENDENT GRADE COMPLETE $(date +%H:%M:%S)"
