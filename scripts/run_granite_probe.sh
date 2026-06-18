#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

model="${1:-granite3.3:2b}"
max_cases="${2:-25}"
timeout_seconds="${3:-60}"

python3 -m grader eval \
  --suite evals/private/contextual_llm_probe_100.jsonl \
  --policy policies/client_strategy.json \
  --out-dir evals/results \
  --llm ollama \
  --llm-model "${model}" \
  --llm-max-cases "${max_cases}" \
  --llm-timeout-seconds "${timeout_seconds}"
