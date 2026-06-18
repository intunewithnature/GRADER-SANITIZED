#!/usr/bin/env bash
set -euo pipefail

find_ollama() {
  if [[ -n "${OLLAMA_BIN:-}" && -x "${OLLAMA_BIN}" ]]; then
    printf '%s\n' "${OLLAMA_BIN}"
    return 0
  fi

  for candidate in \
    "${HOME}/.local/ollama/bin/ollama" \
    "${HOME}/.local/bin/ollama" \
    "ollama"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

ollama_bin="$(find_ollama)" || {
  echo "Ollama binary not found. Install the official Linux archive under \$HOME/.local/ollama first." >&2
  exit 1
}

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-1}"

case "${OLLAMA_HOST}" in
  127.0.0.1:*|http://127.0.0.1:*) ;;
  *)
    echo "Refusing to start Ollama on non-localhost OLLAMA_HOST=${OLLAMA_HOST}" >&2
    exit 2
    ;;
esac

exec "${ollama_bin}" serve
