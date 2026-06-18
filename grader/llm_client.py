from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    ok: bool
    text: str
    error: str | None = None
    timed_out: bool = False
    transport_failed: bool = False


class NoopLLMClient:
    provider = "none"

    def chat(self, message: str, *, model: str | None = None, timeout_seconds: int = 30) -> LLMResponse:
        return LLMResponse(False, "", "No LLM is configured.")


class OllamaClient:
    provider = "ollama"

    def __init__(self, endpoint: str = "http://127.0.0.1:11434/api/chat", model: str = "granite3.3:2b") -> None:
        self.endpoint = endpoint
        self.model = model

    def chat(self, message: str, *, model: str | None = None, timeout_seconds: int = 30) -> LLMResponse:
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            return LLMResponse(False, "", "Local Ollama request timed out.", timed_out=True)
        except (urllib.error.URLError, json.JSONDecodeError):
            return LLMResponse(False, "", "Local Ollama transport or JSON error.", transport_failed=True)
        content = data.get("message", {}).get("content", "")
        return LLMResponse(True, str(content))


class RemoteChatClient:
    provider = "remote"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GRADER_REMOTE_API_KEY")
        self.base_url = (base_url or os.environ.get("GRADER_REMOTE_BASE_URL") or "").strip()
        self.model = model or os.environ.get("GRADER_REMOTE_MODEL") or ""

    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def chat(self, message: str, *, model: str | None = None, timeout_seconds: int = 30) -> LLMResponse:
        if not self.configured():
            return LLMResponse(False, "", "Remote LLM env vars are not configured.", transport_failed=True)
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": message}],
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            return LLMResponse(False, "", "Remote LLM request timed out.", timed_out=True)
        except (urllib.error.URLError, json.JSONDecodeError):
            return LLMResponse(False, "", "Remote LLM transport or JSON error.", transport_failed=True)
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            return LLMResponse(True, str(content))
        content = data.get("message", {}).get("content", "")
        return LLMResponse(True, str(content))


def choose_ollama_model() -> str:
    preferred = ["granite3.3:2b", "granite3.2:2b", "qwen3:4b", "smollm3:3b"]
    candidates = [
        os.environ.get("OLLAMA_BIN", ""),
        os.path.expanduser("~/.local/ollama/bin/ollama"),
        os.path.expanduser("~/.local/bin/ollama"),
        "ollama",
    ]
    ollama_bin = next((candidate for candidate in candidates if candidate and (candidate == "ollama" or os.path.exists(candidate))), "ollama")
    try:
        proc = subprocess.run([ollama_bin, "list"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "granite3.3:2b"
    lines = proc.stdout.splitlines()[1:]
    installed = []
    for line in lines:
        parts = line.split()
        if parts:
            installed.append(parts[0])
    for model in preferred:
        if model in installed:
            return model
    return installed[0] if installed else "granite3.3:2b"
