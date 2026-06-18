from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field

from .llm_client import OllamaClient, RemoteChatClient
from .llm_pass import _load_first_json_object


USEFULNESS_PROMPT = """You judge semantic preservation for synthetic blinded text.
You only see safe_text and expected_semantics.
You are not a safety gate.
Return JSON only:
{"semantic_preservation_score":0.0,"missing_semantics":["..."],"overredaction":false,"notes":"safe explanation only"}
"""


@dataclass
class UsefulnessResult:
    ok: bool
    score: float = 0.0
    missing_count: int = 0
    overredaction: bool = False
    failure: bool = False


@dataclass
class UsefulnessMetrics:
    usefulness_cases_judged: int = 0
    usefulness_scores: list[float] = field(default_factory=list)
    overredaction_count: int = 0
    missing_semantics_count: int = 0
    usefulness_failures: int = 0

    def public(self) -> dict:
        scores = sorted(self.usefulness_scores)
        return {
            "usefulness_cases_judged": self.usefulness_cases_judged,
            "usefulness_avg_score": round(statistics.fmean(scores), 3) if scores else 0.0,
            "usefulness_p50": round(_percentile(scores, 50), 3),
            "usefulness_p10": round(_percentile(scores, 10), 3),
            "usefulness_p50_score": round(_percentile(scores, 50), 3),
            "usefulness_p10_score": round(_percentile(scores, 10), 3),
            "overredaction_count": self.overredaction_count,
            "missing_semantics_count": self.missing_semantics_count,
            "usefulness_overredaction_count": self.overredaction_count,
            "usefulness_missing_semantics_count": self.missing_semantics_count,
            "usefulness_failures": self.usefulness_failures,
        }


class UsefulnessJudge:
    def __init__(self, mode: str, *, model: str | None = None, timeout_seconds: int = 30) -> None:
        self.mode = mode
        self.model = model
        self.timeout_seconds = timeout_seconds
        if mode == "local":
            self.client = OllamaClient(model=model or "granite3.3:2b")
        elif mode == "remote":
            self.client = RemoteChatClient(model=model)
        else:
            self.client = None

    def judge(
        self,
        *,
        safe_text: str,
        expected_semantics: list[str],
        policy_name: str,
        case_id: str | None = None,
    ) -> UsefulnessResult:
        if not self.client:
            return UsefulnessResult(ok=False, failure=True)
        payload = {
            "case_id": case_id,
            "policy_name": policy_name,
            "safe_text": safe_text,
            "expected_semantics": expected_semantics,
        }
        response = self.client.chat(
            USEFULNESS_PROMPT + json.dumps(payload, sort_keys=True),
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )
        if not response.ok:
            return UsefulnessResult(ok=False, failure=True)
        data = _load_first_json_object(response.text)
        if not isinstance(data, dict):
            return UsefulnessResult(ok=False, failure=True)
        try:
            score = float(data.get("semantic_preservation_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = min(1.0, max(0.0, score))
        missing = data.get("missing_semantics", [])
        missing_count = len(missing) if isinstance(missing, list) else 0
        overredaction = bool(data.get("overredaction", False))
        return UsefulnessResult(ok=True, score=score, missing_count=missing_count, overredaction=overredaction)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * (percentile / 100)))
    return values[index]
