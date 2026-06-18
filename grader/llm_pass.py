from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .detect import semantic_candidates
from .llm_client import LLMResponse, NoopLLMClient, OllamaClient, RemoteChatClient
from .models import DetectedSpan, Policy


ALLOWED_FIELD_TYPES = {
    "person_name",
    "business_name",
    "email",
    "phone",
    "url",
    "api_key",
    "secret_token",
    "exact_price",
    "budget",
    "file_path",
    "account_id",
    "location",
    "date",
    "client_status",
    "business_relationship",
    "private_note",
    "model_name",
}

STRICT_SPAN_V1_PROMPT = """You are Grader's local blinding-compiler judgment pass.
You only propose sensitive spans.
You do not approve egress.
The verifier decides.
Identify spans that reveal identity, private business context, client relationship, deal terms, credentials, private paths, account IDs, locations, dates, model/source identity, or sensitive notes.
Return JSON only.
Do not explain outside JSON.
Do not include a safe_to_send field.
Return exactly:
{"spans":[{"text":"...","field_type":"business_relationship","reason":"contextual client reference","confidence":0.82}]}
Text:
"""

CONTEXTUAL_SPAN_V2_PROMPT = """You are Grader's local contextual span proposal pass.
You only propose sensitive spans.
You do not approve egress. The verifier decides.
Identify exact text spans that reveal private identity, private business status, client relationship, deal terms, billing/account problems, private locations, dates, credentials, private paths, model/source identity, or sensitive notes.
Focus especially on indirect references such as role-only client references, pronouns with private business context, launch/billing blockers, private verbal agreements, competitor mentions, and contextual identifiers.
Return JSON only.
Do not explain outside JSON.
Do not include safe_to_send, approved, allow_egress, reveal maps, or placeholder mappings.
Every span text must be copied exactly from the input.
Return exactly:
{"spans":[{"text":"...","field_type":"business_relationship","reason":"contextual client reference","confidence":0.82}]}
Text:
"""

COREFERENCE_SPAN_V3_PROMPT = """You are Grader's local coreference-aware span proposal pass.
You only propose sensitive spans and never approve egress. The verifier decides.
Return JSON only with a spans list. Do not include safe_to_send, approved, allow_egress, reveal maps, or placeholder mappings.
Find exact text spans from the input that reveal private identity, client relationship, business status, deal terms, billing/account blockers, private notes, model/source identity, private paths, locations, dates, credentials, or account IDs.
Pay close attention to pronouns, nicknames, initials, role-only references, indirect client descriptions, and phrases that connect a person/account/org to private business context.
Every span text must be copied exactly from the input.
Return exactly:
{"spans":[{"text":"...","field_type":"business_relationship","reason":"coreference reveals private client context","confidence":0.82}]}
Text:
"""

PROMPT_PROFILES = {
    "strict": ("strict_span_v1", STRICT_SPAN_V1_PROMPT),
    "contextual": ("contextual_span_v2", CONTEXTUAL_SPAN_V2_PROMPT),
    "coreference": ("coreference_span_v3", COREFERENCE_SPAN_V3_PROMPT),
}

SPAN_PROMPT = STRICT_SPAN_V1_PROMPT


@dataclass
class SpanProposalStats:
    llm_json_failures: int = 0
    llm_timeouts: int = 0
    llm_transport_failures: int = 0
    llm_spans_proposed: int = 0
    llm_spans_accepted: int = 0
    llm_spans_rejected: int = 0
    llm_added_field_types: set[str] = field(default_factory=set)
    llm_latency_ms: float = 0.0


@dataclass
class SpanProposalResult:
    spans: list[DetectedSpan] = field(default_factory=list)
    stats: SpanProposalStats = field(default_factory=SpanProposalStats)


class LocalLLMJudgmentPass:
    provider = "base"

    def propose_spans_with_stats(self, text: str, policy: Policy, *, timeout_seconds: int = 30) -> SpanProposalResult:
        raise NotImplementedError

    def propose_spans(self, text: str, policy: Policy) -> list[DetectedSpan]:
        return self.propose_spans_with_stats(text, policy).spans


class NoopLLMJudgmentPass(LocalLLMJudgmentPass):
    provider = "none"

    def propose_spans_with_stats(self, text: str, policy: Policy, *, timeout_seconds: int = 30) -> SpanProposalResult:
        return SpanProposalResult()


def normalize_prompt_profile(profile: str | None) -> str:
    if profile in PROMPT_PROFILES:
        return profile or "strict"
    aliases = {
        "strict_span_v1": "strict",
        "contextual_span_v2": "contextual",
        "coreference_span_v3": "coreference",
        None: "strict",
    }
    return aliases.get(profile, "strict")


def prompt_profile_name(profile: str | None) -> str:
    key = normalize_prompt_profile(profile)
    return PROMPT_PROFILES[key][0]


def prompt_for_profile(profile: str | None) -> str:
    key = normalize_prompt_profile(profile)
    return PROMPT_PROFILES[key][1]


class ChatLLMJudgmentPass(LocalLLMJudgmentPass):
    def __init__(self, client, *, model: str | None = None, provider: str = "llm", prompt_profile: str = "strict") -> None:
        self.client = client
        self.model = model
        self.provider = provider
        self.prompt_profile = normalize_prompt_profile(prompt_profile)

    def propose_spans_with_stats(self, text: str, policy: Policy, *, timeout_seconds: int = 30) -> SpanProposalResult:
        started = time.perf_counter()
        response: LLMResponse = self.client.chat(
            prompt_for_profile(self.prompt_profile) + text,
            model=self.model,
            timeout_seconds=timeout_seconds,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = SpanProposalResult()
        result.stats.llm_latency_ms = latency_ms
        if response.timed_out:
            result.stats.llm_timeouts += 1
            return result
        if not response.ok:
            result.stats.llm_transport_failures += 1
            return result
        parse_result = parse_span_json(response.text, text)
        parse_result.stats.llm_latency_ms = latency_ms
        return parse_result


class OllamaLLMJudgmentPass(ChatLLMJudgmentPass):
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        model: str = "granite3.3:2b",
        prompt_profile: str = "strict",
    ) -> None:
        super().__init__(OllamaClient(endpoint=endpoint, model=model), model=model, provider="ollama", prompt_profile=prompt_profile)


class RemoteLLMJudgmentPass(ChatLLMJudgmentPass):
    def __init__(self, model: str | None = None, prompt_profile: str = "strict") -> None:
        super().__init__(RemoteChatClient(model=model), model=model, provider="remote", prompt_profile=prompt_profile)


def _grader_prompt(text: str, candidates: list[DetectedSpan], triage: bool) -> str:
    lines = [
        "You are GRADER's local span grader. You do NOT approve egress; the verifier decides.",
        "Below is a private work note, then a numbered list of candidate phrases taken from it.",
        "For EACH candidate, decide whether it must be HIDDEN before the note is sent to an outside AI —",
        "i.e. it reveals private client identity, a private business/working relationship, deal or",
        "billing/payment status, a private location/date, credentials, private paths, model/source",
        "identity, or a sensitive note.",
    ]
    if triage:
        lines.append('Answer "hide" if it must be hidden, "keep" if clearly harmless, or "unsure" if you cannot tell.')
    else:
        lines.append('Answer "hide" if it must be hidden, otherwise "keep".')
    lines += [
        "Answer for every number. Return JSON only, no prose:",
        '{"verdicts":[{"i":1,"v":"hide"},{"i":2,"v":"keep"}]}',
        "",
        "NOTE:",
        text,
        "",
        "CANDIDATES:",
    ]
    lines += [f"{i}. {c.text}" for i, c in enumerate(candidates, 1)]
    return "\n".join(lines)


def parse_grader_verdicts(text: str) -> dict[int, str] | None:
    obj = _load_first_json_object(text)
    if not isinstance(obj, dict):
        return None
    arr = obj.get("verdicts")
    if not isinstance(arr, list):
        return None
    out: dict[int, str] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        verdict = item.get("v")
        if isinstance(verdict, str):
            out[idx] = verdict.strip().lower()
    return out


class GraderLLMJudgmentPass(LocalLLMJudgmentPass):
    """Flip the model's job from span GENERATION to span CLASSIFICATION.

    The detector over-generates candidate phrases (high recall, low precision);
    the local model only votes hide/keep[/unsure] on each. Small models are far
    better at this than at free-form span discovery, and because candidates carry
    known offsets the model never has to copy text (no verbatim-match rejection).
    triage=True adds the "unsure" option, routed to hide (fail-safe).
    """

    provider = "grader"

    def __init__(self, client, *, model: str | None = None, triage: bool = False) -> None:
        self.client = client
        self.model = model
        self.triage = triage

    def propose_spans_with_stats(self, text: str, policy: Policy, *, timeout_seconds: int = 30) -> SpanProposalResult:
        result = SpanProposalResult()
        candidates = semantic_candidates(text, policy)
        if not candidates:
            return result  # nothing to grade -> no LLM call, zero latency
        started = time.perf_counter()
        response = self.client.chat(_grader_prompt(text, candidates, self.triage), model=self.model, timeout_seconds=timeout_seconds)
        result.stats.llm_latency_ms = (time.perf_counter() - started) * 1000
        result.stats.llm_spans_proposed = len(candidates)
        if response.timed_out:
            result.stats.llm_timeouts += 1
            return result
        if not response.ok:
            result.stats.llm_transport_failures += 1
            return result
        verdicts = parse_grader_verdicts(response.text)
        if verdicts is None:
            result.stats.llm_json_failures += 1
            result.stats.llm_spans_rejected = len(candidates)
            return result
        approve = {"hide", "unsure"} if self.triage else {"hide"}
        for i, cand in enumerate(candidates, 1):
            verdict = verdicts.get(i, "keep")  # missing verdict -> keep (measure pure model decisions)
            if verdict in approve:
                result.spans.append(
                    DetectedSpan(cand.start, cand.end, cand.text, cand.field_type, "llm", 0.9, f"grader:{verdict}")
                )
                result.stats.llm_spans_accepted += 1
                result.stats.llm_added_field_types.add(cand.field_type)
            else:
                result.stats.llm_spans_rejected += 1
        return result


class OllamaGraderJudgmentPass(GraderLLMJudgmentPass):
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        model: str = "granite3.3:2b",
        triage: bool = False,
    ) -> None:
        super().__init__(OllamaClient(endpoint=endpoint, model=model), model=model, triage=triage)


def parse_span_json(text: str, source_text: str) -> SpanProposalResult:
    result = SpanProposalResult()
    obj = _load_first_json_object(text)
    if not isinstance(obj, dict):
        result.stats.llm_json_failures += 1
        return result
    spans = obj.get("spans")
    if not isinstance(spans, list):
        result.stats.llm_json_failures += 1
        return result
    seen: set[tuple[int, int, str]] = set()
    for item in spans:
        result.stats.llm_spans_proposed += 1
        if not isinstance(item, dict):
            result.stats.llm_spans_rejected += 1
            continue
        span_text = item.get("text")
        field_type = item.get("field_type")
        if not isinstance(span_text, str) or not span_text.strip():
            result.stats.llm_spans_rejected += 1
            continue
        if field_type not in ALLOWED_FIELD_TYPES:
            result.stats.llm_spans_rejected += 1
            continue
        start = source_text.find(span_text)
        if start < 0:
            result.stats.llm_spans_rejected += 1
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        end = start + len(span_text)
        key = (start, end, field_type)
        if key in seen:
            result.stats.llm_spans_rejected += 1
            continue
        seen.add(key)
        result.spans.append(
            DetectedSpan(
                start=start,
                end=end,
                text=span_text,
                field_type=field_type,
                detector="llm",
                confidence=confidence,
                reason=str(item.get("reason", "llm proposed sensitive span"))[:160],
            )
        )
        result.stats.llm_spans_accepted += 1
        result.stats.llm_added_field_types.add(field_type)
    return result


def _load_first_json_object(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
