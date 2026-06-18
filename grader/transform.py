from __future__ import annotations

import uuid
from pathlib import Path

from .detect import detect_spans
from .generalize import generalize
from .models import BlindAttribution, BlindResult, DetectedSpan, Policy
from .pseudonymize import pseudonymize
from .reveal_store import RevealStore
from .util import now_iso, sha256_text
from .verify import verify_safe_text


SENSITIVITY = {
    "api_key": 100,
    "secret_token": 95,
    "email": 90,
    "phone": 88,
    "account_id": 86,
    "file_path": 84,
    "person_name": 80,
    "business_name": 78,
    "model_name": 77,
    "url": 76,
    "private_note": 70,
    "business_relationship": 68,
    "client_status": 66,
    "budget": 64,
    "exact_price": 64,
    "location": 60,
    "date": 50,
}


def blind_text(
    text: str,
    policy: Policy,
    *,
    artifact_id: str | None = None,
    scope_id: str | None = None,
    base_dir: str | Path | None = None,
    fixture_canaries: list[str] | None = None,
    fixture_field_types: list[str] | None = None,
    llm_spans: list[DetectedSpan] | None = None,
) -> BlindResult:
    artifact_id = artifact_id or f"art_{uuid.uuid4().hex[:16]}"
    scope_id = scope_id or artifact_id
    store = RevealStore(base_dir)
    store.delete_artifact(artifact_id)
    spans = detect_spans(text, policy, fixture_canaries, fixture_field_types)
    if llm_spans:
        spans.extend(llm_spans)
    threshold_spans = _threshold_spans(spans, policy)
    selected = resolve_overlaps(threshold_spans, text)
    attribution = _attribute_spans(selected, threshold_spans)

    safe = text
    hidden: set[str] = set()
    for span in sorted(selected, key=lambda item: item.start, reverse=True):
        field_policy = policy.fields.get(span.field_type)
        if not field_policy or field_policy.op == "ignore":
            continue
        hidden.add(span.field_type)
        if field_policy.op == "redact":
            replacement = f"[REDACTED_{span.field_type.upper()}]"
        elif field_policy.op == "pseudonymize":
            replacement = pseudonymize(
                store,
                artifact_id,
                scope_id,
                span.field_type,
                span.text,
                field_policy.format,
            )
        elif field_policy.op == "generalize":
            replacement = generalize(span.field_type, span.text, field_policy.style)
        else:
            replacement = span.text
        safe = safe[: span.start] + replacement + safe[span.end :]

    verification = verify_safe_text(
        safe,
        policy,
        artifact_id=artifact_id,
        base_dir=base_dir,
        canaries=fixture_canaries or [],
        canary_field_types=fixture_field_types or [],
    )
    return BlindResult(
        artifact_id=artifact_id,
        safe_text=safe,
        hidden_field_types=sorted(hidden),
        canary_count=len(fixture_canaries or []),
        verification_status="passed" if verification.passed else "failed",
        verification_summary=verification.summary(),
        input_sha256=sha256_text(text),
        safe_sha256=sha256_text(safe),
        created_at=now_iso(),
        attribution=attribution,
    )


def resolve_overlaps(spans: list[DetectedSpan], text: str | None = None) -> list[DetectedSpan]:
    """Merge overlapping spans into clusters so coverage only ever GROWS.

    Previously this picked the single highest-ranked span per overlap and dropped
    the rest. That let a high-confidence narrow span (e.g. an LLM proposal) evict a
    wider detector span that fully covered a canary — silently *un-redacting* it
    ("displacement"). Instead we now redact the UNION of every overlapping removal
    span; the highest-ranked span in a cluster only decides HOW the merged range is
    replaced (its field_type/op), never whether the rest of the range survives.
    """
    candidates = [span for span in spans if span.start < span.end]
    if not candidates:
        return []

    def rank(span: DetectedSpan) -> tuple[float, int, int]:
        return (span.confidence, span.end - span.start, SENSITIVITY.get(span.field_type, 0))

    ordered = sorted(candidates, key=lambda span: (span.start, span.end))
    clusters: list[list[DetectedSpan]] = [[ordered[0]]]
    cluster_end = ordered[0].end
    for span in ordered[1:]:
        if span.start < cluster_end:  # overlaps the current cluster
            clusters[-1].append(span)
            cluster_end = max(cluster_end, span.end)
        else:
            clusters.append([span])
            cluster_end = span.end

    merged: list[DetectedSpan] = []
    for cluster in clusters:
        start = min(span.start for span in cluster)
        end = max(span.end for span in cluster)
        winner = max(cluster, key=rank)
        if text is not None:
            merged_text = text[start:end]
        elif winner.start == start and winner.end == end:
            merged_text = winner.text
        else:
            merged_text = winner.text  # best-effort when source text is unavailable
        merged.append(
            DetectedSpan(start, end, merged_text, winner.field_type, winner.detector, winner.confidence, winner.reason)
        )
    return sorted(merged, key=lambda span: span.start)


def _threshold_spans(spans: list[DetectedSpan], policy: Policy) -> list[DetectedSpan]:
    selected = []
    for span in spans:
        field_policy = policy.fields.get(span.field_type)
        if not field_policy:
            continue
        threshold = field_policy.confidence_threshold if field_policy.confidence_threshold is not None else 0.0
        if span.confidence >= threshold:
            selected.append(span)
    return selected


def _attribute_spans(selected: list[DetectedSpan], all_spans: list[DetectedSpan]) -> BlindAttribution:
    fixture = 0
    detector = 0
    llm = 0
    combined = 0
    llm_accepted: set[tuple[int, int, str]] = set()
    llm_field_types: set[str] = set()
    for span in selected:
        sources = {_source_category(other) for other in all_spans if _overlaps(span, other)}
        source = _source_category(span)
        if source == "fixture":
            fixture += 1
        elif source == "llm":
            llm += 1
        else:
            detector += 1
        if len(sources) > 1:
            combined += 1
        if "llm" in sources:
            llm_accepted.add((span.start, span.end, span.field_type))
            llm_field_types.add(span.field_type)
    return BlindAttribution(
        removed_by_fixture_canary=fixture,
        removed_by_detector=detector,
        removed_by_llm=llm,
        removed_by_combined=combined,
        llm_unique_spans_accepted=len(llm_accepted),
        llm_unique_span_field_types=sorted(llm_field_types),
    )


def _source_category(span: DetectedSpan) -> str:
    if span.detector == "fixture":
        return "fixture"
    if span.detector == "llm":
        return "llm"
    return "detector"


def _overlaps(a: DetectedSpan, b: DetectedSpan) -> bool:
    return not (a.end <= b.start or a.start >= b.end)
