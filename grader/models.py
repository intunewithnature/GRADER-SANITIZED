from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Operation = Literal["redact", "pseudonymize", "generalize", "ignore"]
DetectorName = Literal["regex", "fixture", "llm", "manual", "heuristic"]


@dataclass(frozen=True)
class FieldPolicy:
    op: Operation
    format: str | None = None
    style: str | None = None
    confidence_threshold: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Policy:
    name: str
    scope: Literal["artifact", "session", "global"]
    fields: dict[str, FieldPolicy]


@dataclass(frozen=True)
class DetectedSpan:
    start: int
    end: int
    text: str
    field_type: str
    detector: DetectorName
    confidence: float
    reason: str


@dataclass(frozen=True)
class BlindAttribution:
    removed_by_fixture_canary: int = 0
    removed_by_detector: int = 0
    removed_by_llm: int = 0
    removed_by_combined: int = 0
    llm_unique_spans_accepted: int = 0
    llm_unique_span_field_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BlindResult:
    artifact_id: str
    safe_text: str
    hidden_field_types: list[str]
    canary_count: int
    verification_status: str
    verification_summary: str
    input_sha256: str
    safe_sha256: str
    created_at: str
    attribution: BlindAttribution = field(default_factory=BlindAttribution)


@dataclass(frozen=True)
class RevealMapEntry:
    artifact_id: str
    scope_id: str
    field_type: str
    placeholder: str
    original_value: str
    original_sha256: str
    created_at: str


@dataclass
class VerificationResult:
    passed: bool
    hard_leaks: int = 0
    near_leaks: int = 0
    numeric_leaks: int = 0
    reveal_map_leaks: int = 0
    coreference_errors: int = 0
    warnings: list[str] = field(default_factory=list)
    sanitized_failure_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"passed={self.passed}; hard={self.hard_leaks}; near={self.near_leaks}; "
            f"numeric={self.numeric_leaks}; reveal_map={self.reveal_map_leaks}; "
            f"coreference={self.coreference_errors}; warnings={len(self.warnings)}"
        )
