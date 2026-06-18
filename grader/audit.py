from __future__ import annotations

from pathlib import Path

from .models import BlindResult, Policy, VerificationResult
from .util import append_jsonl, now_iso


def write_audit_event(
    *,
    base_dir: str | Path | None,
    event_type: str,
    artifact_id: str,
    policy: Policy | None = None,
    blind_result: BlindResult | None = None,
    verification: VerificationResult | None = None,
    canary_count: int = 0,
) -> None:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    event = {
        "event_type": event_type,
        "artifact_id": artifact_id,
        "input_sha256": blind_result.input_sha256 if blind_result else None,
        "safe_sha256": blind_result.safe_sha256 if blind_result else None,
        "policy_name": policy.name if policy else None,
        "hidden_field_types": blind_result.hidden_field_types if blind_result else [],
        "canary_count": canary_count,
        "verification_passed": blind_result.verification_status == "passed" if blind_result else None,
        "hard_leak_count": _count_from_summary(blind_result.verification_summary, "hard") if blind_result else 0,
        "near_leak_count": _count_from_summary(blind_result.verification_summary, "near") if blind_result else 0,
        "numeric_leak_count": _count_from_summary(blind_result.verification_summary, "numeric") if blind_result else 0,
        "reveal_map_leak_count": _count_from_summary(blind_result.verification_summary, "reveal_map") if blind_result else 0,
        "timestamp": now_iso(),
    }
    if verification:
        event["verification_passed"] = verification.passed
        event["hard_leak_count"] = verification.hard_leaks
        event["near_leak_count"] = verification.near_leaks
        event["numeric_leak_count"] = verification.numeric_leaks
        event["reveal_map_leak_count"] = verification.reveal_map_leaks
    append_jsonl(root / "logs" / "audit.jsonl", event)


def _count_from_summary(summary: str, key: str) -> int:
    marker = f"{key}="
    if marker not in summary:
        return 0
    try:
        return int(summary.split(marker, 1)[1].split(";", 1)[0])
    except ValueError:
        return 0
