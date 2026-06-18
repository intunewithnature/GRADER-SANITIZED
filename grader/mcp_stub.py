from __future__ import annotations

from pathlib import Path

from .commands import blind_file, read_safe, reveal_report, submit_judgment


def blind_artifact(file_path: str, policy_name: str, scope_id: str | None = None) -> dict:
    out = Path("artifacts") / f"{Path(file_path).stem}.safe.md"
    result = blind_file(file_path, policy_name, out)
    return {
        "artifact_id": result["artifact_id"],
        "hidden_field_types": result["hidden_field_types"],
        "canary_count": 0,
    }


def read_blind_artifact(artifact_id: str) -> str:
    return read_safe(artifact_id)


def submit_blind_judgment(artifact_id: str, judgment: str) -> str:
    return submit_judgment(artifact_id, judgment)


def reveal_artifact(artifact_id: str, judgment_id: str) -> dict:
    return reveal_report(artifact_id, judgment_id)
