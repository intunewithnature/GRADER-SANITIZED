from __future__ import annotations

import getpass
import json
import uuid
from pathlib import Path

from .artifact_store import ArtifactStore
from .audit import write_audit_event
from .corpus import make_corpus
from .eval_runner import run_eval_suite
from .policy import load_policy
from .transform import blind_text
from .verify import verify_safe_text


def project_root(base_dir: str | Path | None = None) -> Path:
    return Path(base_dir) if base_dir is not None else Path.cwd()


def resolve_policy(policy_arg: str, base_dir: str | Path | None = None):
    root = project_root(base_dir)
    path = Path(policy_arg)
    if path.exists():
        return load_policy(path), path
    if not policy_arg.endswith(".json"):
        candidate = root / "policies" / f"{policy_arg}.json"
        if candidate.exists():
            return load_policy(candidate), candidate
    candidate = root / "policies" / policy_arg
    if candidate.exists():
        return load_policy(candidate), candidate
    raise FileNotFoundError(f"Policy not found: {policy_arg}")


def blind_file(in_path: str | Path, policy_arg: str, out_path: str | Path, *, base_dir: str | Path | None = None) -> dict:
    root = project_root(base_dir)
    policy, policy_path = resolve_policy(policy_arg, root)
    raw = Path(in_path).read_text(encoding="utf-8")
    result = blind_text(raw, policy, base_dir=root)
    safe_artifact_path = None
    if result.verification_status == "passed":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(result.safe_text, encoding="utf-8")
        ArtifactStore(root).save_artifact(result)
        safe_artifact_path = str(out_path)
    write_audit_event(
        base_dir=root,
        event_type="blind",
        artifact_id=result.artifact_id,
        policy=policy,
        blind_result=result,
    )
    return {
        "artifact_id": result.artifact_id,
        "policy": policy.name,
        "policy_path": str(policy_path),
        "verification_status": result.verification_status,
        "hidden_field_types": result.hidden_field_types,
        "safe_artifact_path": safe_artifact_path,
    }


def read_safe(artifact_id: str, *, base_dir: str | Path | None = None) -> str:
    artifact = ArtifactStore(project_root(base_dir)).get_artifact(artifact_id)
    if not artifact:
        raise KeyError(f"Unknown artifact_id: {artifact_id}")
    if artifact.verification_status != "passed":
        raise PermissionError("Artifact failed verification and cannot be read as safe text")
    return artifact.safe_text


def submit_judgment(
    artifact_id: str,
    judgment_text: str,
    *,
    base_dir: str | Path | None = None,
) -> str:
    root = project_root(base_dir)
    artifact = ArtifactStore(root).get_artifact(artifact_id)
    if not artifact:
        raise KeyError(f"Unknown artifact_id: {artifact_id}")
    if artifact.verification_status != "passed":
        raise PermissionError("Cannot submit judgment for an artifact that failed verification")
    judgment_id = f"judg_{uuid.uuid4().hex[:16]}"
    ArtifactStore(root).save_judgment(artifact_id, judgment_id, judgment_text)
    write_audit_event(base_dir=root, event_type="submit_judgment", artifact_id=artifact_id)
    return judgment_id


def reveal_report(
    artifact_id: str,
    judgment_id: str,
    *,
    base_dir: str | Path | None = None,
    local_full: bool = False,
    print_secrets: bool = False,
) -> dict:
    root = project_root(base_dir)
    store = ArtifactStore(root)
    artifact = store.get_artifact(artifact_id)
    judgment = store.get_judgment(judgment_id)
    if not artifact:
        raise KeyError(f"Unknown artifact_id: {artifact_id}")
    if not judgment:
        raise KeyError(f"Unknown judgment_id: {judgment_id}")
    if judgment[0] != artifact_id:
        raise ValueError("Judgment does not belong to artifact")
    from .reveal_store import RevealStore

    reveal_store = RevealStore(root)
    entries = reveal_store.list_entries(artifact_id)
    report = {
        "artifact_id": artifact_id,
        "judgment_id": judgment_id,
        "mode": "sanitized_comparison_report",
        "placeholder_count": len(entries),
        "field_types": sorted({entry.field_type for entry in entries}),
        "safe_judgment_chars": len(judgment[1]),
        "comparison": "Judgment stored locally; default report withholds reveal values.",
    }
    if local_full:
        report["mode"] = "local_full_no_secrets" if not print_secrets else "local_full_with_secrets"
        report["placeholder_hashes"] = [
            {"field_type": entry.field_type, "placeholder": entry.placeholder, "original_sha256": entry.original_sha256}
            for entry in entries
        ]
        if print_secrets:
            report["revealed_judgment"] = reveal_store.replace_placeholders(artifact_id, judgment[1])
    write_audit_event(base_dir=root, event_type="reveal_sanitized", artifact_id=artifact_id)
    return report


def run_make_corpus(n: int, out: str | Path, seed: int) -> dict:
    make_corpus(n, out, seed)
    return {"cases_written": n, "out": str(out), "seed": seed}


def run_eval(
    suite: str | Path,
    policy_arg: str,
    out_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
    llm_mode: str = "none",
    llm_model: str | None = None,
    llm_max_cases: int | None = None,
    llm_timeout_seconds: int = 30,
    llm_required: bool = False,
    llm_prompt_profile: str = "strict",
    llm_strategy: str = "propose",
    grader_verdicts: str = "binary",
    fixture_transform: bool = True,
    judge_usefulness: str = "none",
    judge_max_cases: int | None = None,
) -> dict:
    root = project_root(base_dir)
    policy, _ = resolve_policy(policy_arg, root)
    return run_eval_suite(
        suite,
        policy,
        out_dir,
        base_dir=root,
        llm_mode=llm_mode,
        llm_model=llm_model,
        llm_max_cases=llm_max_cases,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_required=llm_required,
        llm_prompt_profile=llm_prompt_profile,
        llm_strategy=llm_strategy,
        grader_verdicts=grader_verdicts,
        fixture_transform=fixture_transform,
        judge_usefulness=judge_usefulness,
        judge_max_cases=judge_max_cases,
    )


def confirm_local_full() -> bool:
    prompt = "Type LOCAL REVEAL to continue with local full reveal detail: "
    return getpass.getpass(prompt) == "LOCAL REVEAL"


def load_audit_events(*, base_dir: str | Path | None = None, limit: int = 50) -> list[dict]:
    path = project_root(base_dir) / "logs" / "audit.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
