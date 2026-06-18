from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .artifact_store import ArtifactStore
from .audit import write_audit_event
from .llm_client import choose_ollama_model
from .models import Policy
from .llm_pass import (
    NoopLLMJudgmentPass,
    OllamaGraderJudgmentPass,
    OllamaLLMJudgmentPass,
    RemoteLLMJudgmentPass,
    prompt_profile_name,
)
from .reveal_store import RevealStore
from .transform import blind_text
from .usefulness import UsefulnessJudge, UsefulnessMetrics
from .verify import verify_safe_text


def run_eval_suite(
    suite_path: str | Path,
    policy: Policy,
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
    suite_path = Path(suite_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(base_dir)

    latencies: list[float] = []
    llm_latencies: list[float] = []
    safe_lengths: list[int] = []
    raw_lengths: list[int] = []
    resolved_prompt_profile = prompt_profile_name(llm_prompt_profile)
    output_stem = suite_path.stem + _mode_suffix(
        llm_mode, judge_usefulness, fixture_transform, resolved_prompt_profile, llm_strategy, grader_verdicts
    )
    per_case_path = out_dir / f"{output_stem}.results.jsonl"
    failures = 0
    llm_pass = _make_llm_pass(llm_mode, llm_model, llm_prompt_profile, llm_strategy, grader_verdicts)
    llm_disabled = False
    llm_cases_attempted = 0
    usefulness_judge = UsefulnessJudge(judge_usefulness, model=llm_model, timeout_seconds=llm_timeout_seconds) if judge_usefulness != "none" else None
    usefulness_disabled = False
    usefulness_cases_attempted = 0
    usefulness_metrics = UsefulnessMetrics()
    totals = {
        "cases_total": 0,
        "cases_passed": 0,
        "hard_leaks": 0,
        "near_leaks": 0,
        "numeric_leaks": 0,
        "reveal_map_leaks": 0,
        "json_or_process_failures": 0,
        "over_redaction_warnings": 0,
        "coreference_errors": 0,
        "llm_json_failures": 0,
        "llm_timeouts": 0,
        "llm_transport_failures": 0,
        "llm_spans_proposed": 0,
        "llm_spans_accepted": 0,
        "llm_spans_rejected": 0,
        "removed_by_fixture_canary": 0,
        "removed_by_detector": 0,
        "removed_by_llm": 0,
        "removed_by_combined": 0,
        "llm_unique_spans_accepted": 0,
        "cases_where_llm_changed_output": 0,
        "cases_where_llm_prevented_failure": 0,
        "cases_where_llm_caused_failure": 0,
        "cases_where_llm_overredacted": 0,
        "llm_mode": llm_mode,
        "llm_model": llm_model,
        "llm_prompt_profile": resolved_prompt_profile,
        "judge_usefulness": judge_usefulness,
        "fixture_transform_enabled": fixture_transform,
        "cases_where_fixture_would_have_saved": 0,
        "cases_failed_without_fixture_transform": 0,
    }
    llm_field_types: set[str] = set()
    llm_unique_span_field_types: set[str] = set()

    with suite_path.open("r", encoding="utf-8") as in_fh, per_case_path.open("w", encoding="utf-8") as out_fh:
        for line_no, line in enumerate(in_fh, 1):
            if not line.strip():
                continue
            totals["cases_total"] += 1
            try:
                case = json.loads(line)
                start = time.perf_counter()
                artifact_id = f"art_{_artifact_stem(_artifact_output_stem(output_stem, llm_max_cases))}_{case.get('case_id', line_no)}"
                store.delete_artifact(artifact_id)
                fixture_canaries_for_transform = case.get("canaries", []) if fixture_transform else None
                fixture_fields_for_transform = case.get("expected_field_types", []) if fixture_transform else None
                llm_spans = []
                llm_case_stats = {}
                deterministic_blind = None
                deterministic_verification = None
                fixture_saved_blind = None
                fixture_saved_verification = None
                use_llm_for_case = (
                    llm_mode != "none"
                    and not llm_disabled
                    and (llm_max_cases is None or llm_cases_attempted < llm_max_cases)
                )
                if use_llm_for_case:
                    llm_cases_attempted += 1
                    deterministic_id = f"{artifact_id}_det_compare"
                    deterministic_blind = blind_text(
                        case["raw"],
                        policy,
                        artifact_id=deterministic_id,
                        base_dir=base_dir,
                        fixture_canaries=fixture_canaries_for_transform,
                        fixture_field_types=fixture_fields_for_transform,
                    )
                    deterministic_verification = verify_safe_text(
                        deterministic_blind.safe_text,
                        policy,
                        artifact_id=deterministic_blind.artifact_id,
                        base_dir=base_dir,
                        canaries=case.get("canaries", []),
                        canary_field_types=case.get("expected_field_types", []),
                    )
                    RevealStore(base_dir).delete_artifact(deterministic_id)
                    proposal = llm_pass.propose_spans_with_stats(
                        case["raw"],
                        policy,
                        timeout_seconds=llm_timeout_seconds,
                    )
                    stats = proposal.stats
                    totals["llm_json_failures"] += stats.llm_json_failures
                    totals["llm_timeouts"] += stats.llm_timeouts
                    totals["llm_transport_failures"] += stats.llm_transport_failures
                    totals["llm_spans_proposed"] += stats.llm_spans_proposed
                    totals["llm_spans_accepted"] += stats.llm_spans_accepted
                    totals["llm_spans_rejected"] += stats.llm_spans_rejected
                    llm_field_types.update(stats.llm_added_field_types)
                    if stats.llm_latency_ms:
                        llm_latencies.append(stats.llm_latency_ms)
                    llm_case_stats = {
                        "llm_spans_proposed": stats.llm_spans_proposed,
                        "llm_spans_accepted": stats.llm_spans_accepted,
                        "llm_spans_rejected": stats.llm_spans_rejected,
                        "llm_json_failures": stats.llm_json_failures,
                        "llm_timeouts": stats.llm_timeouts,
                        "llm_transport_failures": stats.llm_transport_failures,
                    }
                    if stats.llm_timeouts or stats.llm_transport_failures:
                        if llm_required:
                            raise RuntimeError("required_llm_unavailable")
                        llm_disabled = True
                    else:
                        llm_spans = proposal.spans
                blind = blind_text(
                    case["raw"],
                    policy,
                    artifact_id=artifact_id,
                    base_dir=base_dir,
                    fixture_canaries=fixture_canaries_for_transform,
                    fixture_field_types=fixture_fields_for_transform,
                    llm_spans=llm_spans,
                )
                verification = verify_safe_text(
                    blind.safe_text,
                    policy,
                    artifact_id=blind.artifact_id,
                    base_dir=base_dir,
                    canaries=case.get("canaries", []),
                    canary_field_types=case.get("expected_field_types", []),
                )
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                safe_lengths.append(len(blind.safe_text))
                raw_lengths.append(len(case["raw"]))
                if verification.passed:
                    store.save_artifact(blind)
                else:
                    RevealStore(base_dir).delete_artifact(blind.artifact_id)
                    if not fixture_transform:
                        totals["cases_failed_without_fixture_transform"] += 1
                        fixture_saved_id = f"{artifact_id}_fixture_compare"
                        fixture_saved_blind = blind_text(
                            case["raw"],
                            policy,
                            artifact_id=fixture_saved_id,
                            base_dir=base_dir,
                            fixture_canaries=case.get("canaries", []),
                            fixture_field_types=case.get("expected_field_types", []),
                        )
                        fixture_saved_verification = verify_safe_text(
                            fixture_saved_blind.safe_text,
                            policy,
                            artifact_id=fixture_saved_blind.artifact_id,
                            base_dir=base_dir,
                            canaries=case.get("canaries", []),
                            canary_field_types=case.get("expected_field_types", []),
                        )
                        if fixture_saved_verification.passed:
                            totals["cases_where_fixture_would_have_saved"] += 1
                        RevealStore(base_dir).delete_artifact(fixture_saved_id)
                write_audit_event(
                    base_dir=base_dir,
                    event_type="eval_case",
                    artifact_id=blind.artifact_id,
                    policy=policy,
                    blind_result=blind,
                    verification=verification,
                    canary_count=len(case.get("canaries", [])),
                )
                if verification.passed:
                    totals["cases_passed"] += 1
                totals["hard_leaks"] += verification.hard_leaks
                totals["near_leaks"] += verification.near_leaks
                totals["numeric_leaks"] += verification.numeric_leaks
                totals["reveal_map_leaks"] += verification.reveal_map_leaks
                totals["coreference_errors"] += verification.coreference_errors
                totals["over_redaction_warnings"] += len([w for w in verification.warnings if "over" in w])
                totals["removed_by_fixture_canary"] += blind.attribution.removed_by_fixture_canary
                totals["removed_by_detector"] += blind.attribution.removed_by_detector
                totals["removed_by_llm"] += blind.attribution.removed_by_llm
                totals["removed_by_combined"] += blind.attribution.removed_by_combined
                totals["llm_unique_spans_accepted"] += blind.attribution.llm_unique_spans_accepted
                llm_unique_span_field_types.update(blind.attribution.llm_unique_span_field_types)
                if deterministic_blind and blind.safe_text != deterministic_blind.safe_text:
                    totals["cases_where_llm_changed_output"] += 1
                    deterministic_passed = deterministic_verification.passed if deterministic_verification else deterministic_blind.verification_status == "passed"
                    if not deterministic_passed and verification.passed:
                        totals["cases_where_llm_prevented_failure"] += 1
                    if deterministic_passed and not verification.passed:
                        totals["cases_where_llm_caused_failure"] += 1

                if (
                    usefulness_judge
                    and not usefulness_disabled
                    and verification.passed
                    and (judge_max_cases is None or usefulness_cases_attempted < judge_max_cases)
                ):
                    usefulness_cases_attempted += 1
                    judged = usefulness_judge.judge(
                        safe_text=blind.safe_text,
                        expected_semantics=case.get("expected_semantics", []),
                        policy_name=policy.name,
                        case_id=str(case.get("case_id", f"line_{line_no}")),
                    )
                    if judged.ok:
                        usefulness_metrics.usefulness_cases_judged += 1
                        usefulness_metrics.usefulness_scores.append(judged.score)
                        usefulness_metrics.missing_semantics_count += judged.missing_count
                        if judged.overredaction:
                            usefulness_metrics.overredaction_count += 1
                            totals["cases_where_llm_overredacted"] += 1
                    else:
                        usefulness_metrics.usefulness_failures += 1
                        usefulness_disabled = True

                out_fh.write(
                    json.dumps(
                        {
                            "case_id": case.get("case_id", f"line_{line_no}"),
                            "artifact_id": blind.artifact_id,
                            "verification_passed": verification.passed,
                            "verification_summary": verification.summary(),
                            "sanitized_failure_ids": verification.sanitized_failure_ids,
                            "hidden_field_types": blind.hidden_field_types,
                            "latency_ms": round(elapsed, 3),
                            "llm": llm_case_stats,
                            "fixture_transform_enabled": fixture_transform,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            except Exception as exc:
                failures += 1
                totals["json_or_process_failures"] += 1
                out_fh.write(
                    json.dumps(
                        {
                            "case_id": f"line_{line_no}",
                            "verification_passed": False,
                            "sanitized_failure_ids": [f"process_failure:{type(exc).__name__}"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    totals.update(
        {
            "p50_latency_ms": round(_percentile(latencies, 50), 3),
            "p95_latency_ms": round(_percentile(latencies, 95), 3),
            "max_latency_ms": round(max(latencies) if latencies else 0.0, 3),
            "avg_safe_chars": round(statistics.fmean(safe_lengths) if safe_lengths else 0.0, 3),
            "avg_raw_chars": round(statistics.fmean(raw_lengths) if raw_lengths else 0.0, 3),
            "p50_llm_latency_ms": round(_percentile(llm_latencies, 50), 3),
            "p95_llm_latency_ms": round(_percentile(llm_latencies, 95), 3),
            "llm_added_field_types": sorted(llm_field_types),
            "llm_unique_span_field_types": sorted(llm_unique_span_field_types),
        }
    )
    totals.update(usefulness_metrics.public())
    aggregate_path = out_dir / f"{output_stem}.aggregate.json"
    aggregate_path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return totals | {"per_case_path": str(per_case_path), "aggregate_path": str(aggregate_path), "process_failures": failures}


def _make_llm_pass(mode: str, model: str | None, prompt_profile: str, strategy: str = "propose", grader_verdicts: str = "binary"):
    if mode == "ollama":
        if strategy == "grade":
            return OllamaGraderJudgmentPass(model=model or choose_ollama_model(), triage=(grader_verdicts == "triage"))
        return OllamaLLMJudgmentPass(model=model or choose_ollama_model(), prompt_profile=prompt_profile)
    if mode == "remote":
        return RemoteLLMJudgmentPass(model=model, prompt_profile=prompt_profile)
    return NoopLLMJudgmentPass()


def _mode_suffix(
    llm_mode: str,
    judge_usefulness: str,
    fixture_transform: bool,
    prompt_profile: str,
    strategy: str = "propose",
    grader_verdicts: str = "binary",
) -> str:
    parts = []
    if not fixture_transform:
        parts.append("fixture_off")
    if llm_mode != "none":
        parts.append(f"llm_{llm_mode}")
        if strategy == "grade":
            parts.append("grade")
            parts.append(grader_verdicts)
        elif prompt_profile != "strict_span_v1":
            parts.append(prompt_profile)
    if judge_usefulness != "none":
        parts.append(f"judge_{judge_usefulness}")
    return "." + ".".join(parts) if parts else ""


def _artifact_stem(output_stem: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in output_stem).strip("_")


def _artifact_output_stem(output_stem: str, llm_max_cases: int | None) -> str:
    if llm_max_cases is None:
        return output_stem
    return f"{output_stem}.max{llm_max_cases}"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * (percentile / 100)))
    return ordered[index]
