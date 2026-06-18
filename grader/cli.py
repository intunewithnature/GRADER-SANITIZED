from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .commands import (
    blind_file,
    confirm_local_full,
    read_safe,
    reveal_report,
    run_eval,
    run_make_corpus,
    submit_judgment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grader")
    sub = parser.add_subparsers(dest="command", required=True)

    blind = sub.add_parser("blind")
    blind.add_argument("--in", dest="in_path", required=True)
    blind.add_argument("--policy", required=True)
    blind.add_argument("--out", required=True)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--suite", required=True)
    eval_cmd.add_argument("--policy", required=True)
    eval_cmd.add_argument("--out-dir", required=True)
    eval_cmd.add_argument("--llm", choices=["none", "ollama", "remote"], default="none")
    eval_cmd.add_argument("--llm-model")
    eval_cmd.add_argument("--llm-max-cases", type=int)
    eval_cmd.add_argument("--llm-timeout-seconds", type=int, default=30)
    eval_cmd.add_argument("--llm-required", action="store_true")
    eval_cmd.add_argument("--llm-prompt-profile", choices=["strict", "contextual", "coreference"], default="strict")
    eval_cmd.add_argument("--llm-strategy", choices=["propose", "grade"], default="propose")
    eval_cmd.add_argument("--grader-verdicts", choices=["binary", "triage"], default="binary")
    eval_cmd.add_argument("--fixture-transform", choices=["on", "off"], default="on")
    eval_cmd.add_argument("--judge-usefulness", choices=["none", "local", "remote"], default="none")
    eval_cmd.add_argument("--judge-max-cases", type=int)

    corpus = sub.add_parser("make-corpus")
    corpus.add_argument("--n", type=int, required=True)
    corpus.add_argument("--out", required=True)
    corpus.add_argument("--seed", type=int, default=42)

    read = sub.add_parser("read-safe")
    read.add_argument("--artifact-id", required=True)

    judge = sub.add_parser("submit-judgment")
    judge.add_argument("--artifact-id", required=True)
    judge.add_argument("--judgment-file", required=True)

    reveal = sub.add_parser("reveal")
    reveal.add_argument("--artifact-id", required=True)
    reveal.add_argument("--judgment-id", required=True)
    reveal.add_argument("--local-full", action="store_true")
    reveal.add_argument("--i-understand-this-prints-secrets", action="store_true")

    sub.add_parser("console")

    args = parser.parse_args(argv)
    try:
        if args.command == "blind":
            print(json.dumps(blind_file(args.in_path, args.policy, args.out), indent=2, sort_keys=True))
        elif args.command == "eval":
            metrics = run_eval(
                args.suite,
                args.policy,
                args.out_dir,
                llm_mode=args.llm,
                llm_model=args.llm_model,
                llm_max_cases=args.llm_max_cases,
                llm_timeout_seconds=args.llm_timeout_seconds,
                llm_required=args.llm_required,
                llm_prompt_profile=args.llm_prompt_profile,
                llm_strategy=args.llm_strategy,
                grader_verdicts=args.grader_verdicts,
                fixture_transform=args.fixture_transform == "on",
                judge_usefulness=args.judge_usefulness,
                judge_max_cases=args.judge_max_cases,
            )
            print(json.dumps(_public_eval_metrics(metrics), indent=2, sort_keys=True))
        elif args.command == "make-corpus":
            print(json.dumps(run_make_corpus(args.n, args.out, args.seed), indent=2, sort_keys=True))
        elif args.command == "read-safe":
            print(read_safe(args.artifact_id))
        elif args.command == "submit-judgment":
            text = Path(args.judgment_file).read_text(encoding="utf-8")
            print(json.dumps({"judgment_id": submit_judgment(args.artifact_id, text)}, indent=2, sort_keys=True))
        elif args.command == "reveal":
            if args.local_full and not confirm_local_full():
                print(json.dumps({"error": "local full reveal cancelled"}, indent=2), file=sys.stderr)
                return 2
            report = reveal_report(
                args.artifact_id,
                args.judgment_id,
                local_full=args.local_full,
                print_secrets=args.i_understand_this_prints_secrets,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "console":
            from .console import run_console

            run_console()
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


def _public_eval_metrics(metrics: dict) -> dict:
    keys = [
        "cases_total",
        "cases_passed",
        "hard_leaks",
        "near_leaks",
        "numeric_leaks",
        "reveal_map_leaks",
        "json_or_process_failures",
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "avg_safe_chars",
        "avg_raw_chars",
        "over_redaction_warnings",
        "coreference_errors",
        "llm_mode",
        "llm_model",
        "llm_prompt_profile",
        "judge_usefulness",
        "fixture_transform_enabled",
        "llm_json_failures",
        "llm_timeouts",
        "llm_transport_failures",
        "llm_spans_proposed",
        "llm_spans_accepted",
        "llm_spans_rejected",
        "llm_added_field_types",
        "p50_llm_latency_ms",
        "p95_llm_latency_ms",
        "removed_by_fixture_canary",
        "removed_by_detector",
        "removed_by_llm",
        "removed_by_combined",
        "llm_unique_spans_accepted",
        "llm_unique_span_field_types",
        "cases_where_llm_changed_output",
        "cases_where_llm_prevented_failure",
        "cases_where_llm_caused_failure",
        "cases_where_llm_overredacted",
        "cases_where_fixture_would_have_saved",
        "cases_failed_without_fixture_transform",
        "usefulness_cases_judged",
        "usefulness_avg_score",
        "usefulness_p50",
        "usefulness_p10",
        "usefulness_p50_score",
        "usefulness_p10_score",
        "overredaction_count",
        "missing_semantics_count",
        "usefulness_overredaction_count",
        "usefulness_missing_semantics_count",
        "usefulness_failures",
        "aggregate_path",
        "per_case_path",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


if __name__ == "__main__":
    raise SystemExit(main())
