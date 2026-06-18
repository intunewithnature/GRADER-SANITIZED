from __future__ import annotations

import json
import shlex
from pathlib import Path

from .artifact_store import ArtifactStore
from .commands import (
    blind_file,
    load_audit_events,
    read_safe,
    resolve_policy,
    reveal_report,
    run_eval,
    submit_judgment,
)
from .llm_client import OllamaClient
from .policy import policy_to_dict


MODE_SAFE = "SAFE VIEW"
MODE_REVEAL = "REVEAL MODE"
MODE_EVAL = "EVAL MODE"
MODE_CHAT = "LOCAL LLM CHAT"


def run_console() -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        print("Textual is not installed. For the richer terminal UI, run: pip install textual")
        repl()
        return
    _run_textual_console()


def _run_textual_console() -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Static
    except ImportError:
        repl()
        return

    class AcmeConsoleApp(App):
        CSS = """
        Screen { layout: vertical; }
        #body { height: 1fr; }
        #left { width: 28; border: solid gray; }
        #main { width: 1fr; border: solid green; }
        #events { width: 36; border: solid gray; }
        #cmd { dock: bottom; }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                yield Static(_left_panel(), id="left")
                yield Static(f"{MODE_SAFE}\n\nacme-console ready. Type /help.", id="main")
                yield Static(_audit_panel(), id="events")
            yield Input(placeholder="acme> /eval evals/private/synthetic_smoke.jsonl --policy client_strategy", id="cmd")
            yield Footer()

        def on_input_submitted(self, event) -> None:
            output = handle_console_command(event.value)
            self.query_one("#main", Static).update(output)
            self.query_one("#events", Static).update(_audit_panel())
            event.input.value = ""

    AcmeConsoleApp().run()


def repl() -> None:
    print(f"{MODE_SAFE} acme-console. Type /help or /quit.")
    while True:
        try:
            line = input("acme> ")
        except EOFError:
            print()
            return
        if not line.strip():
            continue
        if line.strip() in {"/quit", "/exit"}:
            return
        print(handle_console_command(line))


def handle_console_command(line: str) -> str:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        return f"{MODE_SAFE}\nParse error: {exc}"
    if not parts:
        return MODE_SAFE
    cmd, args = parts[0], parts[1:]
    try:
        if cmd == "/help":
            return _help()
        if cmd == "/artifacts":
            return f"{MODE_SAFE}\n" + _left_panel()
        if cmd == "/blind":
            return _cmd_blind(args)
        if cmd == "/read":
            if len(args) != 1:
                return f"{MODE_SAFE}\nUsage: /read <artifact_id>"
            return f"{MODE_SAFE}\n{read_safe(args[0])}"
        if cmd == "/eval":
            return _cmd_eval(args)
        if cmd == "/judge":
            return _cmd_judge(args)
        if cmd == "/reveal":
            return _cmd_reveal(args)
        if cmd == "/policies":
            return f"{MODE_SAFE}\n" + "\n".join(sorted(path.stem for path in Path("policies").glob("*.json")))
        if cmd == "/policy":
            if len(args) != 1:
                return f"{MODE_SAFE}\nUsage: /policy <name>"
            policy, _ = resolve_policy(args[0])
            return f"{MODE_SAFE}\n" + json.dumps(policy_to_dict(policy), indent=2, sort_keys=True)
        if cmd == "/audit":
            return f"{MODE_SAFE}\n" + json.dumps(load_audit_events(limit=20), indent=2, sort_keys=True)
        if cmd == "/chat":
            return _cmd_chat(args)
        return f"{MODE_SAFE}\nUnknown command. Type /help."
    except Exception as exc:
        return f"{MODE_SAFE}\nLocal error: {type(exc).__name__}: {exc}"


def _cmd_blind(args: list[str]) -> str:
    if "--policy" not in args or not args:
        return f"{MODE_SAFE}\nUsage: /blind <path> --policy <policy_name>"
    path = args[0]
    policy = args[args.index("--policy") + 1]
    out = Path("artifacts") / f"{Path(path).stem}.safe.md"
    result = blind_file(path, policy, out)
    return f"{MODE_SAFE}\n" + json.dumps(result, indent=2, sort_keys=True)


def _cmd_eval(args: list[str]) -> str:
    if "--policy" not in args or not args:
        return f"{MODE_EVAL}\nUsage: /eval <suite_path> --policy <policy_name>"
    suite = args[0]
    policy = args[args.index("--policy") + 1]
    metrics = run_eval(suite, policy, "evals/results")
    public = {
        key: metrics[key]
        for key in [
            "cases_total",
            "cases_passed",
            "hard_leaks",
            "near_leaks",
            "numeric_leaks",
            "reveal_map_leaks",
            "p50_latency_ms",
            "p95_latency_ms",
        ]
        if key in metrics
    }
    return f"{MODE_EVAL}\n" + json.dumps(public, indent=2, sort_keys=True)


def _cmd_judge(args: list[str]) -> str:
    if len(args) != 1:
        return f"{MODE_SAFE}\nUsage: /judge <artifact_id>"
    artifact_id = args[0]
    print("Paste judgment based only on safe_text. End with a single line containing /end.")
    lines: list[str] = []
    while True:
        line = input("judgment> ")
        if line == "/end":
            break
        lines.append(line)
    judgment_id = submit_judgment(artifact_id, "\n".join(lines))
    return f"{MODE_SAFE}\n" + json.dumps({"judgment_id": judgment_id}, indent=2)


def _cmd_reveal(args: list[str]) -> str:
    if len(args) < 2:
        return f"{MODE_REVEAL}\nUsage: /reveal <artifact_id> <judgment_id> [--local-full --i-understand-this-prints-secrets]"
    local_full = "--local-full" in args
    print_secrets = "--i-understand-this-prints-secrets" in args
    if print_secrets and not local_full:
        return f"{MODE_REVEAL}\nSecret printing requires --local-full."
    if local_full:
        confirmation = input("Type LOCAL REVEAL to continue: ")
        if confirmation != "LOCAL REVEAL":
            return f"{MODE_REVEAL}\nCancelled."
    report = reveal_report(args[0], args[1], local_full=local_full, print_secrets=print_secrets)
    return f"{MODE_REVEAL}\n" + json.dumps(report, indent=2, sort_keys=True)


def _cmd_chat(args: list[str]) -> str:
    prompt = " ".join(args).strip()
    if not prompt:
        print(f"{MODE_CHAT}. Do not paste secrets unless you intend the local model process to see them.")
        prompt = input("local-llm> ")
    response = OllamaClient().chat(prompt)
    if not response.ok:
        return f"{MODE_CHAT}\n{response.error}"
    return f"{MODE_CHAT}\n{response.text}"


def _left_panel() -> str:
    artifacts = ArtifactStore().list_artifacts(limit=12)
    lines = ["Artifacts:"]
    lines.extend(f"{item.artifact_id} {item.verification_status}" for item in artifacts)
    lines.append("\nPolicies:")
    lines.extend(sorted(path.stem for path in Path("policies").glob("*.json")))
    lines.append("\nEval suites:")
    lines.extend(sorted(str(path) for path in Path("evals/private").glob("*.jsonl")))
    return "\n".join(lines)


def _audit_panel() -> str:
    events = load_audit_events(limit=8)
    if not events:
        return "Sanitized audit/events/metrics:\n(no events)"
    return "Sanitized audit/events/metrics:\n" + "\n".join(
        f"{event.get('timestamp')} {event.get('event_type')} {event.get('artifact_id')} passed={event.get('verification_passed')}"
        for event in events
    )


def _help() -> str:
    return """SAFE VIEW
/blind <path> --policy <policy_name>
/read <artifact_id>
/eval <suite_path> --policy <policy_name>
/judge <artifact_id>
/reveal <artifact_id> <judgment_id>
/reveal <artifact_id> <judgment_id> --local-full --i-understand-this-prints-secrets
/policies
/policy <name>
/audit
/chat [message]
/artifacts
/quit"""
