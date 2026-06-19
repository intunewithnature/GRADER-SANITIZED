"""Build span-extraction SFT data from the synthetic train corpus.

Each example teaches the proposer to emit the canonical span JSON the verifier/parser
expect, for the SAME prompt used at eval time (grader.llm_pass.STRICT_SPAN_V1_PROMPT).
Targets are built from each case's canaries + expected_field_types (only canaries that
appear verbatim in the note, deduped). Output: training/data/{train,val}.jsonl as
{"input": <prompt+note>, "target": <spans json>}.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grader.llm_pass import STRICT_SPAN_V1_PROMPT  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "evals/private/synthetic_train_1000.jsonl"
OUT = Path(__file__).resolve().parent / "data"
VAL_N = 50


def target_json(case: dict) -> str:
    raw = case["raw"]
    cans = case.get("canaries", [])
    fts = case.get("expected_field_types", [])
    seen = set()
    spans = []
    for i, can in enumerate(cans):
        if not can or can in seen or can not in raw:
            continue
        seen.add(can)
        ft = fts[i] if i < len(fts) else "private_note"
        spans.append({"text": can, "field_type": ft, "reason": f"sensitive {ft.replace('_', ' ')}", "confidence": 0.9})
    # compact, deterministic JSON (what we want the model to emit)
    return json.dumps({"spans": spans}, ensure_ascii=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    extra = Path(__file__).resolve().parent / "expansion" / "train_new.jsonl"
    if extra.exists():
        rows += [json.loads(l) for l in extra.open() if l.strip()]
        print(f"merged expansion/train_new.jsonl -> {len(rows)} total cases")
    examples = []
    for case in rows:
        if "raw" not in case:
            continue
        examples.append({"input": STRICT_SPAN_V1_PROMPT + case["raw"], "target": target_json(case)})
    val = examples[-VAL_N:]
    train = examples[:-VAL_N]
    for name, data in [("train", train), ("val", val)]:
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for ex in data:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # stats
    import statistics
    spans_per = [len(json.loads(e["target"])["spans"]) for e in examples]
    empty = sum(1 for n in spans_per if n == 0)
    print(f"wrote {len(train)} train + {len(val)} val to {OUT}")
    print(f"avg spans/example: {statistics.mean(spans_per):.2f}  empty-target examples: {empty}")


if __name__ == "__main__":
    main()
