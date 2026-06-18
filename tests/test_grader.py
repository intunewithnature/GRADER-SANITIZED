import unittest
from pathlib import Path

from grader.detect import semantic_candidates
from grader.llm_client import LLMResponse
from grader.llm_pass import GraderLLMJudgmentPass, parse_grader_verdicts
from grader.policy import load_policy

POLICY = load_policy(Path("policies") / "client_strategy.json")
NOTE = "Call the silent partner behind the account; they keep stalling the launch until billing clears."


class FakeClient:
    """Returns a fixed verdict string for every candidate index."""

    def __init__(self, verdict: str, n: int = 20) -> None:
        self.verdict = verdict
        self.n = n
        self.calls = 0

    def chat(self, message, *, model=None, timeout_seconds=30):
        self.calls += 1
        items = ",".join(f'{{"i":{i},"v":"{self.verdict}"}}' for i in range(1, self.n + 1))
        return LLMResponse(True, '{"verdicts":[' + items + "]}")


class GraderTests(unittest.TestCase):
    def test_candidates_cover_semantic_phrases(self):
        cands = semantic_candidates(NOTE, POLICY)
        self.assertGreaterEqual(len(cands), 2)
        joined = " | ".join(c.text for c in cands)
        self.assertIn("silent partner", joined)
        self.assertTrue(any("stalling" in c.text for c in cands))
        # candidates must carry valid offsets into the source text
        for c in cands:
            self.assertEqual(NOTE[c.start:c.end], c.text)

    def test_parse_verdicts(self):
        v = parse_grader_verdicts('{"verdicts":[{"i":1,"v":"hide"},{"i":2,"v":"keep"},{"i":3,"v":"UNSURE"}]}')
        self.assertEqual(v, {1: "hide", 2: "keep", 3: "unsure"})
        self.assertIsNone(parse_grader_verdicts("not json"))

    def test_binary_approves_only_hide(self):
        cands = semantic_candidates(NOTE, POLICY)
        gp = GraderLLMJudgmentPass(FakeClient("hide"), triage=False)
        res = gp.propose_spans_with_stats(NOTE, POLICY)
        self.assertEqual(res.stats.llm_spans_accepted, len(cands))
        gp_keep = GraderLLMJudgmentPass(FakeClient("keep"), triage=False)
        self.assertEqual(gp_keep.propose_spans_with_stats(NOTE, POLICY).stats.llm_spans_accepted, 0)

    def test_unsure_routed_to_hide_only_in_triage(self):
        cands = semantic_candidates(NOTE, POLICY)
        binary = GraderLLMJudgmentPass(FakeClient("unsure"), triage=False).propose_spans_with_stats(NOTE, POLICY)
        triage = GraderLLMJudgmentPass(FakeClient("unsure"), triage=True).propose_spans_with_stats(NOTE, POLICY)
        self.assertEqual(binary.stats.llm_spans_accepted, 0)          # "unsure" is not "hide"
        self.assertEqual(triage.stats.llm_spans_accepted, len(cands))  # unsure -> hide (fail-safe)

    def test_no_candidates_means_no_llm_call(self):
        client = FakeClient("hide")
        res = GraderLLMJudgmentPass(client).propose_spans_with_stats("nothing sensitive here at all", POLICY)
        # may or may not have candidates; if none, the client must not be called
        if res.stats.llm_spans_proposed == 0:
            self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
