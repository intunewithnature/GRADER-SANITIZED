import json
import tempfile
import unittest
from pathlib import Path

from grader.eval_runner import run_eval_suite
from grader.commands import read_safe
from grader.policy import load_policy


class EvalRunnerTests(unittest.TestCase):
    def test_eval_suite_aggregate_metrics(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.jsonl"
            case = {
                "case_id": "case_001",
                "raw": "Bob Calder at Smith Roofing has $75/day in Grand Rapids.",
                "canaries": ["Bob Calder", "Smith Roofing", "$75/day", "Grand Rapids"],
                "expected_field_types": ["person_name", "business_name", "budget", "location"],
                "expected_semantics": ["client context"],
                "notes": "synthetic",
            }
            suite.write_text(json.dumps(case) + "\n", encoding="utf-8")
            metrics = run_eval_suite(suite, policy, root / "results", base_dir=root)
            aggregate_exists = (root / "results" / "suite.aggregate.json").exists()
        self.assertEqual(metrics["cases_total"], 1)
        self.assertEqual(metrics["cases_passed"], 1)
        self.assertEqual(metrics["hard_leaks"], 0)
        self.assertEqual(metrics["reveal_map_leaks"], 0)
        self.assertIn("removed_by_fixture_canary", metrics)
        self.assertIn("llm_spans_accepted", metrics)
        self.assertTrue(aggregate_exists)

    def test_fixture_transform_off_uses_canaries_for_verification_only(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.jsonl"
            # Canary is a structure-free contextual reference (no role-noun / idiom),
            # so the deterministic floors cannot catch it: fixture-off must therefore
            # leak, proving the canary is used for verification ONLY and is never fed
            # to the transform as an answer key. (Structured references like
            # "the roofing guy" are now caught by the generalizing role-only floor.)
            case = {
                "case_id": "case_001",
                "raw": "Loop back on the situation from our last sync.",
                "canaries": ["the situation from our last sync"],
                "expected_field_types": ["business_relationship"],
                "expected_semantics": ["private client reference"],
                "notes": "synthetic",
            }
            suite.write_text(json.dumps(case) + "\n", encoding="utf-8")
            metrics = run_eval_suite(
                suite,
                policy,
                root / "results",
                base_dir=root,
                fixture_transform=False,
            )
            artifact_id = "art_suite_case_001"
            with self.assertRaises(KeyError):
                read_safe(artifact_id, base_dir=root)
        self.assertFalse(metrics["fixture_transform_enabled"])
        self.assertEqual(metrics["cases_total"], 1)
        self.assertEqual(metrics["cases_passed"], 0)
        self.assertEqual(metrics["hard_leaks"], 1)
        self.assertEqual(metrics["removed_by_fixture_canary"], 0)
        self.assertEqual(metrics["cases_failed_without_fixture_transform"], 1)
        self.assertEqual(metrics["cases_where_fixture_would_have_saved"], 1)

    def test_fixture_transform_on_preserves_existing_behavior(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.jsonl"
            case = {
                "case_id": "case_001",
                "raw": "Call the roofing guy after lunch.",
                "canaries": ["the roofing guy"],
                "expected_field_types": ["business_relationship"],
                "expected_semantics": ["private client reference"],
                "notes": "synthetic",
            }
            suite.write_text(json.dumps(case) + "\n", encoding="utf-8")
            metrics = run_eval_suite(suite, policy, root / "results", base_dir=root)
        self.assertTrue(metrics["fixture_transform_enabled"])
        self.assertEqual(metrics["cases_passed"], 1)
        self.assertEqual(metrics["hard_leaks"], 0)
        self.assertGreater(metrics["removed_by_fixture_canary"], 0)


if __name__ == "__main__":
    unittest.main()
