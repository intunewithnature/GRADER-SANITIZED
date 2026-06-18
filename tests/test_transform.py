import tempfile
import unittest
from pathlib import Path

from grader.models import DetectedSpan
from grader.policy import load_policy
from grader.transform import blind_text, resolve_overlaps


class TransformTests(unittest.TestCase):
    def test_redaction_pseudonymization_and_budget_generalization(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            raw = "Bob Calder has sk-EXAMPLEfakekey00000 and budget $75/day. Bob Calder agrees."
            result = blind_text(
                raw,
                policy,
                artifact_id="art_transform",
                base_dir=tmp,
                fixture_canaries=["Bob Calder", "$75/day"],
                fixture_field_types=["person_name", "budget"],
            )
        self.assertNotIn("sk-", result.safe_text)
        self.assertIn("[REDACTED_API_KEY]", result.safe_text)
        self.assertEqual(result.safe_text.count("Person_1"), 2)
        self.assertIn("a modest daily budget", result.safe_text)
        self.assertNotIn("Bob", result.safe_text)
        self.assertNotIn("$75", result.safe_text)

    def test_overlap_prefers_more_confident_longer_sensitive_span(self):
        spans = [
            DetectedSpan(0, 10, "Bob Calder", "person_name", "heuristic", 0.6, "name"),
            DetectedSpan(4, 10, "Calder", "private_note", "manual", 0.5, "partial"),
            DetectedSpan(20, 30, "acct_12345", "account_id", "regex", 0.9, "account"),
        ]
        selected = resolve_overlaps(spans)
        self.assertEqual([span.text for span in selected], ["Bob Calder", "acct_12345"])

    def test_overlap_merge_prevents_llm_displacement(self):
        # A high-confidence narrow LLM span must NOT be able to un-cover a canary
        # that a wider, lower-confidence detector span was covering.
        detector = DetectedSpan(0, 18, "the silent partner", "business_relationship", "regex", 0.6, "floor")
        llm = DetectedSpan(4, 10, "silent", "private_note", "llm", 0.9, "granite")
        merged = resolve_overlaps([detector, llm], "the silent partner behind it")
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0, 18))
        self.assertEqual(merged[0].text, "the silent partner")

    def test_llm_span_cannot_cause_canary_leak_in_blind(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        raw = "Keep the silent partner confidential."
        narrow_llm = [DetectedSpan(9, 15, "silent", "private_note", "llm", 0.95, "granite")]
        with tempfile.TemporaryDirectory() as tmp:
            result = blind_text(raw, policy, artifact_id="art_disp", base_dir=tmp, llm_spans=narrow_llm)
        # the role-only floor + union-merge must still remove the full phrase
        self.assertNotIn("the silent partner", result.safe_text)

    def test_homoglyph_fixture_variant_is_blinded(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = blind_text(
                "Smіth Roofing should not survive.",
                policy,
                artifact_id="art_homoglyph_transform",
                base_dir=tmp,
                fixture_canaries=["Smith Roofing"],
                fixture_field_types=["business_name"],
            )
        self.assertEqual(result.verification_status, "passed")
        self.assertIn("Org_1", result.safe_text)

    def test_model_name_blinding(self):
        policy = load_policy(Path("policies") / "blind_model_eval.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = blind_text(
                "Claude beat Gemini while Qwen lagged.",
                policy,
                artifact_id="art_models",
                base_dir=tmp,
            )
        self.assertNotIn("Claude", result.safe_text)
        self.assertNotIn("Gemini", result.safe_text)
        self.assertNotIn("Qwen", result.safe_text)
        self.assertIn("Model_1", result.safe_text)

    def test_labeled_syntactic_private_values_are_blinded(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        raw = (
            "Use key_alpha123 and token_4455. "
            "Call 555-1212 for client_7788 next week near Lansing. "
            "The local routing model seat is named."
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = blind_text(raw, policy, artifact_id="art_syntactic_floor", base_dir=tmp)
        self.assertEqual(result.verification_status, "passed")
        self.assertIn("[REDACTED_API_KEY]", result.safe_text)
        self.assertIn("[REDACTED_SECRET_TOKEN]", result.safe_text)
        self.assertIn("Phone_1", result.safe_text)
        self.assertIn("Account_1", result.safe_text)
        self.assertIn("a recent date", result.safe_text)
        self.assertIn("a regional location", result.safe_text)
        self.assertIn("Model_1", result.safe_text)
        self.assertNotIn("key_alpha123", result.safe_text)
        self.assertNotIn("token_4455", result.safe_text)
        self.assertNotIn("555-1212", result.safe_text)
        self.assertNotIn("client_7788", result.safe_text)
        self.assertNotIn("Lansing", result.safe_text)
        self.assertNotIn("local routing model seat", result.safe_text)


if __name__ == "__main__":
    unittest.main()
