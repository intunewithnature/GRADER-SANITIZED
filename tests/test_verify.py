import tempfile
import unittest
from pathlib import Path

from grader.policy import load_policy
from grader.verify import verify_safe_text


class VerifyTests(unittest.TestCase):
    def test_no_canary_in_safe_text(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "Person_1 has a modest daily budget.",
                policy,
                artifact_id="art_missing",
                base_dir=tmp,
                canaries=["Bob Calder", "$75/day"],
            )
        self.assertTrue(result.passed)

    def test_numeric_leak_detection(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "Client still has 75 per day.",
                policy,
                artifact_id="art_numeric",
                base_dir=tmp,
                canaries=["$75/day"],
            )
        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.numeric_leaks, 1)

    def test_spelled_number_leak_detection(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "Client still has seventy five per day.",
                policy,
                artifact_id="art_spelled_numeric",
                base_dir=tmp,
                canaries=["$75/day"],
            )
        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.numeric_leaks, 1)

    def test_numeric_range_leak_detection(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "Client still has 70-80 per day.",
                policy,
                artifact_id="art_range_numeric",
                base_dir=tmp,
                canaries=["$75/day"],
            )
        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.numeric_leaks, 1)

    def test_punctuation_and_homoglyph_normalized_leak_detection(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "Smіth-Roofing is still visible.",
                policy,
                artifact_id="art_homoglyph",
                base_dir=tmp,
                canaries=["Smith Roofing"],
            )
        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.hard_leaks, 1)

    def test_field_aware_verification_avoids_private_note_false_positive(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "a private contextual note",
                policy,
                artifact_id="art_private_note_false_positive",
                base_dir=tmp,
                canaries=["private-contextual"],
                canary_field_types=["private_note"],
            )
        self.assertTrue(result.passed)

    def test_short_reveal_originals_do_not_trigger_substring_leaks(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            from grader.reveal_store import RevealStore

            RevealStore(tmp).get_or_create("art_short", "art_short", "person_name", "B", "Person_{n}")
            result = verify_safe_text(
                "a business budget amount",
                policy,
                artifact_id="art_short",
                base_dir=tmp,
                canaries=[],
            )
        self.assertTrue(result.passed)

    def test_placeholder_field_label_does_not_count_as_original_hint(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            from grader.reveal_store import RevealStore

            RevealStore(tmp).get_or_create("art_email", "art_email", "email", "contact@email.test", "Email_{n}")
            result = verify_safe_text(
                "Email_1 asked for a private business update.",
                policy,
                artifact_id="art_email",
                base_dir=tmp,
                canaries=[],
            )
        self.assertTrue(result.passed)

    def test_near_match_detection(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_safe_text(
                "B0b Calder is still visible.",
                policy,
                artifact_id="art_near",
                base_dir=tmp,
                canaries=["Bob Calder"],
            )
        self.assertFalse(result.passed)
        self.assertTrue(result.hard_leaks >= 1 or result.near_leaks >= 1)


if __name__ == "__main__":
    unittest.main()
