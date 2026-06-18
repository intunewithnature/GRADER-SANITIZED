import tempfile
import unittest
from pathlib import Path

from grader.artifact_store import ArtifactStore
from grader.commands import read_safe, reveal_report, submit_judgment
from grader.models import BlindResult
from grader.policy import load_policy
from grader.reveal_store import RevealStore
from grader.transform import blind_text


class RevealStoreTests(unittest.TestCase):
    def test_reveal_map_stored_locally_and_normal_report_hides_values(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = blind_text(
                "Bob Calder met Smith Roofing.",
                policy,
                artifact_id="art_reveal",
                base_dir=tmp,
                fixture_canaries=["Bob Calder", "Smith Roofing"],
                fixture_field_types=["person_name", "business_name"],
            )
            ArtifactStore(tmp).save_artifact(result)
            entries = RevealStore(tmp).list_entries("art_reveal")
            judgment_id = submit_judgment("art_reveal", "Person_1 should follow up with Org_1.", base_dir=tmp)
            report = reveal_report("art_reveal", judgment_id, base_dir=tmp)
            db_exists = (Path(tmp) / ".grader" / "reveal_map.sqlite").exists()
        self.assertEqual(len(entries), 2)
        self.assertTrue(db_exists)
        serialized = str(report)
        self.assertNotIn("Bob Calder", serialized)
        self.assertNotIn("Smith Roofing", serialized)

    def test_failed_artifact_cannot_be_read_as_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            ArtifactStore(tmp).save_artifact(
                BlindResult(
                    artifact_id="art_failed",
                    safe_text="unsafe failed text",
                    hidden_field_types=[],
                    canary_count=1,
                    verification_status="failed",
                    verification_summary="passed=False; hard=1; near=0; numeric=0; reveal_map=0; coreference=0; warnings=0",
                    input_sha256="input",
                    safe_sha256="safe",
                    created_at="2026-06-16T00:00:00+00:00",
                )
            )
            with self.assertRaises(PermissionError):
                read_safe("art_failed", base_dir=tmp)


if __name__ == "__main__":
    unittest.main()
