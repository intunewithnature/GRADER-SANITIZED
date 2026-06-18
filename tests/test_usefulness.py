import json
import unittest

from grader.llm_client import LLMResponse
from grader.usefulness import UsefulnessJudge


class UsefulnessJudgeTests(unittest.TestCase):
    def test_remote_judge_payload_excludes_raw_canaries_and_reveal_values(self):
        class FakeClient:
            def __init__(self):
                self.message = ""

            def chat(self, message, *, model=None, timeout_seconds=30):
                self.message = message
                return LLMResponse(
                    True,
                    json.dumps(
                        {
                            "semantic_preservation_score": 0.9,
                            "missing_semantics": [],
                            "overredaction": False,
                            "notes": "safe",
                        }
                    ),
                )

        judge = UsefulnessJudge("none")
        fake = FakeClient()
        judge.client = fake
        result = judge.judge(
            safe_text="Person_1 has a private client status.",
            expected_semantics=["client status preserved"],
            policy_name="client_strategy",
            case_id="case_safe",
        )
        self.assertTrue(result.ok)
        self.assertIn("Person_1", fake.message)
        self.assertIn("case_safe", fake.message)
        self.assertNotIn("Bob Calder", fake.message)
        self.assertNotIn("the roofing guy", fake.message)
        self.assertNotIn("ORIGINAL_VALUE_SENTINEL", fake.message)


if __name__ == "__main__":
    unittest.main()
