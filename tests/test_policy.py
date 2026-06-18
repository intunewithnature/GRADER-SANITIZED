import unittest
from pathlib import Path

from grader.policy import load_policy


class PolicyTests(unittest.TestCase):
    def test_policy_loading(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        self.assertEqual(policy.name, "client_strategy")
        self.assertEqual(policy.fields["api_key"].op, "redact")
        self.assertEqual(policy.fields["person_name"].format, "Person_{n}")


if __name__ == "__main__":
    unittest.main()
