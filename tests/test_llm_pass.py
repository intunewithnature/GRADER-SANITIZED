import unittest
from pathlib import Path

from grader.llm_client import LLMResponse
from grader.llm_pass import ChatLLMJudgmentPass, NoopLLMJudgmentPass, parse_span_json, prompt_profile_name
from grader.policy import load_policy


class LLMPassTests(unittest.TestCase):
    def test_noop_llm_pass(self):
        policy = load_policy(Path("policies") / "client_strategy.json")
        self.assertEqual(NoopLLMJudgmentPass().propose_spans("Bob Calder", policy), [])

    def test_parse_span_json_accepts_valid_span(self):
        result = parse_span_json(
            '{"spans":[{"text":"the roofing guy","field_type":"business_relationship","reason":"context","confidence":1.2}]}',
            "Call the roofing guy after the invoice issue.",
        )
        self.assertEqual(len(result.spans), 1)
        self.assertEqual(result.spans[0].field_type, "business_relationship")
        self.assertEqual(result.spans[0].confidence, 1.0)
        self.assertEqual(result.stats.llm_spans_accepted, 1)

    def test_parse_span_json_rejects_invalid_span(self):
        result = parse_span_json(
            '{"safe_to_send":true,"spans":[{"text":"missing","field_type":"allow_egress","confidence":0.5}]}',
            "No matching sensitive text here.",
        )
        self.assertEqual(result.spans, [])
        self.assertEqual(result.stats.llm_spans_rejected, 1)

    def test_parse_span_json_rejects_span_not_found_in_input(self):
        result = parse_span_json(
            '{"spans":[{"text":"not in input","field_type":"private_note","confidence":0.9}]}',
            "Only this sentence exists.",
        )
        self.assertEqual(result.spans, [])
        self.assertEqual(result.stats.llm_spans_rejected, 1)

    def test_prompt_profile_selection(self):
        self.assertEqual(prompt_profile_name("strict"), "strict_span_v1")
        self.assertEqual(prompt_profile_name("contextual"), "contextual_span_v2")
        self.assertEqual(prompt_profile_name("coreference"), "coreference_span_v3")

    def test_chat_pass_uses_selected_prompt_profile(self):
        class FakeClient:
            def __init__(self):
                self.message = ""

            def chat(self, message, *, model=None, timeout_seconds=30):
                self.message = message
                return LLMResponse(True, '{"spans":[]}')

        policy = load_policy(Path("policies") / "client_strategy.json")
        fake = FakeClient()
        ChatLLMJudgmentPass(fake, prompt_profile="coreference").propose_spans_with_stats(
            "The client is nervous.",
            policy,
        )
        self.assertIn("coreference-aware", fake.message)


if __name__ == "__main__":
    unittest.main()
