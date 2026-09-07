from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("ALEXA_SKILL_ID", "amzn1.ask.skill.test-skill")
os.environ.setdefault(
    "AGENT_RUNTIME_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/LooseEnds",
)

fake_boto3 = types.ModuleType("boto3")
fake_boto3.client = Mock()
sys.modules.setdefault("boto3", fake_boto3)

from alexa import lambda_function


def event(intent_name: str | None = None, *, new_session: bool = True) -> dict:
    request = {"type": "LaunchRequest", "requestId": "request-onboarding-123"}
    if intent_name:
        request = {
            "type": "IntentRequest",
            "requestId": "request-onboarding-123",
            "intent": {"name": intent_name, "slots": {}},
        }
    return {
        "version": "1.0",
        "session": {
            "new": new_session,
            "sessionId": "session-onboarding-123",
            "application": {"applicationId": "amzn1.ask.skill.test-skill"},
            "user": {"userId": "amzn1.ask.account.private-user"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.test-skill"},
                "user": {"userId": "amzn1.ask.account.private-user"},
            }
        },
        "request": request,
    }


class AlexaOnboardingRepeatTests(unittest.TestCase):
    def test_interaction_model_includes_repeat_intent(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "interaction-model.json"
        )
        with open(model_path, encoding="utf-8") as model_file:
            intents = {
                item["name"]
                for item in json.load(model_file)["interactionModel"]["languageModel"]["intents"]
            }
        self.assertIn("AMAZON.RepeatIntent", intents)

    def test_unpaired_launch_gives_setup_instructions(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_status", "linked": False},
        ) as invoke:
            response = lambda_function.lambda_handler(event(), None)

        self.assertEqual("pair_status", invoke.call_args.args[1]["operation"])
        spoken = response["response"]["outputSpeech"]["text"]
        self.assertIn("To get started", spoken)
        self.assertIn("receipts dash app dot lol", spoken)
        self.assertIn("link code", spoken)
        self.assertFalse(response["response"]["shouldEndSession"])

    def test_paired_launch_skips_setup_instructions(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_status", "linked": True},
        ):
            response = lambda_function.lambda_handler(event(), None)

        spoken = response["response"]["outputSpeech"]["text"]
        self.assertEqual("Receipts is listening. What should I hold onto?", spoken)
        self.assertNotIn("receipts dash app dot lol", spoken)

    def test_pair_status_failure_does_not_break_launch(self):
        with patch.object(
            lambda_function,
            "_invoke",
            side_effect=RuntimeError("runtime unavailable"),
        ):
            response = lambda_function.lambda_handler(event(), None)

        self.assertEqual(
            "Receipts is listening. What should I hold onto?",
            response["response"]["outputSpeech"]["text"],
        )

    def test_repeat_replays_previous_speech_and_preserves_state(self):
        request = event("AMAZON.RepeatIntent", new_session=False)
        request["session"]["attributes"] = {
            "lastSpeech": "What time should I use?",
            "pendingCommitmentId": "commitment-1",
        }

        response = lambda_function.lambda_handler(request, None)

        self.assertEqual(
            "What time should I use?",
            response["response"]["outputSpeech"]["text"],
        )
        self.assertFalse(response["response"]["shouldEndSession"])
        self.assertEqual(
            "commitment-1",
            response["sessionAttributes"]["pendingCommitmentId"],
        )
        self.assertEqual(
            "What time should I use?",
            response["sessionAttributes"]["lastSpeech"],
        )

    def test_unpaired_help_repeats_setup_path(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_status", "linked": False},
        ):
            response = lambda_function.lambda_handler(
                event("AMAZON.HelpIntent", new_session=False), None
            )

        spoken = response["response"]["outputSpeech"]["text"]
        self.assertIn("receipts dash app dot lol", spoken)
        self.assertIn("repeat that", spoken)

    def test_paired_help_is_command_help_only(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_status", "linked": True},
        ):
            response = lambda_function.lambda_handler(
                event("AMAZON.HelpIntent", new_session=False), None
            )

        spoken = response["response"]["outputSpeech"]["text"]
        self.assertIn("review my receipts", spoken)
        self.assertIn("repeat that", spoken)
        self.assertNotIn("receipts dash app dot lol", spoken)


if __name__ == "__main__":
    unittest.main()
