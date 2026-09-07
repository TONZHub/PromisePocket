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


def unlink_event(*, new_session: bool = False):
    return {
        "version": "1.0",
        "session": {
            "new": new_session,
            "sessionId": "session-unlink-123",
            "application": {"applicationId": "amzn1.ask.skill.test-skill"},
            "user": {"userId": "amzn1.ask.account.private-user"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.test-skill"},
                "user": {"userId": "amzn1.ask.account.private-user"},
            }
        },
        "request": {
            "type": "IntentRequest",
            "requestId": "request-unlink-123",
            "intent": {"name": "UnlinkAlexaIntent", "slots": {}},
        },
    }


class AlexaUnlinkTests(unittest.TestCase):
    def test_interaction_model_includes_unlink_phrases(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "interaction-model.json"
        )
        with open(model_path, encoding="utf-8") as model_file:
            intents = {
                intent["name"]: set(intent.get("samples", []))
                for intent in json.load(model_file)["interactionModel"]["languageModel"]["intents"]
            }

        self.assertIn("UnlinkAlexaIntent", intents)
        self.assertTrue(
            {"unlink my receipts", "disconnect this Alexa", "forget this device"}.issubset(
                intents["UnlinkAlexaIntent"]
            )
        )

    def test_unlink_intent_calls_pair_unlink_and_confirms(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_unlink", "unlinked": True},
        ) as invoke:
            response = lambda_function.lambda_handler(unlink_event(), None)

        self.assertEqual("pair_unlink", invoke.call_args.args[1]["operation"])
        self.assertTrue(response["response"]["shouldEndSession"])
        self.assertIn("Disconnected", response["response"]["outputSpeech"]["text"])

    def test_unlink_intent_is_safe_when_already_unlinked(self):
        with patch.object(
            lambda_function,
            "_invoke",
            return_value={"operation": "pair_unlink", "unlinked": False},
        ):
            response = lambda_function.lambda_handler(unlink_event(), None)

        self.assertIn("isn't linked", response["response"]["outputSpeech"]["text"])


if __name__ == "__main__":
    unittest.main()
