"""Thin Alexa Custom Skill adapter for the Receipts AgentCore runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import boto3


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
ALEXA_SKILL_ID = os.environ.get("ALEXA_SKILL_ID", "")
DEFAULT_TIMEZONE = os.environ.get("LOOSE_ENDS_TIMEZONE", "America/New_York")

_agentcore = None

PENDING_REVIEW_ID = "pendingV2CommitmentId"
PENDING_REVIEW_KIND = "pendingV2ReviewKind"
LAST_SPEECH_KEY = "lastSpeech"
REVIEW_CONFIRM_CANDIDATE = "confirm_candidate"
REVIEW_CONFIRM_LIKELY_DONE = "confirm_likely_done"
REVIEW_RESOLVE_OVERDUE = "resolve_overdue"
REVIEW_KINDS = {
    REVIEW_CONFIRM_CANDIDATE,
    REVIEW_CONFIRM_LIKELY_DONE,
    REVIEW_RESOLVE_OVERDUE,
}


def _agentcore_client():
    global _agentcore
    if _agentcore is None:
        _agentcore = boto3.client("bedrock-agentcore")
    return _agentcore


def _speech(
    text: str,
    *,
    end_session: bool = True,
    reprompt: str | None = None,
    session_attributes: dict[str, Any] | None = None,
):
    response: dict[str, Any] = {
        "outputSpeech": {"type": "PlainText", "text": text},
        "shouldEndSession": end_session,
    }
    if reprompt:
        response["reprompt"] = {
            "outputSpeech": {"type": "PlainText", "text": reprompt}
        }
    attributes = dict(session_attributes or {})
    attributes[LAST_SPEECH_KEY] = text
    return {
        "version": "1.0",
        "response": response,
        "sessionAttributes": attributes,
    }


def _first_turn(event: dict[str, Any], text: str) -> str:
    """Identify the skill when Alexa routes directly into a new session."""
    if event.get("session", {}).get("new") is True:
        return f"Receipts here. {text}"
    return text


def _verify_skill(event: dict[str, Any]) -> None:
    application_id = (
        event.get("context", {})
        .get("System", {})
        .get("application", {})
        .get("applicationId")
    ) or event.get("session", {}).get("application", {}).get("applicationId")
    if not ALEXA_SKILL_ID or application_id != ALEXA_SKILL_ID:
        raise ValueError("request application ID does not match this skill")


def _hashed_actor_id(prefix: str, source: str) -> str:
    return f"{prefix}-" + hashlib.sha256(source.encode()).hexdigest()


def _actor_id(event: dict[str, Any]) -> str:
    alexa_user_id = (
        event.get("context", {}).get("System", {}).get("user", {}).get("userId")
        or event.get("session", {}).get("user", {}).get("userId")
    )
    if not isinstance(alexa_user_id, str) or not alexa_user_id:
        raise ValueError("Alexa user ID is missing")
    # Pairing resolves this pseudonymous device identity to the shared ledger
    # inside AgentCore. No Alexa account-linking token is required.
    return _hashed_actor_id("alexa", alexa_user_id)


def _session_id(event: dict[str, Any]) -> str:
    source = event.get("session", {}).get("sessionId") or event["request"]["requestId"]
    # AgentCore session IDs must be at least 33 characters and are safest as a
    # stable, character-limited digest.
    return "alexa-session-" + hashlib.sha256(source.encode()).hexdigest()


def _read_runtime_response(response: dict[str, Any]) -> dict[str, Any]:
    body = response["response"]
    raw = body.read() if hasattr(body, "read") else b"".join(body)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _invoke(event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if not AGENT_RUNTIME_ARN:
        raise RuntimeError("AGENT_RUNTIME_ARN is not configured")
    actor_id = _actor_id(event)
    trusted_payload = {**payload, "actor_id": actor_id}
    response = _agentcore_client().invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        qualifier="DEFAULT",
        runtimeSessionId=_session_id(event),
        runtimeUserId=actor_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(trusted_payload).encode("utf-8"),
    )
    if response.get("statusCode") != 200:
        raise RuntimeError(f"AgentCore returned {response.get('statusCode')}")
    return _read_runtime_response(response)


def _pair_status(event: dict[str, Any]) -> bool | None:
    """Return pairing state without making skill launch depend on runtime health."""
    try:
        result = _invoke(event, {"operation": "pair_status"})
    except Exception:
        LOGGER.exception("Could not read Alexa pairing status")
        return None
    return result.get("linked") is True


def _launch(event: dict[str, Any]):
    paired = _pair_status(event)
    if paired is False:
        text = (
            "Receipts is listening. To get started, visit receipts dash app dot lol "
            "and create a linking code. Then say, link code, followed by the six digits."
        )
        return _speech(
            text,
            end_session=False,
            reprompt="Visit receipts dash app dot lol, create a linking code, then say link code followed by the six digits.",
        )
    return _speech(
        "Receipts is listening. What should I hold onto?",
        end_session=False,
        reprompt="Tell me a promise or task to remember.",
    )


def _help(event: dict[str, Any]):
    session_attributes = dict(event.get("session", {}).get("attributes", {}) or {})
    paired = _pair_status(event)
    if paired is False:
        text = (
            "To get started, visit receipts dash app dot lol and create a linking code. "
            "Then say, link code, followed by the six digits. You can say, repeat that, "
            "if you need me to say it again."
        )
        return _speech(
            _first_turn(event, text),
            end_session=False,
            reprompt="Visit receipts dash app dot lol, create a linking code, then say link code followed by the six digits.",
            session_attributes=session_attributes,
        )
    return _speech(
        _first_turn(
            event,
            "Try saying, review my receipts, or, I promised to call Mom tomorrow at noon. You can also say, repeat that.",
        ),
        end_session=False,
        reprompt="What should I remember?",
        session_attributes=session_attributes,
    )


def _repeat(event: dict[str, Any]):
    session_attributes = dict(event.get("session", {}).get("attributes", {}) or {})
    previous = session_attributes.get(LAST_SPEECH_KEY)
    if isinstance(previous, str) and previous.strip():
        return _speech(
            previous,
            end_session=False,
            reprompt=previous,
            session_attributes=session_attributes,
        )
    return _speech(
        "I don't have anything to repeat yet.",
        end_session=False,
        reprompt="What should I remember?",
        session_attributes=session_attributes,
    )


def _slot_value(intent: dict[str, Any], name: str) -> str | None:
    value = intent.get("slots", {}).get(name, {}).get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _capture(event: dict[str, Any], intent: dict[str, Any]):
    commitment = _slot_value(intent, "commitment")
    if not commitment:
        return _speech(
            _first_turn(event, "What should I hold onto?"),
            end_session=False,
            reprompt="Tell me the promise or task you want me to remember.",
        )
    result = _invoke(
        event,
        {"operation": "capture", "prompt": commitment, "timezone": DEFAULT_TIMEZONE},
    )
    if result.get("captured_commitment_ids"):
        captured = result.get("captured_commitments") or []
        if captured and captured[0].get("missing_information"):
            question = captured[0]["missing_information"][0]
            return _speech(
                _first_turn(event, question),
                end_session=False,
                reprompt=question,
                session_attributes={
                    "pendingCommitmentId": captured[0]["commitment_id"]
                },
            )
        return _speech(
            _first_turn(event, "Got it. I tucked that loose end away.")
        )
    return _speech(
        _first_turn(
            event,
            "I didn't hear a definite commitment there. What should I keep track of?",
        ),
        end_session=False,
        reprompt="Tell me what you need to do and when.",
    )


def _pair(event: dict[str, Any], intent: dict[str, Any]):
    code = _slot_value(intent, "code")
    if not code:
        return _speech(
            _first_turn(event, "What's the six digit linking code?"),
            end_session=False,
            reprompt="Say the six digit code from Receipts.",
        )

    digits = "".join(character for character in code if character.isdigit())
    if len(digits) != 6:
        return _speech(
            _first_turn(event, "That linking code should be six digits. Try it again."),
            end_session=False,
            reprompt="Say the six digit code from Receipts.",
        )

    result = _invoke(event, {"operation": "pair_claim", "code": digits})
    if result.get("linked") is True:
        return _speech(
            _first_turn(event, "Connected. Your Alexa promises will use the same receipts now.")
        )
    return _speech(
        _first_turn(event, "That code is invalid or expired. Make a new linking code and try again.")
    )


def _review(event: dict[str, Any]):
    result = _invoke(event, {"operation": "v2_review"})
    items = result.get("items", [])
    if not items:
        return _speech(_first_turn(event, "Nothing needs you right now."))

    item = items[0]
    commitment_id = item.get("commitment_id")
    review_kind = item.get("kind")
    prompt = item.get("prompt")
    if (
        not isinstance(commitment_id, str)
        or not commitment_id
        or review_kind not in REVIEW_KINDS
        or not isinstance(prompt, str)
        or not prompt
    ):
        raise ValueError("AgentCore returned an invalid v2 review item")

    return _speech(
        _first_turn(event, prompt),
        end_session=False,
        reprompt="Say yes or no.",
        session_attributes={
            PENDING_REVIEW_ID: commitment_id,
            PENDING_REVIEW_KIND: review_kind,
        },
    )


def _answer_review(event: dict[str, Any], *, accepted: bool):
    session_attributes = dict(
        event.get("session", {}).get("attributes", {}) or {}
    )
    commitment_id = session_attributes.get(PENDING_REVIEW_ID)
    review_kind = session_attributes.get(PENDING_REVIEW_KIND)
    if (
        not isinstance(commitment_id, str)
        or not commitment_id
        or review_kind not in REVIEW_KINDS
    ):
        return _speech(
            "Ask me to review your receipts first, so I know which promise you mean."
        )

    if review_kind == REVIEW_CONFIRM_CANDIDATE:
        operation = "v2_confirm" if accepted else "v2_cancel"
        confirmation = (
            "Okay. I'll track that promise."
            if accepted
            else "Okay. I won't track that promise."
        )
    elif review_kind == REVIEW_CONFIRM_LIKELY_DONE:
        operation = "v2_done" if accepted else "v2_reopen"
        confirmation = (
            "Done. I closed that promise."
            if accepted
            else "Okay. I reopened that promise."
        )
    elif accepted:
        operation = "v2_done"
        confirmation = "Done. I closed that promise."
    else:
        # A negative answer to an overdue review is itself useful human input,
        # but it must not change state or fabricate completion.
        return _speech("Okay. I'll keep that promise open.")

    _invoke(
        event,
        {
            "operation": operation,
            "commitment_id": commitment_id,
        },
    )
    return _speech(confirmation)


def _clarify(event: dict[str, Any], intent: dict[str, Any]):
    session_attributes = dict(
        event.get("session", {}).get("attributes", {}) or {}
    )
    commitment_id = session_attributes.get("pendingCommitmentId")
    answer = _slot_value(intent, "answer")

    if not commitment_id:
        LOGGER.warning("Alexa clarification arrived without pending commitment state")
        return _speech(
            _first_turn(
                event,
                "What should I hold onto?",
            ),
            end_session=False,
            reprompt="Tell me a promise or task to remember.",
        )

    if not answer:
        # Alexa can match the clarification intent while still failing to fill the
        # AMAZON.TIME slot. Keep the pending commitment in the response so one
        # imperfect recognition does not permanently strand the conversation.
        LOGGER.info("Alexa clarification intent arrived without an answer time slot")
        return _speech(
            "What time should I use?",
            end_session=False,
            reprompt="Say a time, like 9 A.M.",
            session_attributes=session_attributes,
        )

    result = _invoke(
        event,
        {
            "operation": "clarify",
            "commitment_id": commitment_id,
            "answer": answer,
            "timezone": DEFAULT_TIMEZONE,
        },
    )
    if result.get("updated_commitment_ids"):
        return _speech("Perfect. I added the time to that loose end.")
    return _speech(
        "I still need a specific time.",
        end_session=False,
        reprompt="Say a time, like 9 A.M.",
        session_attributes=session_attributes,
    )


def _handle_scheduled_event(event: dict[str, Any]) -> dict[str, Any]:
    """Quiet scheduled review triggered by EventBridge.

    Evaluates overdue promises. If nothing is overdue or no action is required,
    it exits quietly without emitting false notifications.
    """
    if not AGENT_RUNTIME_ARN:
        raise RuntimeError("AGENT_RUNTIME_ARN is not configured")
    actor_id = event.get("actor_id") or "scheduled-review"
    event_id = event.get("id") or "default-scheduled-id"
    session_id = f"alexa-scheduled-{hashlib.sha256(event_id.encode()).hexdigest()}"
    try:
        response = _agentcore_client().invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            runtimeUserId=actor_id,
            contentType="application/json",
            accept="application/json",
            payload=json.dumps({"operation": "v2_overdue", "actor_id": actor_id}).encode("utf-8"),
        )
        if response.get("statusCode") != 200:
            raise RuntimeError(f"AgentCore returned {response.get('statusCode')}")
        result = _read_runtime_response(response)
        overdue_ids = result.get("overdue_ids") or []
        nudges = result.get("nudges") or []
        return {
            "status": "ok",
            "overdue_count": len(overdue_ids),
            "nudges_prepared": len(nudges),
        }
    except Exception as e:
        LOGGER.exception("Scheduled review evaluation failed: %s", e)
        return {"status": "error", "error": str(e)}


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context
    try:
        source = event.get("source")
        if source == "aws.events" or event.get("detail-type") == "Scheduled Event":
            return _handle_scheduled_event(event)

        _verify_skill(event)
        request = event.get("request", {})
        request_type = request.get("type")
        if request_type == "AlexaSkillEvent.ProactiveSubscriptionChanged":
            # Multicast events do not require retaining Alexa's raw user ID.
            return {"version": "1.0", "response": {}}
        if request_type == "LaunchRequest":
            return _launch(event)
        if request_type == "SessionEndedRequest":
            return {"version": "1.0", "response": {}}
        if request_type != "IntentRequest":
            return _speech("I couldn't understand that request.")

        intent = request.get("intent", {})
        name = intent.get("name")
        if name == "AMAZON.RepeatIntent":
            return _repeat(event)
        if name == "CaptureCommitmentIntent":
            return _capture(event, intent)
        if name == "LinkAlexaIntent":
            return _pair(event, intent)
        if name in {"ReviewPromisePocketIntent", "ReviewReceiptsIntent"}:
            return _review(event)
        if name in {"AMAZON.YesIntent", "CompleteReviewedPromiseIntent"}:
            return _answer_review(event, accepted=True)
        if name in {"AMAZON.NoIntent", "KeepPromiseOpenIntent"}:
            return _answer_review(event, accepted=False)
        if name == "ClarifyCommitmentIntent":
            return _clarify(event, intent)
        if name == "AMAZON.HelpIntent":
            return _help(event)
        if name in {"AMAZON.CancelIntent", "AMAZON.StopIntent"}:
            return _speech("Okay. Your promises will wait here.")
        return _speech(
            _first_turn(
                event, "Try saying, remember that I need to call Mom tomorrow."
            )
        )
    except Exception:
        LOGGER.exception("Alexa request failed")
        return _speech("I hit a snag while reaching Receipts. Please try again in a moment.")
