"""Amazon Bedrock AgentCore Runtime entrypoint for Pocket Promise.

The legacy Promise Pocket operations remain available while the v2 cross-system
commitment ledger is built alongside them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from promise_pocket.agent import build_agent, build_clarification_agent
from promise_pocket.arbiter_v2 import arbitrate_outgoing_message, build_arbiter_agent
from promise_pocket.ingest_v2 import SourceMessage
from promise_pocket.ledger_v2 import PromiseLedger, PromiseState
from promise_pocket.ledger_v2_storage import build_ledger_v2_store
from promise_pocket.pairing import build_pairing_store
from promise_pocket.reconcile_v2 import build_evidence_agent, reconcile_outgoing_message
from promise_pocket.review_v2 import build_review_queue
from promise_pocket.service import CommitmentService
from promise_pocket.settings import Settings
from promise_pocket.storage import build_store


app = BedrockAgentCoreApp()
settings = Settings.from_environment()
service = CommitmentService(build_store(settings))
v2_ledger = PromiseLedger(build_ledger_v2_store(settings))
pairing_store = build_pairing_store(settings)


def _actor_id(payload: dict[str, Any], context: Any | None) -> str:
    """Resolve identity supplied by an IAM-authorized invocation adapter."""

    actor_id = payload.get("actor_id")
    if settings.local_dev and not actor_id:
        actor_id = settings.dev_actor
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError(
            "actor_id from an IAM-authorized invocation adapter is required"
        )
    return actor_id.strip()


def _parse_now(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, str):
        raise ValueError("now must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("now must include a UTC offset")
    return parsed


def _require_commitment_id(payload: dict[str, Any]) -> str:
    commitment_id = payload.get("commitment_id")
    if not isinstance(commitment_id, str) or not commitment_id.strip():
        raise ValueError("commitment_id is required")
    return commitment_id.strip()


def _require_pairing_code(payload: dict[str, Any]) -> str:
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("pairing code is required")
    return code.strip()


def _source_message(payload: dict[str, Any], operation: str) -> SourceMessage:
    raw_message = payload.get("source_message")
    if not isinstance(raw_message, dict):
        raise ValueError(f"source_message is required for {operation}")
    message = SourceMessage.model_validate(raw_message)
    if message.direction != "sent":
        raise ValueError(f"{operation} only accepts outgoing source messages")
    return message


def _invoke_v2(
    *,
    operation: str,
    payload: dict[str, Any],
    actor_id: str,
) -> dict[str, Any] | None:
    """Handle Pocket Promise v2 operations, or return None for legacy routing."""

    if operation == "v2_arbitrate":
        message = _source_message(payload, operation)
        timezone_name = payload.get("timezone") or settings.timezone_name
        if not isinstance(timezone_name, str) or not timezone_name.strip():
            raise ValueError("timezone must be a non-empty IANA timezone name")

        candidate_ids: list[str] = []
        agent = build_arbiter_agent(
            v2_ledger,
            settings.model_id,
            on_candidate=candidate_ids.append,
        )
        result = arbitrate_outgoing_message(
            agent,
            actor_id=actor_id,
            message=message,
            timezone_name=timezone_name,
        )
        candidates = [
            v2_ledger.get(actor_id=actor_id, commitment_id=commitment_id)
            for commitment_id in candidate_ids
        ]
        return {
            "operation": operation,
            "result": result,
            "candidate_ids": candidate_ids,
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in candidates
                if candidate is not None
            ],
        }

    if operation == "v2_reconcile":
        message = _source_message(payload, operation)
        active_promises = [
            record
            for record in v2_ledger.list_for_actor(actor_id=actor_id)
            if record.status in {PromiseState.ACTIVE, PromiseState.OVERDUE}
        ]
        if not active_promises:
            return {
                "operation": operation,
                "result": "No active promises to reconcile.",
                "likely_done_ids": [],
                "items": [],
            }

        likely_done_ids: list[str] = []
        agent = build_evidence_agent(
            v2_ledger,
            settings.model_id,
            on_likely_done=likely_done_ids.append,
        )
        result = reconcile_outgoing_message(
            agent,
            actor_id=actor_id,
            message=message,
            active_promises=active_promises,
        )
        items = [
            v2_ledger.get(actor_id=actor_id, commitment_id=commitment_id)
            for commitment_id in likely_done_ids
        ]
        return {
            "operation": operation,
            "result": result,
            "likely_done_ids": likely_done_ids,
            "items": [
                item.model_dump(mode="json")
                for item in items
                if item is not None
            ],
        }

    if operation == "v2_list":
        records = v2_ledger.list_for_actor(actor_id=actor_id)
        return {
            "operation": operation,
            "items": [record.model_dump(mode="json") for record in records],
        }

    if operation == "v2_review":
        items = build_review_queue(
            v2_ledger,
            actor_id=actor_id,
            now=_parse_now(payload.get("now")),
        )
        return {
            "operation": operation,
            "attention_required": bool(items),
            "items": [item.model_dump(mode="json") for item in items],
        }

    if operation == "v2_overdue":
        now = _parse_now(payload.get("now"))
        transitioned = v2_ledger.evaluate_overdue(actor_id=actor_id, now=now)
        nudges = v2_ledger.prepare_overdue_nudges(actor_id=actor_id, now=now)
        return {
            "operation": operation,
            "overdue_ids": [record.commitment_id for record in transitioned],
            "nudges": nudges,
        }

    if operation == "v2_confirm":
        record = v2_ledger.confirm(
            actor_id=actor_id,
            commitment_id=_require_commitment_id(payload),
        )
        return {"operation": operation, "item": record.model_dump(mode="json")}

    if operation == "v2_done":
        record = v2_ledger.mark_done(
            actor_id=actor_id,
            commitment_id=_require_commitment_id(payload),
        )
        return {"operation": operation, "item": record.model_dump(mode="json")}

    if operation == "v2_reopen":
        record = v2_ledger.reopen(
            actor_id=actor_id,
            commitment_id=_require_commitment_id(payload),
        )
        return {"operation": operation, "item": record.model_dump(mode="json")}

    if operation == "v2_cancel":
        record = v2_ledger.cancel(
            actor_id=actor_id,
            commitment_id=_require_commitment_id(payload),
        )
        return {"operation": operation, "item": record.model_dump(mode="json")}

    if operation.startswith("v2_"):
        raise ValueError(f"unsupported Pocket Promise v2 operation: {operation}")

    return None


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any | None = None) -> dict[str, Any]:
    """Route Pocket Promise pairing, v2, and legacy operations."""

    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    source_actor_id = _actor_id(payload, context)
    operation = payload.get("operation", "capture")
    if not isinstance(operation, str):
        raise ValueError("operation must be a string")

    if operation == "pair_create":
        pairing = pairing_store.create(target_actor_id=source_actor_id)
        return {
            "operation": operation,
            "code": pairing.code,
            "expires_at": pairing.expires_at.isoformat(),
        }

    if operation == "pair_claim":
        target_actor_id = pairing_store.claim(
            source_actor_id=source_actor_id,
            code=_require_pairing_code(payload),
        )
        return {
            "operation": operation,
            "linked": target_actor_id is not None,
        }

    if operation == "pair_status":
        resolved_actor_id = pairing_store.resolve(source_actor_id)
        return {
            "operation": operation,
            "linked": resolved_actor_id != source_actor_id,
        }

    if operation == "pair_unlink":
        return {
            "operation": operation,
            "unlinked": pairing_store.unlink(source_actor_id),
        }

    # Pairing is deliberately resolved inside the IAM-authorized runtime. Alexa
    # only supplies its opaque, pseudonymous actor id; linked surfaces then land
    # in the same ledger without giving the Alexa Lambda direct DynamoDB access.
    actor_id = pairing_store.resolve(source_actor_id)

    v2_response = _invoke_v2(
        operation=operation,
        payload=payload,
        actor_id=actor_id,
    )
    if v2_response is not None:
        return v2_response

    # ---- Legacy Promise Pocket path: retained until the v2 adapters replace it. ----
    if operation == "review":
        items = service.review(actor_id=actor_id, now=_parse_now(payload.get("now")))
        return {
            "operation": "review",
            "attention_required": bool(items),
            "items": [item.model_dump(mode="json") for item in items],
        }

    if operation == "clarify":
        commitment_id = payload.get("commitment_id")
        answer = payload.get("answer")
        if not isinstance(commitment_id, str) or not commitment_id:
            raise ValueError("commitment_id is required for clarification")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer is required for clarification")
        existing = service.get(actor_id=actor_id, commitment_id=commitment_id)
        if existing is None:
            raise ValueError("commitment was not found for this actor")
        timezone_name = payload.get("timezone") or settings.timezone_name
        invocation_time = datetime.now(timezone.utc).isoformat()
        updated_ids: list[str] = []
        agent = build_clarification_agent(
            service,
            settings.model_id,
            actor_id,
            commitment_id,
            on_update=updated_ids.append,
        )
        result = agent(
            f"Invocation time in UTC: {invocation_time}\n"
            f"User timezone: {timezone_name}\n"
            f"Original commitment: {existing.raw_text}\n"
            f"Follow-up answer: {answer.strip()}"
        )
        return {
            "operation": "clarify",
            "result": result.message,
            "updated_commitment_ids": updated_ids,
        }

    if operation != "capture":
        raise ValueError(
            "operation must be capture, clarify, review, pair_create, pair_claim, "
            "pair_status, pair_unlink, or a supported v2 operation"
        )

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required for capture")

    timezone_name = payload.get("timezone") or settings.timezone_name
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("timezone must be a non-empty IANA timezone name")

    invocation_time = datetime.now(timezone.utc).isoformat()
    contextual_prompt = (
        f"Invocation time in UTC: {invocation_time}\n"
        f"User timezone: {timezone_name}\n"
        f"User message: {prompt.strip()}"
    )

    captured_commitment_ids: list[str] = []
    agent = build_agent(
        service=service,
        model_id=settings.model_id,
        on_capture=captured_commitment_ids.append,
    )
    result = agent(
        contextual_prompt,
        actor_id=actor_id,
        timezone_name=timezone_name,
        invocation_time=invocation_time,
        captured_commitment_ids=[],
    )

    return {
        "operation": "capture",
        "result": result.message,
        "captured_commitment_ids": captured_commitment_ids,
        "captured_commitments": [
            service.get(actor_id=actor_id, commitment_id=commitment_id).model_dump(
                mode="json"
            )
            for commitment_id in captured_commitment_ids
        ],
    }


if __name__ == "__main__":
    app.run()