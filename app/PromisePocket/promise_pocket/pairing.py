"""Short-lived Alexa pairing codes stored beside the Promise Pocket ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from typing import Protocol

from .settings import Settings


PAIRING_TTL = timedelta(minutes=10)
PAIRING_SORT_KEY = "link"
PAIR_CODE_PREFIX = "__pair_code__#"
PAIR_ACTOR_PREFIX = "__pair_actor__#"


@dataclass(frozen=True, slots=True)
class PairingCode:
    code: str
    target_actor_id: str
    expires_at: datetime


class PairingStore(Protocol):
    def create(
        self,
        *,
        target_actor_id: str,
        now: datetime | None = None,
    ) -> PairingCode: ...

    def claim(
        self,
        *,
        source_actor_id: str,
        code: str,
        now: datetime | None = None,
    ) -> str | None: ...

    def resolve(self, source_actor_id: str) -> str: ...

    def unlink(self, source_actor_id: str) -> bool: ...


def _utc_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("pairing timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_actor_id(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _validate_code(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("pairing code is required")
    code = "".join(character for character in value if character.isdigit())
    if len(code) != 6:
        raise ValueError("pairing code must contain exactly six digits")
    return code


class InMemoryPairingStore:
    def __init__(self) -> None:
        self._codes: dict[str, PairingCode] = {}
        self._links: dict[str, str] = {}

    def create(
        self,
        *,
        target_actor_id: str,
        now: datetime | None = None,
    ) -> PairingCode:
        target = _validate_actor_id(target_actor_id, field="target_actor_id")
        created_at = _utc_now(now)
        for _ in range(20):
            code = f"{secrets.randbelow(900000) + 100000:06d}"
            if code in self._codes:
                continue
            pairing = PairingCode(
                code=code,
                target_actor_id=target,
                expires_at=created_at + PAIRING_TTL,
            )
            self._codes[code] = pairing
            return pairing
        raise RuntimeError("could not allocate a unique pairing code")

    def claim(
        self,
        *,
        source_actor_id: str,
        code: str,
        now: datetime | None = None,
    ) -> str | None:
        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        clean_code = _validate_code(code)
        claimed_at = _utc_now(now)
        pairing = self._codes.get(clean_code)
        if pairing is None or pairing.expires_at < claimed_at:
            return None
        self._codes.pop(clean_code, None)
        self._links[source] = pairing.target_actor_id
        return pairing.target_actor_id

    def resolve(self, source_actor_id: str) -> str:
        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        return self._links.get(source, source)

    def unlink(self, source_actor_id: str) -> bool:
        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        return self._links.pop(source, None) is not None


class DynamoDbPairingStore:
    def __init__(self, table_name: str, region_name: str | None = None) -> None:
        import boto3

        self._table = boto3.resource("dynamodb", region_name=region_name).Table(
            table_name
        )

    @staticmethod
    def _code_key(code: str) -> dict[str, str]:
        return {
            "actor_id": f"{PAIR_CODE_PREFIX}{code}",
            "commitment_id": PAIRING_SORT_KEY,
        }

    @staticmethod
    def _actor_key(source_actor_id: str) -> dict[str, str]:
        return {
            "actor_id": f"{PAIR_ACTOR_PREFIX}{source_actor_id}",
            "commitment_id": PAIRING_SORT_KEY,
        }

    def create(
        self,
        *,
        target_actor_id: str,
        now: datetime | None = None,
    ) -> PairingCode:
        from botocore.exceptions import ClientError

        target = _validate_actor_id(target_actor_id, field="target_actor_id")
        created_at = _utc_now(now)
        expires_at = created_at + PAIRING_TTL
        for _ in range(20):
            code = f"{secrets.randbelow(900000) + 100000:06d}"
            item = {
                **self._code_key(code),
                "pairing_kind": "alexa_code",
                "target_actor_id": target,
                "pairing_created_at": created_at.isoformat(),
                "pairing_expires_at": int(expires_at.timestamp()),
            }
            try:
                self._table.put_item(
                    Item=item,
                    ConditionExpression="attribute_not_exists(actor_id)",
                )
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                    continue
                raise
            return PairingCode(
                code=code,
                target_actor_id=target,
                expires_at=expires_at,
            )
        raise RuntimeError("could not allocate a unique pairing code")

    def claim(
        self,
        *,
        source_actor_id: str,
        code: str,
        now: datetime | None = None,
    ) -> str | None:
        from botocore.exceptions import ClientError

        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        clean_code = _validate_code(code)
        claimed_at = _utc_now(now)
        now_epoch = int(claimed_at.timestamp())
        try:
            response = self._table.update_item(
                Key=self._code_key(clean_code),
                ConditionExpression=(
                    "attribute_exists(actor_id) AND "
                    "pairing_kind = :kind AND "
                    "attribute_not_exists(pairing_claimed_at) AND "
                    "pairing_expires_at >= :now"
                ),
                UpdateExpression=(
                    "SET pairing_claimed_at = :claimed_at, "
                    "pairing_claimed_by = :source"
                ),
                ExpressionAttributeValues={
                    ":kind": "alexa_code",
                    ":now": now_epoch,
                    ":claimed_at": claimed_at.isoformat(),
                    ":source": source,
                },
                ReturnValues="ALL_NEW",
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return None
            raise

        item = response.get("Attributes") or {}
        target = item.get("target_actor_id")
        if not isinstance(target, str) or not target:
            raise ValueError("pairing record is missing target_actor_id")

        self._table.put_item(
            Item={
                **self._actor_key(source),
                "pairing_kind": "alexa_link",
                "target_actor_id": target,
                "pairing_linked_at": claimed_at.isoformat(),
            }
        )
        return target

    def resolve(self, source_actor_id: str) -> str:
        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        response = self._table.get_item(Key=self._actor_key(source))
        item = response.get("Item")
        if not isinstance(item, dict) or item.get("pairing_kind") != "alexa_link":
            return source
        target = item.get("target_actor_id")
        return target if isinstance(target, str) and target else source

    def unlink(self, source_actor_id: str) -> bool:
        source = _validate_actor_id(source_actor_id, field="source_actor_id")
        response = self._table.delete_item(
            Key=self._actor_key(source),
            ReturnValues="ALL_OLD",
        )
        item = response.get("Attributes")
        return isinstance(item, dict) and item.get("pairing_kind") == "alexa_link"


def build_pairing_store(settings: Settings) -> PairingStore:
    if settings.table_name:
        return DynamoDbPairingStore(
            table_name=settings.table_name,
            region_name=settings.region_name,
        )
    if settings.local_dev:
        return InMemoryPairingStore()
    raise RuntimeError(
        "LOOSE_ENDS_TABLE is required outside local development; refusing "
        "to use ephemeral pairing storage"
    )
