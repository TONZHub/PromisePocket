from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from promise_pocket.pairing import InMemoryPairingStore


class PairingTests(unittest.TestCase):
    def test_code_links_source_actor_to_target_actor_once(self):
        store = InMemoryPairingStore()
        now = datetime(2026, 8, 31, 6, 0, tzinfo=timezone.utc)

        with patch("promise_pocket.pairing.secrets.randbelow", return_value=382731):
            pairing = store.create(target_actor_id="demo-actor", now=now)

        self.assertEqual("482731", pairing.code)
        self.assertEqual("alexa-source", store.resolve("alexa-source"))
        self.assertEqual(
            "demo-actor",
            store.claim(source_actor_id="alexa-source", code="482731", now=now),
        )
        self.assertEqual("demo-actor", store.resolve("alexa-source"))
        self.assertIsNone(
            store.claim(source_actor_id="other-alexa", code="482731", now=now)
        )

    def test_unlink_restores_source_actor_identity(self):
        store = InMemoryPairingStore()
        now = datetime(2026, 8, 31, 6, 0, tzinfo=timezone.utc)

        with patch("promise_pocket.pairing.secrets.randbelow", return_value=382731):
            pairing = store.create(target_actor_id="demo-actor", now=now)

        store.claim(source_actor_id="alexa-source", code=pairing.code, now=now)
        self.assertEqual("demo-actor", store.resolve("alexa-source"))
        self.assertTrue(store.unlink("alexa-source"))
        self.assertEqual("alexa-source", store.resolve("alexa-source"))
        self.assertFalse(store.unlink("alexa-source"))

    def test_expired_code_cannot_be_claimed(self):
        store = InMemoryPairingStore()
        now = datetime(2026, 8, 31, 6, 0, tzinfo=timezone.utc)
        with patch("promise_pocket.pairing.secrets.randbelow", return_value=1):
            pairing = store.create(target_actor_id="demo-actor", now=now)

        self.assertIsNone(
            store.claim(
                source_actor_id="alexa-source",
                code=pairing.code,
                now=now + timedelta(minutes=11),
            )
        )
        self.assertEqual("alexa-source", store.resolve("alexa-source"))


if __name__ == "__main__":
    unittest.main()
