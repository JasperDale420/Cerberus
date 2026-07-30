"""Ownership must survive the gateway's client_order_id namespacing.

Cerberus submits through Data-Gateway (``--order-executor`` defaults to
``gateway``) but reads orders and positions back with the Alpaca SDK directly.
The gateway wraps every submitted ``client_order_id`` as ``c-{client_id}-{id}``,
so an order sent as ``cerberus_...`` is read back as ``c-cerberus-cerberus_...``.

Matching only the bare form meant flatten skipped its own open orders, refused
to close its own positions, and reconciliation dropped its own positions — all
silently, because "none of these are mine" is a normal-looking outcome on a
shared account.
"""

from __future__ import annotations

import pytest

from src.engine.execution import is_own_client_order_id


@pytest.mark.parametrize(
    "coid,expected",
    [
        # What the gateway returns for a gateway-routed order.
        ("c-cerberus-cerberus_momentum-AAPL-1750000000000-ab12cd", True),
        # --order-executor alpaca still mints the bare form.
        ("cerberus_momentum-AAPL-1750000000000-ab12cd", True),
        # Other Empire systems on the shared account.
        ("c-3roses-3roses_abc", False),
        ("c-orion-orion_abc", False),
        ("orbit-abc", False),
        ("optionsbot_abc", False),
        ("drogon-abc", False),
        # Alpaca-assigned ids carry no ownership information.
        ("aabd3317-b12a-4528-a029-eab886774ada", False),
        ("", False),
        (None, False),
        (12345, False),
        # A foreign client whose id merely begins with ours must not match.
        ("c-cerberusXX-cerberus_abc", False),
    ],
)
def test_is_own_client_order_id(coid, expected) -> None:
    assert is_own_client_order_id(coid) is expected
