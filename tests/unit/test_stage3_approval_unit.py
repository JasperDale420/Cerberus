from __future__ import annotations

from src.agent.approval import get_stage3_approval


def test_get_stage3_approval_uses_default_env_var() -> None:
    approved, env_key = get_stage3_approval({}, env={"CERBERUS_STAGE3_APPROVED": "true"})
    assert env_key == "CERBERUS_STAGE3_APPROVED"
    assert approved is True


def test_get_stage3_approval_uses_custom_env_var() -> None:
    cfg = {"approval_env_var": "STAGE3_OK"}
    approved, env_key = get_stage3_approval(cfg, env={"STAGE3_OK": "yes"})
    assert env_key == "STAGE3_OK"
    assert approved is True


def test_get_stage3_approval_rejects_unapproved_values() -> None:
    cfg = {"approval_env_var": "STAGE3_OK"}
    approved, env_key = get_stage3_approval(cfg, env={"STAGE3_OK": "no"})
    assert env_key == "STAGE3_OK"
    assert approved is False
