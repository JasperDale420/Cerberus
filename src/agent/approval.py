from __future__ import annotations

import os
from typing import Any, Mapping, Tuple


def get_stage3_approval(stage3_cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Tuple[bool, str]:
    env_key = str(stage3_cfg.get("approval_env_var", "CERBERUS_STAGE3_APPROVED"))
    env_map = env or os.environ
    approved = str(env_map.get(env_key, "")).strip().lower() in ("1", "true", "yes")
    return approved, env_key
