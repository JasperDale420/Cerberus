import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.core.errors import ErrorCode

# Load environment variables from .env file
load_dotenv()


class ConfigLoader:
    """
    Loads configuration from YAML files and environment variables.
    """

    def __init__(self, config_dir: str = "config", logger: Optional[object] = None):
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self.logger = logger

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursive deep merge of dictionaries.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load_config(self, config_path_or_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads configuration from multiple YAML files and overrides with env vars.
        Files loaded: config.yaml, strategies.yaml, risk.yaml, scanner.yaml, universe.yaml, logging.yaml
        Also loads strategies.auto.yaml if present.
        """
        config: Dict[str, Any] = {}

        # 1. Load Standard Config Files
        # PRD 2.1: support YAML/JSON config suites.
        # Deterministic precedence per basename: .yaml > .yml > .json.
        basenames = [
            "config",
            "strategies",
            "risk",
            "scanner",
            "universe",
            "logging",
        ]

        # Honor caller-provided config path/dir (e.g., CLI `--config`).
        # - If a file path is provided, load the suite from its parent directory,
        #   then load the specific file to override suite settings.
        # - If a directory path is provided, load the suite from that directory.
        # - Otherwise, fall back to `self.config_dir`.
        base_dir = self.config_dir
        specific_file: Optional[Path] = None  # Track specific file for override loading
        if config_path_or_dir:
            base = Path(config_path_or_dir)
            if base.exists() and base.is_file():
                base_dir = base.parent
                specific_file = base  # Remember the specific file to load later
            elif base.exists() and base.is_dir():
                base_dir = base

        def _load_file(path: Path) -> Dict[str, Any]:
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            if suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8")) or {}
            raise ValueError(f"Unsupported config extension: {path}")

        for basename in basenames:
            chosen: Optional[Path] = None
            for ext in (".yaml", ".yml", ".json"):
                p = base_dir / f"{basename}{ext}"
                if p.exists():
                    chosen = p
                    break
            if chosen is None:
                continue
            try:
                c = _load_file(chosen)
                self._deep_merge(config, c)
            except Exception as e:
                if self.logger is not None and hasattr(self.logger, "error"):
                    self.logger.error(
                        "Failed to load config file",
                        error_code=ErrorCode.CONFIG_LOAD_FAILED.value,
                        file=str(chosen),
                        error=str(e),
                    )
                raise

        # 1b. Load Specific Config File if one was explicitly provided
        # This allows e.g. config_vwap_trend_rider.yaml to override config.yaml settings
        if specific_file is not None and specific_file.exists():
            try:
                c = _load_file(specific_file)
                self._deep_merge(config, c)
            except Exception as e:
                if self.logger is not None and hasattr(self.logger, "error"):
                    self.logger.error(
                        "Failed to load specific config file",
                        error_code=ErrorCode.CONFIG_LOAD_FAILED.value,
                        file=str(specific_file),
                        error=str(e),
                    )
                raise

        # 2. Load Agent Overrides (strategies.auto.yaml)
        auto_config_path = base_dir / "strategies.auto.yaml"
        if auto_config_path.exists():
            try:
                with open(auto_config_path, "r") as f:
                    auto_config = yaml.safe_load(f) or {}
                    if "strategies" not in config:
                        config["strategies"] = {}
                    self._deep_merge(config["strategies"], auto_config)
            except Exception as e:
                if self.logger is not None and hasattr(self.logger, "warning"):
                    self.logger.warning(
                        "Failed to load strategies.auto.yaml; continuing without overrides",
                        error_code=ErrorCode.CONFIG_LOAD_FAILED.value,
                        file=str(auto_config_path),
                        error=str(e),
                    )

        # 3. Env Var Overrides
        self._override_from_env(config)

        # 4. Normalize strategy overrides (PRD 9.2):
        # Allow agent overrides to use:
        #   vwap_reversion:
        #     params: {band_sigma: 1.5}
        #     metadata: {...}
        # while keeping runtime strategy configs flat (band_sigma at top-level).
        strategies = config.get("strategies") if isinstance(config, dict) else None
        if isinstance(strategies, dict):
            # PRD 9.2 example uses CamelCase keys (e.g., VWAPReversion). The runtime
            # system uses snake_case strategy IDs (e.g., vwap_reversion).
            def _camel_to_snake(value: str) -> str:
                out: list[str] = []
                s = str(value)
                for i, ch in enumerate(s):
                    if ch.isupper() and i > 0 and (s[i - 1].islower() or (i + 1 < len(s) and s[i + 1].islower())):
                        out.append("_")
                    out.append(ch.lower())
                return "".join(out)

            camel_keys = sorted([k for k in strategies.keys() if isinstance(k, str) and any(c.isupper() for c in k)])
            for key in camel_keys:
                nk = _camel_to_snake(key)
                if nk == key:
                    continue
                src_cfg = strategies.get(key)
                if nk in strategies:
                    dst_cfg = strategies.get(nk)
                    if isinstance(dst_cfg, dict) and isinstance(src_cfg, dict):
                        # Merge CamelCase overrides into canonical snake_case key deterministically.
                        self._deep_merge(dst_cfg, dict(src_cfg))
                    else:
                        strategies[nk] = src_cfg
                else:
                    strategies[nk] = src_cfg
                # Avoid duplicated strategy IDs after normalization.
                try:
                    del strategies[key]
                except Exception:
                    if self.logger is not None and hasattr(self.logger, "debug"):
                        self.logger.debug("Failed to delete normalized strategy key", key=key, exc_info=True)

            for strat_name, strat_cfg in strategies.items():
                if not isinstance(strat_cfg, dict):
                    continue
                params = strat_cfg.get("params")
                if isinstance(params, dict) and params:
                    # Params override existing keys deterministically.
                    self._deep_merge(strat_cfg, dict(params))
                # PRD/config compatibility: keep common aliases in sync deterministically.
                # - PRD uses `sigma_band` for VWAP Reversion, earlier configs/agent tuning may use `band_sigma`.
                if str(strat_name).strip().lower() == "vwap_reversion":
                    if "sigma_band" not in strat_cfg and "band_sigma" in strat_cfg:
                        strat_cfg["sigma_band"] = strat_cfg.get("band_sigma")
                    if "band_sigma" not in strat_cfg and "sigma_band" in strat_cfg:
                        strat_cfg["band_sigma"] = strat_cfg.get("sigma_band")

        return config

    # Safety-critical risk keys that CANNOT be overridden via APP_* env vars.
    # These protect against accidental or malicious weakening of risk limits.
    # To change these, edit config/risk.yaml directly.
    _BLOCKED_ENV_OVERRIDE_PATHS: set[str] = {
        "risk__max_daily_loss",
        "risk__max_risk_per_trade",
        "risk__max_open_risk",
        "risk__max_open_positions",
        "risk__max_positions_per_strategy",
        "risk__max_notional_per_order",
        "risk__max_notional_per_symbol",
        "risk__max_trades_per_day",
        "risk__risk_mode",
        "position_mismatch_mode",
    }

    def _override_from_env(self, config: Dict[str, Any]) -> None:
        """
        Overrides configuration values with environment variables.
        Looks for keys like 'APP_SETTING_SUBSET_KEY' to override config['setting']['subset']['key'].

        Safety: blocks overrides to risk-critical keys (see _BLOCKED_ENV_OVERRIDE_PATHS).
        All accepted overrides are logged for audit trail.
        """
        for env_key, env_value in os.environ.items():
            # Convert env_key to config path using double-underscore as nesting separator.
            # Single underscores are preserved within key names (e.g., max_daily_loss).
            # Example: APP_RISK__MAX_DAILY_LOSS=1000 -> config["risk"]["max_daily_loss"] = 1000
            if env_key.startswith("APP_"):
                path_parts = env_key[len("APP_") :].lower().split("__")
                override_path = "__".join(path_parts)

                # Block overrides to safety-critical risk keys
                if override_path in self._BLOCKED_ENV_OVERRIDE_PATHS:
                    if self.logger and hasattr(self.logger, "error"):
                        self.logger.error(
                            "BLOCKED env var override of safety-critical risk key",
                            env_key=env_key,
                            path=override_path,
                        )
                    continue

                # Log all accepted overrides for audit trail
                if self.logger and hasattr(self.logger, "warning"):
                    self.logger.warning(
                        "Config override via env var",
                        env_key=env_key,
                        path=override_path,
                        value=env_value if "secret" not in override_path and "key" not in override_path else "***",
                    )

                current_level = config
                for i, part in enumerate(path_parts):
                    if i == len(path_parts) - 1:  # Last part is the key to set
                        # Attempt to convert type if possible (e.g., 'true'/'false' to bool, numbers to int/float)
                        if isinstance(current_level, dict):
                            if env_value.lower() == "true":
                                current_level[part] = True
                            elif env_value.lower() == "false":
                                current_level[part] = False
                            elif env_value.lstrip("-").isdigit():
                                current_level[part] = int(env_value)
                            elif env_value.lstrip("-").replace(".", "", 1).isdigit():  # Check for float
                                current_level[part] = float(env_value)
                            else:
                                current_level[part] = env_value
                        break

                    if part not in current_level or not isinstance(current_level[part], dict):
                        # Create nested dict if it doesn't exist or is not a dict
                        current_level[part] = {}
                    current_level = current_level[part]

    def get_env(self, key: str, default: Optional[str] = None) -> str:
        """
        Gets an environment variable.
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Missing required environment variable: {key}")
        return value
