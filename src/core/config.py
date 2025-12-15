import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigLoader:
    """
    Loads configuration from YAML files and environment variables.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursive deep merge of dictionaries.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load_config(self, config_dir: str = "config") -> Dict[str, Any]:
        """
        Loads configuration from multiple YAML files and overrides with env vars.
        Files loaded: config.yaml, strategies.yaml, risk.yaml, scanner.yaml, universe.yaml, logging.yaml
        Also loads strategies.auto.yaml if present.
        """
        config: Dict[str, Any] = {}

        # 1. Load Standard Config Files
        # Order ensures keys are merged correctly, though they should be distinct top-level keys mostly.
        files_to_load = [
            "config.yaml",
            "strategies.yaml",
            "risk.yaml",
            "scanner.yaml",
            "universe.yaml",
            "logging.yaml",
        ]

        # Since argument is named config_path in original signature but we want dir,
        # note: the original method signature was: load_config(self, config_path: str = "config/config.yaml")
        # We need to respect that if external callers use it, OR we change it.
        # But wait, the previous code had config_dir in __init__ but load_config took a path default to config/config.yaml
        # To be safe and fix-forward:
        # We will use self.config_dir (Path object) to find these files.
        # And if the user passed a specific single file as 'config_path', we load that FIRST (or last?),
        # but typically we want the whole suite.
        # Let's pivot: we'll ignore the default "config/config.yaml" if it's just the default, and load our suite.
        # If the user passed a custom path that ISNT the default, we might treat it as an override?
        # Simpler: Just load the suite from separate files found in the directory of the passed config_path (or self.config_dir).

        # Let's assume standard usage.
        base_dir = self.config_dir

        for fname in files_to_load:
            fpath = base_dir / fname
            if fpath.exists():
                try:
                    with open(fpath, "r") as f:
                        c = yaml.safe_load(f) or {}
                        self._deep_merge(config, c)
                except Exception as e:
                    print(f"Warning: Failed to load {fname}: {e}")

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
                print(f"Warning: Failed to load auto config: {e}")

        # 3. Env Var Overrides
        self._override_from_env(config)

        return config

    def _override_from_env(self, config: Dict[str, Any]) -> None:
        """
        Overrides configuration values with environment variables.
        Looks for keys like 'APP_SETTING_SUBSET_KEY' to override config['setting']['subset']['key'].
        """
        for env_key, env_value in os.environ.items():
            # Convert env_key (e.g., 'APP_STRATEGIES_MYSTRAT_ENABLED') to config path (e.g., ['strategies', 'mystrat', 'enabled'])
            # Assuming a prefix like 'APP_'
            if env_key.startswith("APP_"):
                path_parts = env_key[len("APP_") :].lower().split("_")

                current_level = config
                for i, part in enumerate(path_parts):
                    if i == len(path_parts) - 1:  # Last part is the key to set
                        # Attempt to convert type if possible (e.g., 'true'/'false' to bool, numbers to int/float)
                        if isinstance(current_level, dict):
                            if env_value.lower() == "true":
                                current_level[part] = True
                            elif env_value.lower() == "false":
                                current_level[part] = False
                            elif env_value.isdigit():
                                current_level[part] = int(env_value)
                            elif env_value.replace(
                                ".", "", 1
                            ).isdigit():  # Check for float
                                current_level[part] = float(env_value)
                            else:
                                current_level[part] = env_value
                        break

                    if part not in current_level or not isinstance(
                        current_level[part], dict
                    ):
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

    def load_all(self) -> Dict[str, Any]:
        """
        Loads all configuration files and merges them.
        This is a placeholder for more complex loading logic if needed.
        """
        # Example: load main config
        self.config.update(self.load_config("config.yaml"))
        return self.config
