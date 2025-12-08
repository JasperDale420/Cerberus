import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
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

    def load_config(self, config_path: str = "config/config.yaml") -> Dict[str, Any]:
        """
        Loads configuration from a YAML file and overrides with env vars.
        Also loads and merges strategies.auto.yaml if present.
        """
        config = {}
        
        # 1. Load Main Config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            # Fallback or error? For now, empty dict + env vars
            pass

        # 2. Load Agent Overrides (strategies.auto.yaml)
        # Assuming it's in the same directory as config_path or fixed path
        auto_config_path = os.path.join(os.path.dirname(config_path), "strategies.auto.yaml")
        if os.path.exists(auto_config_path):
            try:
                with open(auto_config_path, "r") as f:
                    auto_config = yaml.safe_load(f) or {}
                    # Merge logic: Deep merge or top-level?
                    # Strategies config is usually under "strategies" key in main config?
                    # Or is strategies.yaml separate?
                    # If main config has "strategies": {...}, we merge auto_config into it.
                    # Let's assume auto_config structure mirrors "strategies" section.
                    
                    if "strategies" not in config:
                        config["strategies"] = {}
                    
                    # Simple merge for now: auto_config keys override config["strategies"] keys
                    # But we want to merge inner keys (like enabled, risk_factor)
                    for strat_name, strat_overrides in auto_config.items():
                        if strat_name not in config["strategies"]:
                            config["strategies"][strat_name] = {}
                        
                        # Update fields
                        config["strategies"][strat_name].update(strat_overrides)
                        
            except Exception as e:
                # Log error but don't crash? ConfigLoader doesn't have logger yet usually.
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
                path_parts = env_key[len("APP_"):].lower().split('_')
                
                current_level = config
                for i, part in enumerate(path_parts):
                    if i == len(path_parts) - 1: # Last part is the key to set
                        # Attempt to convert type if possible (e.g., 'true'/'false' to bool, numbers to int/float)
                        if isinstance(current_level, dict):
                            if env_value.lower() == 'true':
                                current_level[part] = True
                            elif env_value.lower() == 'false':
                                current_level[part] = False
                            elif env_value.isdigit():
                                current_level[part] = int(env_value)
                            elif env_value.replace('.', '', 1).isdigit(): # Check for float
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

    def load_all(self) -> Dict[str, Any]:
        """
        Loads all configuration files and merges them.
        This is a placeholder for more complex loading logic if needed.
        """
        # Example: load main config
        self.config.update(self.load_config("config.yaml"))
        return self.config
