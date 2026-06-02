"""
Configuration loader for VegeSSL.

Provides a simple interface to load and access YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Configuration object with attribute-style access.
    
    Allows accessing nested config values as attributes:
        config.training.batch_size
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        return f"Config({self.__dict__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default fallback."""
        return getattr(self, key, default)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        Config object with attribute-style access
    """
    if config_path is None:
        # Load default config from package
        config_dir = Path(__file__).parent
        config_path = config_dir / "default.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override values (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
