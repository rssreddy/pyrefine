"""
Configuration module for SELF-REFINE with Change-of-Thought framework.
"""

from .settings import (
    Config,
    get_config,
    set_config,
    load_config,
    get_api_key
)

__all__ = [
    "Config",
    "get_config", 
    "set_config",
    "load_config",
    "get_api_key"
]