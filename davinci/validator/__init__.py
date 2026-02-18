"""
Validator infrastructure layer.

This package contains:
- Validator: Main validator class (chain, wallet, scheduling, state)
- Config: CLI argument parsing and configuration

The actual evaluation business logic is in davinci.orchestration.
"""

from .config import (
    add_args,
    check_config,
    config_to_dict,
    get_config,
    setup_logging,
)
from .validator import Validator

__all__ = [
    "Validator",
    "add_args",
    "check_config",
    "config_to_dict",
    "get_config",
    "setup_logging",
]
