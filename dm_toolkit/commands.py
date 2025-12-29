from typing import Any, Dict, Optional, Protocol, runtime_checkable

# Phase B: Re-export from unified commands_new to maintain API compatibility
from dm_toolkit.commands_new import ICommand, BaseCommand, wrap_action

__all__ = ["ICommand", "BaseCommand", "wrap_action"]
