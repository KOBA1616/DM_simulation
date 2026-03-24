from typing import Dict, Any, Type, Optional
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase

class CommandFormatterRegistry:
    """Registry for command formatters."""

    _formatters: Dict[str, Type[CommandFormatterBase]] = {}

    @classmethod
    def register(cls, command_type: str, formatter_cls: Type[CommandFormatterBase]) -> None:
        """Register a formatter class for a specific command type."""
        cls._formatters[command_type] = formatter_cls

    @classmethod
    def get_formatter(cls, command_type: str) -> Optional[Type[CommandFormatterBase]]:
        """Retrieve the formatter class for the given command type."""
        return cls._formatters.get(command_type)

    @classmethod
    def has_formatter(cls, command_type: str) -> bool:
        """Check if a formatter is registered for the command type."""
        return command_type in cls._formatters

# Decorator for easy registration
def register_formatter(command_type: str):
    def wrapper(cls: Type[CommandFormatterBase]):
        CommandFormatterRegistry.register(command_type, cls)
        return cls
    return wrapper
