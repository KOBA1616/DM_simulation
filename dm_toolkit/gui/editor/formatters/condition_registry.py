from typing import Dict, Any, Type, Optional
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class ConditionFormatterStrategy:
    """Base interface for condition text formatters."""

    @classmethod
    def format(cls, condition: Dict[str, Any], ctx: Optional[TextGenerationContext] = None) -> str:
        """
        Format the specific condition into Japanese text.
        Returns empty string if invalid or shouldn't be rendered.
        """
        raise NotImplementedError

    @classmethod
    def get_suffix(cls) -> str:
        """Returns the suffix to append after the condition text."""
        return ": "

class ConditionFormatterRegistry:
    """Registry for condition formatters."""

    _formatters: Dict[str, Type[ConditionFormatterStrategy]] = {}

    @classmethod
    def register(cls, condition_type: str, formatter_cls: Type[ConditionFormatterStrategy]) -> None:
        """Register a formatter class for a specific condition type."""
        cls._formatters[condition_type] = formatter_cls

    @classmethod
    def get_formatter(cls, condition_type: str) -> Optional[Type[ConditionFormatterStrategy]]:
        """Retrieve the formatter class for the given condition type."""
        return cls._formatters.get(condition_type)

def register_condition(condition_type: str):
    """Decorator to register a condition formatter strategy."""
    def wrapper(cls: Type[ConditionFormatterStrategy]):
        ConditionFormatterRegistry.register(condition_type, cls)
        return cls
    return wrapper
