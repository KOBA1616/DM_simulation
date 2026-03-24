from typing import Dict, Any, Type, Optional

class SpecialKeywordFormatterBase:
    """Base interface for special keyword text generation."""

    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        """
        Format the special keyword's text.
        Returns the formatted string (without the leading bullet '■ '),
        or empty string if it shouldn't be rendered.
        """
        raise NotImplementedError

class SpecialKeywordRegistry:
    """Registry for special keyword text formatters."""

    _formatters: Dict[str, Type[SpecialKeywordFormatterBase]] = {}

    @classmethod
    def register(cls, keyword_id: str, formatter_cls: Type[SpecialKeywordFormatterBase]) -> None:
        """Register a formatter class for a specific special keyword."""
        cls._formatters[keyword_id] = formatter_cls

    @classmethod
    def get_formatter(cls, keyword_id: str) -> Optional[Type[SpecialKeywordFormatterBase]]:
        """Retrieve the formatter class for the given special keyword."""
        return cls._formatters.get(keyword_id)

    @classmethod
    def is_special_keyword(cls, keyword_id: str) -> bool:
        """Check if a keyword is a special keyword managed by this registry."""
        return keyword_id in cls._formatters

# Decorator for easy registration
def register_special_keyword(keyword_id: str):
    def wrapper(cls: Type[SpecialKeywordFormatterBase]):
        SpecialKeywordRegistry.register(keyword_id, cls)
        return cls
    return wrapper
