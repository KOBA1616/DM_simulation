from typing import Dict, Any, Type, Optional, List

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

    @classmethod
    def format_numbered_keyword(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        """Format a numbered keyword (e.g. MEKRAID 5) by extracting its value."""
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        val = card_data.get(keyword_id)
        if not val:
            val = card_data.get("keywords", {}).get(keyword_id)
        if isinstance(val, dict):
            val = val.get("value")
        if val:
            return f"{kw_str}{val}"
        return kw_str

    @classmethod
    def get_unbound_text(cls, card_data: Dict[str, Any]) -> List[str]:
        """
        Returns additional text lines for a special keyword that might exist on a card
        without being strictly defined in the `keywords` flag dict (e.g. effects based).
        Used by the main text generator to ensure critical keywords are displayed.
        """
        return []

    @classmethod
    def extract_requirements(cls, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract required conditions (e.g., source races, civilizations) from card_data explicitly.
        """
        return {}

    @classmethod
    def is_special_only_effect(cls, effect: Dict[str, Any], card_data: Dict[str, Any]) -> bool:
        """
        Determine if the given effect node exists purely to realize this special keyword.
        If it does, the main generator can choose to skip generic formatting for it.
        """
        return False

class SpecialKeywordRegistry:
    """Registry for special keyword text formatters."""

    @classmethod
    def is_special_only_effect(cls, effect: Dict[str, Any], card_data: Dict[str, Any]) -> bool:
        """
        Check if the effect exists solely to realize a special keyword.
        """
        special_keyword_id = effect.get("special_keyword_id")
        if special_keyword_id and special_keyword_id in cls._formatters:
             return True

        return False

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
