from typing import Dict, Any, List, Optional

class TextGenerationContext:
    """
    Holds contextual information about the card being processed during text generation.
    Replaces the need to pass `is_spell`, `card_mega_last_burst`, and `sample` flags
    everywhere.
    """
    def __init__(self, card_data: Dict[str, Any], sample: Optional[List[Any]] = None, evaluated_stats: Optional[Dict[str, Any]] = None):
        self.card_data: Dict[str, Any] = card_data or {}
        self.sample: Optional[List[Any]] = sample
        self.evaluated_stats: Dict[str, Any] = evaluated_stats or {}

        # Derived properties
        self.is_spell: bool = self.card_data.get("type", "CREATURE") == "SPELL"
        self.has_mega_last_burst: bool = self.card_data.get("keywords", {}).get("mega_last_burst", False)

        # Will be set during formatting for context-aware AST input-link resolution
        self.current_commands_list: Optional[List[Dict[str, Any]]] = None

        # Metadata extraction logic
        self.metadata = {
            "targets": False,
            "draws": False,
            "discards": False,
            "destroys": False,
        }

        from dm_toolkit.gui.editor.formatters.error_reporter import ErrorReporter
        self.error_reporter = ErrorReporter()

    @property
    def data(self) -> Dict[str, Any]:
        return self.card_data

class TextGenerationResult:
    """Holds the generated text and semantic metadata about the card's effects."""
    def __init__(self, text: str, metadata: Dict[str, bool]):
        self.text = text
        self.metadata = metadata
