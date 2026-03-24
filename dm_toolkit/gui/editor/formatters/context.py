from typing import Dict, Any, List, Optional

class TextGenerationContext:
    """
    Holds contextual information about the card being processed during text generation.
    Replaces the need to pass `is_spell`, `card_mega_last_burst`, and `sample` flags
    everywhere.
    """
    def __init__(self, card_data: Dict[str, Any], sample: Optional[List[Any]] = None):
        self.card_data: Dict[str, Any] = card_data or {}
        self.sample: Optional[List[Any]] = sample

        # Derived properties
        self.is_spell: bool = self.card_data.get("type", "CREATURE") == "SPELL"
        self.has_mega_last_burst: bool = self.card_data.get("keywords", {}).get("mega_last_burst", False)

    @property
    def data(self) -> Dict[str, Any]:
        return self.card_data
