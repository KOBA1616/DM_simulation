from typing import Any, Dict, List, Set, Optional, TYPE_CHECKING

# Centralized type aliases for gradual typing across the project
CardID = int
PlayerID = int
TurnNumber = int

# Data structures
JSON = Dict[str, Any]
# map CardID to CardDefinition-like objects (use int keys)
CardDB = Dict[int, Any]
Context = Dict[str, Any]
Effects = List[Any]
Deck = List[Any]
CivGroups = List[Any]
CardCounts = Dict[str, int]
SeenCards = Set[str]
ResultsList = List[Any]

__all__ = [
    "CardID",
    "PlayerID",
    "TurnNumber",
    "JSON",
    "CardDB",
    "Context",
    "Effects",
    "Deck",
    "CivGroups",
    "CardCounts",
    "SeenCards",
    "ResultsList",
]

# Optional imports for typing-heavy libraries
if TYPE_CHECKING:
    import dm_ai_module  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

# ML placeholders
if TYPE_CHECKING:
    NPArray = np.ndarray
    Tensor = torch.Tensor
else:
    NPArray = Any
    Tensor = Any

# Runtime aliases for engine types to help other modules import names from dm_toolkit.types
GameState = Any
Action = Any
