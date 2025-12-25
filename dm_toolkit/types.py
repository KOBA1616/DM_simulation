# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import dm_ai_module
    import numpy as np
    import torch

# Basic aliases for C++ binding types
# Since dm_ai_module is a compiled extension, its types are not statically available
# without stub files. We use Any for now or specific names if we want to be semantic.

CardID = int
PlayerID = int
TurnNumber = int

# Data structures
JSON = Dict[str, Any]
CardDB = Dict[int, Any]  # Effectively Dict[CardID, dm_ai_module.CardDefinition]
Context = Dict[str, Any]

# Game Engine Types (Runtime)
# These are technically classes from dm_ai_module, but for mypy they are Any unless we have stubs.
GameState = Any
GameInstance = Any
Action = Any
EffectDef = Any
ConditionDef = Any
ScenarioConfig = Any

# ML Types
if TYPE_CHECKING:
    NPArray = np.ndarray
    Tensor = torch.Tensor
else:
    NPArray = Any
    Tensor = Any

# GUI Types
# We can add PyQt6 aliases here if needed
