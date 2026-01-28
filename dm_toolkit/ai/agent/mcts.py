import math
# import torch
# import numpy as np
import dm_ai_module
from dm_toolkit import commands
from typing import Any, Optional, List, Dict, Tuple, Callable

# WARNING: This module is deprecated. Use dm_ai_module.MCTS (C++) instead.

class MCTSNode:
    """Deprecated. Use C++ implementation."""
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Any = None) -> None:
        pass

class MCTS:
    """
    Deprecated Python implementation of MCTS.
    Use dm_ai_module.MCTS instead.
    """
    def __init__(self, network: Any, card_db: Dict[str, Any], simulations: int = 100, c_puct: float = 1.0, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25, state_converter: Optional[Callable[[Any, int, Dict], Any]] = None, action_encoder: Optional[Callable[[Any, Any, int], int]] = None) -> None:
        raise RuntimeError("This Python MCTS implementation is deprecated. Use dm_ai_module.MCTS (C++) instead.")
