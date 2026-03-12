# -*- coding: utf-8 -*-
"""
Thin wrapper that centralizes interactions with the native C++ engine.

Purpose:
- Encapsulate creation of `GameInstance`, PhaseManager fast_forward/start_game
- Keep C++ access in a single location so `GameSession` can be refactored safely
"""
from typing import Any, Optional

try:
    import dm_ai_module
except Exception:
    dm_ai_module = None


class GameController:
    """Lightweight controller that delegates to `dm_ai_module`.

    This is intentionally minimal: it only wraps a few engine entrypoints so
    higher-level code can be migrated without changing semantics.
    """

    def __init__(self) -> None:
        self.game_instance: Optional[Any] = None
        self.native_card_db: Optional[Any] = None

    def create_instance(self, seed: int, native_card_db: Any) -> Any:
        """Create and hold a GameInstance via the native module."""
        if dm_ai_module is None:
            raise RuntimeError("dm_ai_module not available")
        self.native_card_db = native_card_db
        self.game_instance = dm_ai_module.GameInstance(seed, native_card_db)
        return self.game_instance

    def start_and_fast_forward(self, gs: Any, native_card_db: Any) -> None:
        """Start the game and fast-forward using PhaseManager if available."""
        if dm_ai_module is None:
            return
        if hasattr(dm_ai_module, 'PhaseManager'):
            try:
                dm_ai_module.PhaseManager.start_game(gs, native_card_db)
            except Exception:
                pass
            try:
                dm_ai_module.PhaseManager.fast_forward(gs, native_card_db)
            except Exception:
                pass

    def fast_forward(self, gs: Any, native_card_db: Any) -> None:
        """Fast-forward helper that guards PhaseManager calls."""
        if dm_ai_module is None:
            return
        if hasattr(dm_ai_module, 'PhaseManager'):
            try:
                dm_ai_module.PhaseManager.fast_forward(gs, native_card_db)
            except Exception:
                pass
