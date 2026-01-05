# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Callable, Dict
import random

from dm_toolkit.types import GameState, CardDB, Action
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.commands import wrap_action

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

class GameController:
    """
    Decouples Game Logic from UI (Qt).
    Handles GameState management, action execution loops, and phase transitions.
    """

    def __init__(self, callback_update_ui: Callable[[], None], callback_log: Callable[[str], None]):
        self.gs: Optional[GameState] = None
        self.card_db: CardDB = {}
        self.callback_update_ui = callback_update_ui
        self.callback_log = callback_log
        self.is_running = False

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        self.card_db = card_db
        if dm_ai_module:
            self.gs = dm_ai_module.GameState(seed)
            self.gs.setup_test_duel()
            # Start the game logic (draw initial hands, etc.)
            if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        else:
            self.gs = None
        self.callback_log("Game Initialized via Controller")
        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        if not dm_ai_module:
            return

        seed = random.randint(0, 100000)
        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        if p0_deck: self.gs.set_deck(0, p0_deck)
        if p1_deck: self.gs.set_deck(1, p1_deck)

        if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
             dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)

        self.callback_log("Game Reset")
        self.callback_update_ui()

    def execute_action(self, action: Any) -> None:
        if not self.gs: return

        # This mirrors the logic in app.py but without Qt dependencies
        # In a full refactor, 'wrap_action' and logging would happen here.
        # For now, we rely on the caller (GameWindow) to handle the complex wrapping/logging
        # because it requires 'tr' (translation) and specific formatting.
        # This method is a placeholder for the future pure-logic execution path.

        try:
             # Ensure the action is converted to a unified command dict first
             cmd_dict = ensure_executable_command(action)
             # Wrap it into an ICommand
             command = wrap_action(cmd_dict)
             if command:
                 command.execute(self.gs)
                 self.callback_log(f"Controller Action: {cmd_dict.get('type', 'UNKNOWN')}")
             else:
                 # Fallback if wrapping fails
                 EngineCompat.ExecuteCommand(self.gs, action, self.card_db)

        except Exception as e:
             self.callback_log(f"Controller Execution Error: {e}")

        self.callback_update_ui()

    def step_phase(self) -> None:
        """
        Executes one step of the game loop (AI move or phase transition).
        """
        if not self.gs: return
        if self.gs.game_over:
            self.callback_log("Game Over")
            return

        # Simple step logic
        active_pid = EngineCompat.get_active_player_id(self.gs)
        # AI Logic would go here

        actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
        if not actions:
            EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
        else:
            # Basic AI: Random or First
            best_action = actions[0]
            self.execute_action(best_action)

        self.callback_update_ui()
