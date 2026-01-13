# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Callable, Dict, Tuple
import random
import os

from dm_toolkit.types import GameState, CardDB, Action
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

class GameSession:
    """
    Manages the GameState, execution loop, and phase transitions.
    Logic extracted from app.py and merged with GameController.
    """

    def __init__(self,
                 callback_update_ui: Callable[[], None],
                 callback_log: Callable[[str], None],
                 callback_input_request: Optional[Callable[[], None]] = None,
                 callback_action_executed: Optional[Callable[[Dict[str, Any]], None]] = None):

        self.gs: Optional[GameState] = None
        self.card_db: CardDB = {}
        self.callback_update_ui = callback_update_ui
        self.callback_log = callback_log
        self.callback_input_request = callback_input_request
        self.callback_action_executed = callback_action_executed

        self.is_running = False
        self.is_processing = False

        # Player Modes (0: Human, 1: AI by default)
        self.player_modes = {0: 'AI', 1: 'AI'}

        self.last_action: Optional[Action] = None

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        self.card_db = card_db
        if dm_ai_module:
            self.gs = dm_ai_module.GameState(seed)
            self.gs.setup_test_duel()
            if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        else:
            self.gs = None

        self.callback_log(tr("Game Initialized via Session"))
        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        if not dm_ai_module:
            return

        seed = random.randint(0, 100000)
        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        if p0_deck:
            self.gs.set_deck(0, p0_deck)
        if p1_deck:
            self.gs.set_deck(1, p1_deck)

        if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
            dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def set_player_mode(self, player_id: int, mode: str):
        """ mode: 'Human' or 'AI' """
        self.player_modes[player_id] = mode

    def get_player_mode(self, player_id: int) -> str:
        return self.player_modes.get(player_id, 'AI')

    def execute_action(self, action: Any) -> None:
        """
        Executes an action, wraps it, logs it, and updates UI.
        """
        if not self.gs: return

        self.last_action = action

        # 1. Normalize to Command Dict (Unified Path)
        try:
            # Check for to_dict (e.g. ICommand wrappers)
            if hasattr(action, 'to_dict'):
                raw_action = action.to_dict()
            else:
                raw_action = action

            cmd_dict = ensure_executable_command(raw_action)
        except Exception as e:
            self.callback_log(tr("Command Conversion Error: {error}").format(error=e))
            return

        # 2. Execute via EngineCompat
        try:
            EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)

            # 3. Log
            log_str = f"P{EngineCompat.get_active_player_id(self.gs)} {tr('Action')}: {cmd_dict.get('type', 'UNKNOWN')}"
            if 'to_zone' in cmd_dict:
                log_str += f" -> {cmd_dict['to_zone']}"
            self.callback_log(log_str)

            if self.callback_action_executed:
                self.callback_action_executed(cmd_dict)

        except RuntimeError as e:
            # Distinct handling for Engine errors (C++ std::runtime_error)
            self.callback_log(tr("Engine Error: {error}").format(error=e))
        except Exception as e:
            # Generic execution errors
            self.callback_log(tr("Execution Error: {error}").format(error=e))

        # Check input wait
        if self.check_and_handle_input_wait(): return

        # Auto Pass check
        if dm_ai_module:
            pending_count = self.gs.get_pending_effect_count()
            # Use cmd_dict type for check
            act_type_str = cmd_dict.get('type')

            # Helper to check type equivalence
            is_pass = act_type_str == 'PASS'
            is_charge = act_type_str == 'MANA_CHARGE'

            # Also check against Enum if available
            if hasattr(dm_ai_module, 'ActionType'):
                if not is_pass and act_type_str == str(dm_ai_module.ActionType.PASS): is_pass = True
                if not is_charge and act_type_str == str(dm_ai_module.ActionType.MANA_CHARGE): is_charge = True

            if (is_pass or is_charge) and pending_count == 0:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)

        self.callback_update_ui()

    def check_and_handle_input_wait(self) -> bool:
        if not self.gs.waiting_for_user_input: return False

        # Stop running if auto-stepping
        if self.is_running:
            self.is_running = False

        if self.callback_input_request:
            self.callback_input_request()

        self.callback_update_ui()
        return True

    def resume_from_input(self, result: Any):
        """
        Called by UI when user input is obtained.
        result: index (int) or list of targets (list[int])
        """
        EngineCompat.EffectResolver_resume(self.gs, self.card_db, result)
        self.step_phase()

    def step_phase(self) -> None:
        """
        The main game loop step.
        """
        if self.is_processing: return
        self.is_processing = True
        try:
            if self.check_and_handle_input_wait(): return
            if self.gs.game_over:
                # self.callback_log(tr("Game Over")) # Optional
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (self.player_modes.get(active_pid) == 'Human')

            from dm_toolkit.commands import generate_legal_commands
            cmds = generate_legal_commands(self.gs, self.card_db)

            if is_human:
                resolve_cmds = []
                for c in cmds:
                    try: d = c.to_dict()
                    except: d = {}
                    if d.get('type') == 'RESOLVE_EFFECT':
                        resolve_cmds.append((c, d))

                if len(resolve_cmds) > 1:
                    # Ambiguous trigger resolution
                    pass

                if not cmds:
                    EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                    self.callback_update_ui()

                # Human turn ends here (waiting for input)
                return

            # AI Logic
            if not cmds:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            else:
                best_cmd = cmds[0]
                if best_cmd:
                    self.execute_action(best_cmd)

            self.callback_update_ui()

        finally:
            self.is_processing = False

    def generate_legal_actions(self) -> List[Any]:
        if not self.gs: return []
        from dm_toolkit.commands import generate_legal_commands
        return generate_legal_commands(self.gs, self.card_db)

    def is_game_over(self) -> bool:
        return self.gs.game_over if self.gs else False
