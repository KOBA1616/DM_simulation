# -*- coding: utf-8 -*-
"""
Simplified GameSession that delegates game progression to C++ engine.

Key principles:
1. C++ handles all automatic game progression via fast_forward()
2. Python only handles:
   - Human player input/output
   - UI updates
   - Initial setup
3. AI player logic should ideally be in C++ or called from C++
"""
from typing import Any, List, Optional, Callable, Dict
import random
import os
import sys

from dm_toolkit.dm_types import GameState, CardDB
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.gui.i18n import tr

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None


class GameSession:
    """Simplified game session that leverages C++ engine for game progression."""
    
    DEFAULT_DECK: List[int] = []

    def __init__(self,
                 callback_update_ui: Optional[Callable[[], None]] = None,
                 callback_log: Optional[Callable[[str], None]] = None,
                 callback_input_request: Optional[Callable[[], None]] = None,
                 callback_action_executed: Optional[Callable[[Any], None]] = None):
        self.callback_update_ui = callback_update_ui or (lambda: None)
        self.callback_log = callback_log or (lambda m: None)
        self.callback_input_request = callback_input_request or (lambda: None)
        self.callback_action_executed = callback_action_executed or (lambda a: None)

        self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
        self.is_running = False
        self.is_processing = False
        self.card_db: CardDB = {}
        self.native_card_db = None  # Native C++ CardDatabase for PhaseManager
        self.gs: Optional[GameState] = None

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        """Initialize game with C++ engine."""
        self.card_db = card_db
        if not dm_ai_module:
            self.callback_log("Error: dm_ai_module not available")
            return

        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        # Set decks
        deck0 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        deck1 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Start game via C++ PhaseManager
        try:
            # Load native CardDatabase if not already loaded
            if self.native_card_db is None:
                if hasattr(dm_ai_module, 'JsonLoader'):
                    try:
                        self.native_card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                        self.callback_log("Loaded native CardDatabase via JsonLoader")
                    except Exception as e:
                        self.callback_log(f"ERROR: JsonLoader failed - {e}")
                        self.callback_log("Cannot proceed without native CardDatabase")
                        return
                else:
                    self.callback_log("ERROR: JsonLoader not available in dm_ai_module")
                    return

            dm_ai_module.PhaseManager.start_game(self.gs, self.native_card_db)
            self.callback_log("Game initialized via C++ PhaseManager.start_game")
        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        """Reset game state."""
        if not dm_ai_module:
            return

        seed = random.randint(0, 100000)
        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        deck0 = p0_deck if p0_deck else list(self.DEFAULT_DECK)
        deck1 = p1_deck if p1_deck else list(self.DEFAULT_DECK)
        
        if not deck0:
            deck0 = self._build_default_deck()
        if not deck1:
            deck1 = self._build_default_deck()

        # Set decks
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Debug: check deck sizes after set
        try:
            p0_deck_size = len(self.gs.players[0].deck)
            p1_deck_size = len(self.gs.players[1].deck)
            self.callback_log(f"P0 deck size (post-set): {p0_deck_size}")
            self.callback_log(f"P1 deck size (post-set): {p1_deck_size}")
        except Exception as e:
            self.callback_log(f"Failed to check deck sizes: {e}")

        # Start game via C++ PhaseManager
        try:
            # Load native CardDatabase if not already loaded
            if self.native_card_db is None:
                if hasattr(dm_ai_module, 'JsonLoader'):
                    try:
                        self.native_card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                        self.callback_log("Loaded native CardDatabase via JsonLoader")
                    except Exception as e:
                        self.callback_log(f"ERROR: JsonLoader failed - {e}")
                        self.callback_log("Cannot proceed without native CardDatabase")
                        return
                else:
                    self.callback_log("ERROR: JsonLoader not available in dm_ai_module")
                    return

            dm_ai_module.PhaseManager.start_game(self.gs, self.native_card_db)
            
            # Debug: check state after start_game
            try:
                p0_hand = len(self.gs.players[0].hand)
                p0_shields = len(self.gs.players[0].shield_zone)
                p0_deck = len(self.gs.players[0].deck)
                p1_hand = len(self.gs.players[1].hand)
                p1_shields = len(self.gs.players[1].shield_zone)
                p1_deck = len(self.gs.players[1].deck)
                self.callback_log(f"After start_game - P0 deck:{p0_deck}, hand:{p0_hand}, shields:{p0_shields}")
                self.callback_log(f"After start_game - P1 deck:{p1_deck}, hand:{p1_hand}, shields:{p1_shields}")
            except Exception as e:
                self.callback_log(f"Failed to check post-start_game state: {e}")

        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def step_phase(self):
        """Alias for step_game() for backward compatibility."""
        self.step_game()

    def step_game(self):
        """
        Main game loop - simplified to delegate to C++ engine.
        
        Flow:
        1. Check if game is over
        2. Get legal commands from C++ engine
        3. If human player: show options and wait for input
        4. If AI player: let C++ fast_forward handle it automatically
        """
        if self.is_processing or not self.gs:
            return

        self.is_processing = True
        try:
            # Check for user input wait state
            if self._check_and_handle_input_wait():
                return
            # Check game over
            if self.is_game_over():
                self.callback_log("Game Over")
                self.callback_update_ui()
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (self.player_modes.get(active_pid) == 'Human')

            # Get legal commands from C++ engine
            from dm_toolkit.commands import generate_legal_commands
            cmds = generate_legal_commands(self.gs, self.card_db)

            # DEBUG: Log available commands
            if cmds:
                cmd_types = [cmd.to_dict().get('type', 'UNKNOWN') for cmd in cmds[:5]]  # First 5
                self.callback_log(f"[DEBUG] P{active_pid} has {len(cmds)} commands: {cmd_types}")
            else:
                self.callback_log(f"[DEBUG] P{active_pid} has no commands - will fast_forward")

            if not cmds:
                # No commands - let C++ fast_forward progress the game
                self._fast_forward()
                self.callback_update_ui()
                return

            if is_human:
                # Human player - wait for input
                # UI will call execute_action() when user selects an action
                self.callback_update_ui()
                return

            # AI player - select and execute action
            best_cmd = self._select_ai_action(cmds)
            if best_cmd:
                self.execute_action(best_cmd)
                # After AI action, fast_forward to next decision point
                # This ensures phase progression (e.g., after MANA_CHARGE, move to main phase)
                self._fast_forward()
                self.callback_update_ui()

        finally:
            self.is_processing = False

    def execute_action(self, raw_action: Any):
        """
        Execute an action and update UI immediately.
        
        This method:
        1. Converts action to command dict
        2. Executes via C++ engine
        3. Updates UI to show the result
        
        Note: Does NOT advance the game - that's step_game()'s responsibility
        """
        if not self.gs:
            return

        try:
            # Convert to command dict
            cmd_dict = ensure_executable_command(raw_action)
        except Exception as e:
            self.callback_log(f"Command conversion error: {e}")
            return

        active_pid = EngineCompat.get_active_player_id(self.gs)
        is_human = (self.player_modes.get(active_pid) == 'Human')

        try:
            # Execute command using appropriate C++ command class
            if cmd_dict.get('type') == 'MANA_CHARGE' and dm_ai_module and hasattr(dm_ai_module, 'ManaChargeCommand'):
                instance_id = int(cmd_dict['instance_id'])
                cpp_cmd = dm_ai_module.ManaChargeCommand(instance_id)
                self.gs.execute_command(cpp_cmd)
                self.callback_log(f"P{active_pid}: MANA_CHARGE")
                
            elif (cmd_dict.get('type') == 'PLAY_FROM_ZONE' or cmd_dict.get('legacy_original_type') == 'DECLARE_PLAY') and \
                 dm_ai_module and hasattr(dm_ai_module, 'PlayCardCommand'):
                instance_id = int(cmd_dict.get('instance_id') or cmd_dict.get('source_instance_id', -1))
                cpp_cmd = dm_ai_module.PlayCardCommand(instance_id)
                self.gs.execute_command(cpp_cmd)
                self.callback_log(f"P{active_pid}: PLAY_CARD")
                
            elif cmd_dict.get('type') == 'PASS' and dm_ai_module and hasattr(dm_ai_module, 'PassCommand'):
                cpp_cmd = dm_ai_module.PassCommand()
                self.gs.execute_command(cpp_cmd)
                self.callback_log(f"P{active_pid}: PASS")
                
            else:
                # Fallback to EngineCompat
                EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)
                cmd_type = cmd_dict.get('type', 'UNKNOWN')
                self.callback_log(f"P{active_pid}: {cmd_type}")

            if self.callback_action_executed:
                self.callback_action_executed(cmd_dict)

        except Exception as e:
            self.callback_log(f"Execution error: {e}")
            self.callback_update_ui()
            return

        # Update UI immediately to show action result
        self.callback_update_ui()

    def _fast_forward(self):
        """Call C++ fast_forward to progress game until next decision point."""
        if not dm_ai_module or not hasattr(dm_ai_module.PhaseManager, 'fast_forward'):
            return
        
        if self.native_card_db is None:
            self.callback_log("ERROR: Cannot call fast_forward without native CardDatabase")
            return
        
        try:
            active_pid = EngineCompat.get_active_player_id(self.gs)
            phase = self.gs.get_phase() if hasattr(self.gs, 'get_phase') else 'UNKNOWN'
            self.callback_log(f"[DEBUG] Calling fast_forward: P{active_pid} phase={phase}")
            dm_ai_module.PhaseManager.fast_forward(self.gs, self.native_card_db)
            new_phase = self.gs.get_phase() if hasattr(self.gs, 'get_phase') else 'UNKNOWN'
            self.callback_log(f"[DEBUG] fast_forward completed: phase={new_phase}")
        except Exception as e:
            self.callback_log(f"ERROR executing fast_forward: {e}")
            import traceback
            self.callback_log(traceback.format_exc())

    def _select_ai_action(self, cmds: List[Any]) -> Any:
        """Select best action for AI player. Priority: PLAY > MANA_CHARGE > others > PASS."""
        # First pass: look for PLAY_FROM_ZONE (highest priority)
        for cmd in cmds:
            try:
                d = cmd.to_dict()
                cmd_type = d.get('type')
                if cmd_type == 'PLAY_FROM_ZONE':
                    return cmd
            except Exception:
                pass
        
        # Second pass: look for MANA_CHARGE
        for cmd in cmds:
            try:
                d = cmd.to_dict()
                cmd_type = d.get('type')
                if cmd_type == 'MANA_CHARGE':
                    return cmd
            except Exception:
                pass
        
        # Third pass: any non-PASS, non-NONE action
        for cmd in cmds:
            try:
                d = cmd.to_dict()
                cmd_type = d.get('type')
                # Skip NONE type with legacy_warning (e.g., PAY_COST)
                if cmd_type == 'NONE' and d.get('legacy_warning'):
                    continue
                if cmd_type != 'PASS':
                    return cmd
            except Exception:
                pass
        
        # If only PASS available, return first command
        return cmds[0] if cmds else None

    def _build_default_deck(self) -> List[int]:
        """Build a default deck from card_db (40 cards)."""
        if not self.card_db:
            return []
        
        ids: List[int] = []
        try:
            # dict-like
            if isinstance(self.card_db, dict):
                ids = [int(k) for k in self.card_db.keys()]
            elif isinstance(self.card_db, list):
                # cards.json style: list of dicts with 'id'
                for entry in self.card_db:
                    try:
                        if isinstance(entry, dict) and 'id' in entry:
                            ids.append(int(entry['id']))
                    except Exception:
                        continue
        except Exception:
            ids = []

        if not ids:
            return []

        # Build 40-card deck by repeating available ids
        deck: List[int] = []
        i = 0
        while len(deck) < 40:
            deck.append(ids[i % len(ids)])
            i += 1
        
        return deck

    def generate_legal_commands(self) -> List[Any]:
        """Get legal commands from C++ engine."""
        if not self.gs:
            return []
        from dm_toolkit.commands import generate_legal_commands
        return generate_legal_commands(self.gs, self.card_db)

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.gs.game_over if self.gs else False

    def set_player_mode(self, player_id: int, mode: str):
        """Set player mode: 'Human' or 'AI'."""
        self.player_modes[player_id] = mode
        self.callback_log(f"P{player_id} mode set to: {mode}")

    def toggle_auto_step(self):
        """Toggle automatic game progression."""
        self.is_running = not self.is_running
        if self.is_running:
            self.callback_log("Auto-step enabled")
            self._auto_step_loop()
        else:
            self.callback_log("Auto-step disabled")

    def _auto_step_loop(self):
        """Continuously step game until disabled or game over."""
        if not self.is_running or self.is_game_over():
            self.is_running = False
            return
        
        self.step_game()
        
        if self.is_running:
            # Schedule next step
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self._auto_step_loop)
            except Exception:
                self.is_running = False

    def _check_and_handle_input_wait(self) -> bool:
        """Check if waiting for user input and handle input request."""
        if not self.gs:
            return False
        
        try:
            waiting = getattr(self.gs, 'waiting_for_user_input', False)
        except Exception:
            waiting = False
        
        if not waiting:
            return False

        # Stop auto-stepping if running
        if self.is_running:
            self.is_running = False

        # Trigger input request callback
        if self.callback_input_request:
            try:
                self.callback_input_request()
            except Exception:
                pass

        self.callback_update_ui()
        return True

    def resume_from_input(self, result: Any):
        """Resume game after user input (for effect resolution)."""
        if not self.gs or not dm_ai_module:
            return
        
        try:
            if hasattr(dm_ai_module, 'EffectResolver') and hasattr(dm_ai_module.EffectResolver, 'resume'):
                dm_ai_module.EffectResolver.resume(self.gs, self.card_db, result)
                self._fast_forward()
        except Exception as e:
            self.callback_log(f"Resume error: {e}")
        
        self.callback_update_ui()
