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
from dm_toolkit import commands
# Prefer v2 command-first wrapper
_generate_legal_commands = commands.generate_legal_commands

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None


class GameSession:
    """Simplified game session that leverages C++ engine for game progression."""
    
    # Default deck: 40 cards (ID 1-10, 4 copies each)
    DEFAULT_DECK: List[int] = [1,2,3,4,5,6,7,8,9,10]*4

    def __init__(self,
                 callback_update_ui: Optional[Callable[[], None]] = None,
                 callback_log: Optional[Callable[[str], None]] = None,
                 callback_input_request: Optional[Callable[[], None]] = None,
                 callback_action_executed: Optional[Callable[[Any], None]] = None):
        self.callback_update_ui = callback_update_ui or (lambda: None)
        self.callback_log = callback_log or (lambda m: None)
        self.callback_input_request = callback_input_request or (lambda: None)
        self.callback_action_executed = callback_action_executed or (lambda a: None)

        # NOTE: player_modes is now managed in C++ GameState.player_modes
        # This is kept for backward compatibility only
        self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
        
        self.is_running = False
        self.is_processing = False
        self.card_db: CardDB = {}
        self.native_card_db = None  # Native C++ CardDatabase
        self.game_instance = None  # C++ GameInstance
        self.gs: Optional[GameState] = None  # Alias to game_instance.state for compatibility
        self._no_action_count = 0  # Track consecutive steps with no actions

    def initialize_game(self, card_db: Optional[CardDB] = None, seed: int = 42) -> None:
        """Initialize game with C++ engine.
        
        Args:
            card_db: Optional Python dict card database (for backward compatibility, not used)
            seed: Random seed for game
        """
        if not dm_ai_module:
            self.callback_log("Error: dm_ai_module not available")
            return

        # Load native CardDatabase (always use C++ database)
        if self.native_card_db is None:
            if hasattr(dm_ai_module, 'JsonLoader'):
                try:
                    self.native_card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                    self.callback_log("Loaded native CardDatabase via JsonLoader")
                except Exception as e:
                    self.callback_log(f"ERROR: JsonLoader failed - {e}")
                    return
            else:
                self.callback_log("ERROR: JsonLoader not available")
                return
        
        # Set card_db to native_card_db for compatibility (commands.py expects card_db)
        self.card_db = self.native_card_db
        
        # Reset no-action counter
        self._no_action_count = 0
        
        # Create GameInstance (replaces GameState)
        self.game_instance = dm_ai_module.GameInstance(seed, self.native_card_db)
        self.gs = self.game_instance.state  # Alias for compatibility
        self.gs.setup_test_duel()

        # Set decks
        deck0 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        deck1 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Start game via C++ PhaseManager
        try:
            dm_ai_module.PhaseManager.start_game(self.gs, self.native_card_db)
            dm_ai_module.PhaseManager.fast_forward(self.gs, self.native_card_db)
            self.callback_log("ゲームリセット")
        except Exception as e:
            self.callback_log(f"start_game/fast_forward failed: {e}")
            import traceback
            traceback.print_exc()

        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        """Reset game state."""
        if not dm_ai_module:
            return

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
        
        # Set card_db to native_card_db for compatibility
        self.card_db = self.native_card_db

        # Reset no-action counter
        self._no_action_count = 0

        seed = random.randint(0, 100000)
        
        # Create GameInstance (not just GameState)
        self.game_instance = dm_ai_module.GameInstance(seed, self.native_card_db)
        self.gs = self.game_instance.state  # Alias for compatibility
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

        # Start game via C++ PhaseManager
        try:
            dm_ai_module.PhaseManager.start_game(self.gs, self.native_card_db)
            dm_ai_module.PhaseManager.fast_forward(self.gs, self.native_card_db)
        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def step_phase(self):
        """Alias for step_game() for backward compatibility."""
        self.step_game()

    def step_game(self):
        """
        Main game loop - FULLY delegated to C++ engine.
        
        Simply calls game_instance.step() which handles:
        - Action generation
        - AI selection (first viable action)
        - Action execution
        - State progression (fast_forward when needed)
        
        All game logic is in C++. Python only handles UI updates.
        """
        if self.is_processing or not self.gs or not self.game_instance:
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

            # Human players: need to check if it's their turn and wait for input
            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = self.gs.is_human_player(active_pid)
            
            if is_human:
                # For human players, we still need to generate actions and wait
                # C++ step() is for AI only
                cmds = _generate_legal_commands(self.gs, self.card_db)
                
                if not cmds:
                    # No actions - progress automatically
                    self._fast_forward()
                    self.callback_update_ui()
                    return
                
                # Human player - wait for input
                self.callback_update_ui()
                return

            # AI player - use C++ step() for complete automation
            success = self.game_instance.step()
            
            if not success:
                # Game might be over or stuck
                self._no_action_count += 1
                if self._no_action_count > 20:
                    self.callback_log(f"ERROR: C++ step() failed {self._no_action_count} times. Stopping.")
                    self.is_running = False
                else:
                    # Try fast_forward as fallback
                    self._fast_forward()
            else:
                # Reset counter on success
                self._no_action_count = 0
            
            self.callback_update_ui()

        finally:
            self.is_processing = False

    def execute_action(self, raw_action: Any):
        """
        Execute an action and update UI immediately.

        コマンド方式へ統一した実行経路（レガシーアクション不使用）:
          1. _ActionWrapper._action → game_instance.resolve_command()  (C++ 最優先)
          2. raw C++ CommandDef     → game_instance.resolve_command()  (ネイティブ直接)
          3. 上記失敗時のみ         → EngineCompat.ExecuteCommand()     (最終手段)
          4. 実行後は必ず fast_forward() でゲームを次の判断点まで進める

        再発防止: EngineCompat.ExecuteCommand() や CommandSystem.execute_command() を
                  直接呼ぶ経路はゲーム状態の再同期(gs = game_instance.state)と
                  fast_forward が行われないため、ここ以外で使用しないこと。
        """
        if not self.gs or not self.game_instance:
            return

        active_pid = EngineCompat.get_active_player_id(self.gs)

        # ログ用コマンド辞書（取得失敗は無視）
        cmd_dict: dict = {}
        try:
            cmd_dict = ensure_executable_command(raw_action)
        except Exception:
            pass

        executed = False
        try:
            # --- 経路 1: _ActionWrapper._action (wrap_action 経由の C++ Action) ---
            native_action = getattr(raw_action, '_action', None)
            if native_action is not None and dm_ai_module and self.game_instance:
                self.game_instance.resolve_command(native_action)
                self.gs = self.game_instance.state
                action_type = str(getattr(native_action, 'type', '')).split('.')[-1]
                self.callback_log(f"P{active_pid}: {action_type}")
                executed = True

            # --- 経路 2: raw C++ CommandDef (skip_wrapper なしで生オブジェクトが来た場合) ---
            elif dm_ai_module and self.game_instance and hasattr(dm_ai_module, 'CommandDef') and isinstance(raw_action, dm_ai_module.CommandDef):
                self.game_instance.resolve_command(raw_action)
                self.gs = self.game_instance.state
                action_type = str(getattr(raw_action, 'type', 'CMD')).split('.')[-1]
                self.callback_log(f"P{active_pid}: {action_type}")
                executed = True

            # --- 経路 3: 最終手段 (EngineCompat 経由、ゲーム状態再同期を必ず行う) ---
            else:
                EngineCompat.ExecuteCommand(self.gs, cmd_dict if cmd_dict else raw_action, self.card_db)
                if self.game_instance:
                    self.gs = self.game_instance.state
                cmd_type = cmd_dict.get('type', 'CMD') if cmd_dict else str(raw_action)
                self.callback_log(f"P{active_pid}: {cmd_type} (compat)")
                executed = True

        except Exception as e:
            self.callback_log(f"Execution error: {e}")
            self.callback_update_ui()
            return

        if not executed:
            return

        if self.callback_action_executed:
            try:
                self.callback_action_executed(cmd_dict)
            except Exception:
                pass

        # --- 実行後は必ず fast_forward で次の判断点まで進める ---
        # 再発防止: ここを省くとヒューマンターン後にゲームが止まる
        self._fast_forward()
        if self.game_instance:
            self.gs = self.game_instance.state

        self.callback_update_ui()

    def _fast_forward(self):
        """Call C++ fast_forward to progress game until next decision point."""
        if not dm_ai_module or not hasattr(dm_ai_module.PhaseManager, 'fast_forward'):
            return
        
        if self.native_card_db is None:
            self.callback_log("ERROR: Cannot call fast_forward without native CardDatabase")
            return
        
        try:
            dm_ai_module.PhaseManager.fast_forward(self.gs, self.native_card_db)
            # Re-sync gs after C++ modifies state
            if self.game_instance:
                self.gs = self.game_instance.state
        except Exception as e:
            self.callback_log(f"ERROR executing fast_forward: {e}")
            import traceback
            self.callback_log(traceback.format_exc())

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
        """Get legal commands from C++ engine.

        再発防止: skip_wrapper=True は使用しない。必ず _ActionWrapper でラップして
                  execute_action が game_instance.resolve_command() 経路を使えるようにする。
        """
        if not self.gs:
            return []
        try:
            cmds = _generate_legal_commands(self.gs, self.card_db, strict=False) or []
        except Exception:
            cmds = []
        return cmds

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.gs.game_over if self.gs else False

    def set_player_mode(self, player_id: int, mode: str):
        """Set player mode: 'Human' or 'AI'.
        
        Args:
            player_id: Player ID (0 or 1)
            mode: 'Human' or 'AI'
        """
        # Update C++ GameState
        if self.gs:
            if mode == 'Human':
                self.gs.player_modes[player_id] = dm_ai_module.PlayerMode.HUMAN
            else:
                self.gs.player_modes[player_id] = dm_ai_module.PlayerMode.AI
        
        # Update local dict for backward compatibility
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
