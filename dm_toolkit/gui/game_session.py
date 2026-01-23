# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Callable, Dict, Tuple
import random
import os
import sys
import json

from dm_toolkit.types import GameState, CardDB, Action
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command

# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from training.ai_player import AIPlayer
except ImportError:
    AIPlayer = None

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None


class GameSession:
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
        self.last_action = None
        self.card_db: CardDB = {}
        self.gs: Optional[GameState] = None
        self.ai_player = None

    def _load_latest_ai(self):
        if not AIPlayer:
            return
        try:
            models_dir = os.path.join(project_root, "models")
            if not os.path.exists(models_dir):
                self.callback_log("Models directory not found. Using random AI.")
                return

            files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pth')]
            if not files:
                self.callback_log("No trained models found. Using random AI.")
                return

            latest_model = max(files, key=os.path.getmtime)
            self.callback_log(f"Loading AI Model: {os.path.basename(latest_model)}")
            self.ai_player = AIPlayer(latest_model, device='cpu')
        except Exception as e:
            self.callback_log(f"Failed to load AI: {e}")
            self.ai_player = None

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        self.card_db = card_db
        if dm_ai_module:
            self.gs = dm_ai_module.GameState(seed)
            self.gs.setup_test_duel()

            # 両プレイヤーにデフォルトデッキを設定
            self.gs.set_deck(0, self.DEFAULT_DECK)
            self.gs.set_deck(1, self.DEFAULT_DECK)

            # デバッグ: デッキ設定後の状態を確認
            try:
                self.callback_log(f"P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
                self.callback_log(f"P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
            except Exception:
                pass

            # Ensure we call PhaseManager.start_game with a proper CardDatabase.
            applied = False
            if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                native_db = None
                # Prefer JsonLoader (returns native CardDatabase)
                try:
                    if hasattr(dm_ai_module, 'JsonLoader'):
                        native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                        self.callback_log("Loaded native CardDatabase via JsonLoader")
                except Exception as e:
                    self.callback_log(f"Warning: JsonLoader failed: {e}")

                # Fallback: try CardRegistry.get_all_definitions() if available
                if native_db is None:
                    try:
                        if hasattr(dm_ai_module, 'CardRegistry') and hasattr(dm_ai_module.CardRegistry, 'get_all_definitions'):
                            native_db = dm_ai_module.CardRegistry.get_all_definitions()
                            self.callback_log("Loaded native CardDatabase via CardRegistry.get_all_definitions")
                    except Exception as e:
                        self.callback_log(f"Warning: CardRegistry lookup failed: {e}")

                # Last resort: use the Python-side card_db (dict)
                if native_db is None:
                    native_db = self.card_db

                try:
                    dm_ai_module.PhaseManager.start_game(self.gs, native_db)
                    self.callback_log("start_game executed successfully")
                    applied = True
                except Exception as e:
                    self.callback_log(f"Warning: start_game failed: {e}")

            # If native start_game was not applied, perform minimal Python-level setup
            if not applied:
                try:
                    self._fallback_apply_shields_and_draw()
                except Exception as e:
                    self.callback_log(f"Fallback during initialize_game failed: {e}")

            # デバッグ: start_game実行後の状態を確認
            try:
                self.callback_log(f"After start_game - P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
                self.callback_log(f"After start_game - P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
            except Exception:
                pass
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

        # デッキが指定されていない場合はデフォルトを使用
        deck0 = p0_deck if p0_deck else self.DEFAULT_DECK
        deck1 = p1_deck if p1_deck else self.DEFAULT_DECK

        # 両方のデッキを設定（P0→P1の順で）
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Ensure PhaseManager.start_game is invoked with proper card DB (same logic as initialize_game)
        try:
            has_pm = hasattr(dm_ai_module, 'PhaseManager')
            has_start = has_pm and hasattr(dm_ai_module.PhaseManager, 'start_game')
        except Exception:
            has_pm = False
            has_start = False
        try:
            self.callback_log(f"Debug: PhaseManager present={has_pm}, has start_game={has_start}")
        except Exception:
            pass

        if has_pm and has_start:
            native_db = None
            try:
                if hasattr(dm_ai_module, 'JsonLoader'):
                    native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                    self.callback_log("Loaded native CardDatabase via JsonLoader")
            except Exception as e:
                self.callback_log(f"Warning: JsonLoader failed: {e}")

            if native_db is None:
                try:
                    if hasattr(dm_ai_module, 'CardRegistry') and hasattr(dm_ai_module.CardRegistry, 'get_all_definitions'):
                        native_db = dm_ai_module.CardRegistry.get_all_definitions()
                        self.callback_log("Loaded native CardDatabase via CardRegistry.get_all_definitions")
                except Exception as e:
                    self.callback_log(f"Warning: CardRegistry lookup failed: {e}")

            if native_db is None:
                native_db = self.card_db

            try:
                dm_ai_module.PhaseManager.start_game(self.gs, native_db)
                applied = True
            except Exception as e:
                self.callback_log(f"Warning: start_game failed during reset_game: {e}")
                applied = False
        else:
            # Fallback: if PhaseManager.start_game is not available or failed to run,
            # perform a minimal Python-level setup: place 5 shields and draw 5 cards from deck.
            try:
                if not globals().get('applied', False):
                    self._fallback_apply_shields_and_draw()
            except Exception as e:
                self.callback_log(f"Fallback deck setup failed: {e}")

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def _fallback_apply_shields_and_draw(self) -> None:
        """Perform minimal setup by moving cards from deck -> shields and deck -> hand.
        Uses dm_ai_module.DevTools.move_cards when available, otherwise manipulates lists.
        """
        # Log deck sizes before fallback
        try:
            self.callback_log(f"Pre-fallback deck sizes: P0={len(self.gs.players[0].deck)}, P1={len(self.gs.players[1].deck)}")
        except Exception:
            pass

        # If DevTools is available, use it to move cards safely
        if dm_ai_module and hasattr(dm_ai_module, 'DevTools') and hasattr(dm_ai_module, 'Zone'):
            for pid in (0, 1):
                p = self.gs.players[pid]
                # Place up to 5 shields from top (end) of deck
                for _ in range(5):
                    if not p.deck:
                        break
                    iid = p.deck[-1].instance_id
                    try:
                        dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.DECK, dm_ai_module.Zone.SHIELD)
                    except Exception:
                        pass
                # Draw up to 5 cards into hand
                for _ in range(5):
                    if not p.deck:
                        break
                    iid = p.deck[-1].instance_id
                    try:
                        dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND)
                    except Exception:
                        pass
        else:
            # DevTools not present in this binding; perform direct list moves
            for pid in (0, 1):
                p = self.gs.players[pid]
                # Shields: move up to 5 from deck -> shield_zone
                for _ in range(5):
                    if not p.deck:
                        break
                    card = p.deck.pop()
                    try:
                        p.shield_zone.append(card)
                    except Exception:
                        pass
                # Draw: move up to 5 from deck -> hand
                for _ in range(5):
                    if not p.deck:
                        break
                    card = p.deck.pop()
                    try:
                        p.hand.append(card)
                    except Exception:
                        pass

        # Log deck sizes after fallback
        try:
            self.callback_log(f"Post-fallback deck sizes: P0={len(self.gs.players[0].deck)}, P1={len(self.gs.players[1].deck)}")
        except Exception:
            pass
        self.callback_log(tr("Fallback: performed minimal deck setup (shields+draw)"))

    def set_player_mode(self, player_id: int, mode: str):
        """ mode: 'Human' or 'AI' """
        self.player_modes[player_id] = mode

    def get_player_mode(self, player_id: int) -> str:
        return self.player_modes.get(player_id, 'AI')

    def execute_action(self, action: Any) -> None:
        """
        Executes an action, wraps it, logs it, and updates UI.
        """
        if not self.gs:
            return

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
        if self.check_and_handle_input_wait():
            return

        # Auto Pass check
        if dm_ai_module and self.gs:
            try:
                pending_count = self.gs.get_pending_effect_count()
            except Exception:
                pending_count = 0
            # Use cmd_dict type for check
            act_type_str = cmd_dict.get('type')

            # Helper to check type equivalence
            is_pass = act_type_str == 'PASS'
            is_charge = act_type_str == 'MANA_CHARGE'

            # Also check against Enum if available
            if hasattr(dm_ai_module, 'ActionType'):
                try:
                    if not is_pass and act_type_str == str(dm_ai_module.ActionType.PASS):
                        is_pass = True
                    if not is_charge and act_type_str == str(dm_ai_module.ActionType.MANA_CHARGE):
                        is_charge = True
                except Exception:
                    pass

            if (is_pass or is_charge) and pending_count == 0:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)

                # Check for game over after phase transition
                if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'check_game_over'):
                    try:
                        game_over_result = dm_ai_module.PhaseManager.check_game_over(self.gs)
                        if isinstance(game_over_result, tuple):
                            is_over, winner = game_over_result
                        else:
                            is_over = game_over_result

                        if is_over:
                            self.gs.game_over = True
                    except Exception:
                        pass  # Silent fail, logging happens in step_phase

        self.callback_update_ui()

    def check_and_handle_input_wait(self) -> bool:
        if not self.gs:
            return False
        try:
            waiting = getattr(self.gs, 'waiting_for_user_input', False)
        except Exception:
            waiting = False
        if not waiting:
            return False

        # Stop running if auto-stepping
        if self.is_running:
            self.is_running = False

        if self.callback_input_request:
            try:
                self.callback_input_request()
            except Exception:
                pass

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
        if self.is_processing:
            return
        self.is_processing = True
        try:
            if self.check_and_handle_input_wait():
                return

            # Check for game over using PhaseManager
            if self.gs and dm_ai_module and hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'check_game_over'):
                try:
                    game_over_result = dm_ai_module.PhaseManager.check_game_over(self.gs)
                    if isinstance(game_over_result, tuple):
                        is_over, winner = game_over_result
                    else:
                        is_over = game_over_result

                    if is_over:
                        self.gs.game_over = True
                        try:
                            self.callback_log(tr("Game Over - Winner: {winner}").format(winner=winner if isinstance(game_over_result, tuple) else self.gs.winner))
                        except Exception:
                            pass
                        return
                except Exception as e:
                    try:
                        self.callback_log(f"Warning: check_game_over failed: {e}")
                    except Exception:
                        pass
            elif self.gs and getattr(self.gs, 'game_over', False):
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (self.player_modes.get(active_pid) == 'Human')

            from dm_toolkit.commands import generate_legal_commands
            cmds = generate_legal_commands(self.gs, self.card_db)

            if is_human:
                resolve_cmds = []
                for c in cmds:
                    try:
                        d = c.to_dict()
                    except Exception:
                        d = {}
                    if d.get('type') == 'RESOLVE_EFFECT':
                        resolve_cmds.append((c, d))

                if len(resolve_cmds) > 1:
                    pass

                if not cmds:
                    EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                    self.callback_update_ui()

                return

            # AI Logic
            if not cmds:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            else:
                best_cmd = cmds[0]

                if self.ai_player:
                    try:
                        # Mask invalid actions
                        valid_indices = []
                        cmd_map = {}
                        encoder = self.ai_player.action_encoder

                        for cmd in cmds:
                            idx = encoder.encode_action(cmd, self.gs, active_pid)
                            if idx != -1:
                                valid_indices.append(idx)
                                cmd_map[idx] = cmd

                        if valid_indices:
                            ai_cmd = self.ai_player.get_action(self.gs, active_pid, valid_indices)
                            ai_idx = encoder.encode_action(ai_cmd, self.gs, active_pid)
                            if ai_idx in cmd_map:
                                best_cmd = cmd_map[ai_idx]
                            else:
                                best_cmd = ai_cmd

                    except Exception as e:
                        try:
                            self.callback_log(f"AI Error: {e}")
                        except Exception:
                            pass

                if best_cmd:
                    self.execute_action(best_cmd)

            self.callback_update_ui()

        finally:
            self.is_processing = False

    def generate_legal_actions(self) -> List[Any]:
        if not self.gs:
            return []
        from dm_toolkit.commands import generate_legal_commands
        return generate_legal_commands(self.gs, self.card_db)

    def is_game_over(self) -> bool:
        return self.gs.game_over if self.gs else False
