# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Callable, Dict, Tuple
import random
import os
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
    """
    Manages the GameState, execution loop, and phase transitions.
    Logic extracted from app.py and merged with GameController.
    """

    # デフォルトデッキをmagic.jsonから読み込む
    @staticmethod
    def load_magic_deck() -> List[int]:
        """Load the magic.json deck. Falls back to default if not found."""
        try:
            deck_path = "data/decks/magic.json"
            if not os.path.exists(deck_path):
                # Try relative to project root
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                deck_path = os.path.join(base_dir, deck_path)
            
            if os.path.exists(deck_path):
                with open(deck_path, 'r', encoding='utf-8') as f:
                    deck = json.load(f)
                    if isinstance(deck, list) and len(deck) == 40:
                        return deck
        except Exception:
            pass
        
        # Fallback to default deck
        return [1] * 40
    
    DEFAULT_DECK = load_magic_deck.__func__()

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
        self.ai_player = None
        self._load_latest_ai()

    def _load_latest_ai(self):
        if not AIPlayer: return
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
            self.ai_player = AIPlayer(latest_model, device='cpu') # Force CPU for GUI safety
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
            self.callback_log(f"P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
            self.callback_log(f"P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")

            if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                # start_gameはCardDatabaseオブジェクトを期待しているため、
                # 辞書形式のcard_dbをCardDatabaseに変換する
                if hasattr(dm_ai_module, 'JsonLoader'):
                    try:
                        # JsonLoaderを使用してCardDatabaseを取得
                        native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
                        dm_ai_module.PhaseManager.start_game(self.gs, native_db)
                        self.callback_log("start_game executed successfully")
                    except Exception as e:
                        self.callback_log(f"Warning: Failed to load CardDatabase: {e}")
                        # CardDatabaseが必要ない場合は警告のみで続行
                else:
                    # 古いバージョンでは辞書のままで動作する可能性がある
                    try:
                        dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
                        self.callback_log("start_game executed successfully (dict format)")
                    except Exception as e:
                        self.callback_log(f"Warning: start_game failed: {e}")
            
            # デバッグ: start_game実行後の状態を確認
            self.callback_log(f"After start_game - P0 deck size: {len(self.gs.players[0].deck)}, hand size: {len(self.gs.players[0].hand)}, shields: {len(self.gs.players[0].shield_zone)}")
            self.callback_log(f"After start_game - P1 deck size: {len(self.gs.players[1].deck)}, hand size: {len(self.gs.players[1].hand)}, shields: {len(self.gs.players[1].shield_zone)}")
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
                
                # Check for game over after phase transition
                if hasattr(dm_ai_module.PhaseManager, 'check_game_over'):
                    try:
                        game_over_result = dm_ai_module.PhaseManager.check_game_over(self.gs)
                        if isinstance(game_over_result, tuple):
                            is_over, winner = game_over_result
                        else:
                            is_over = game_over_result
                        
                        if is_over:
                            self.gs.game_over = True
                    except Exception as e:
                        pass  # Silent fail, logging happens in step_phase

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
            
            # Check for game over using PhaseManager
            if self.gs and dm_ai_module and hasattr(dm_ai_module.PhaseManager, 'check_game_over'):
                try:
                    game_over_result = dm_ai_module.PhaseManager.check_game_over(self.gs)
                    if isinstance(game_over_result, tuple):
                        is_over, winner = game_over_result
                    else:
                        is_over = game_over_result
                    
                    if is_over:
                        self.gs.game_over = True
                        self.callback_log(tr("Game Over - Winner: {winner}").format(winner=winner if isinstance(game_over_result, tuple) else self.gs.winner))
                        return
                except Exception as e:
                    self.callback_log(f"Warning: check_game_over failed: {e}")
            elif self.gs and self.gs.game_over:
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

                if self.ai_player:
                    try:
                        # Mask invalid actions
                        valid_indices = []
                        cmd_map = {}
                        encoder = self.ai_player.action_encoder

                        for cmd in cmds:
                            # cmd is wrapped, getattr delegates to inner action
                            idx = encoder.encode_action(cmd, self.gs, active_pid)
                            if idx != -1:
                                valid_indices.append(idx)
                                cmd_map[idx] = cmd

                        if valid_indices:
                            ai_cmd = self.ai_player.get_action(self.gs, active_pid, valid_indices)
                            # Try to map back to original command object if possible
                            ai_idx = encoder.encode_action(ai_cmd, self.gs, active_pid)
                            if ai_idx in cmd_map:
                                best_cmd = cmd_map[ai_idx]
                            else:
                                # Fallback to mapped command if ai_cmd was generated from index
                                # get_action returns a NEW GameCommand decoded from index.
                                # encode_action(ai_cmd) should return the same index.
                                pass
                                # If ai_idx not in map (which shouldn't happen if get_action respects valid_indices),
                                # check if we can execute ai_cmd directly.
                                # But best_cmd is currently cmds[0].
                                # We should update best_cmd ONLY if we found a match or trust ai_cmd.

                                # If AI returns a command that we didn't map (e.g. unencodable command chosen?),
                                # we stick to cmds[0] or execute ai_cmd if it's valid.
                                # But we masked, so it should be valid.

                                # Wait, ai_player.get_action returns a decoded command.
                                # We trust it.
                                best_cmd = ai_cmd

                        else:
                            # No commands could be encoded (e.g. complex commands not supported by simple AI)
                            # Fallback to random/first
                            pass

                    except Exception as e:
                        self.callback_log(f"AI Error: {e}")
                        # Fallback to first command
                        pass

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
