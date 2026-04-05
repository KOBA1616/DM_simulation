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
from typing import Any, List, Optional, Callable
import random
import os
import sys

from dm_toolkit.dm_types import GameState, CardDB
from dm_toolkit.engine.compat import EngineCompat
# 再発防止: unified_execution は削除済み。dm_env 経由で CommandDef を操作する（Phase 3 完了）。
# 再発防止: dm_toolkit.commands.generate_legal_commands はレガシー経路。
#           コマンド生成は必ず _generate_legal_commands() を使用すること。
from python.dm_env._native import get_module as _get_dm
from python.dm_env import builders as _builders


def ensure_executable_command(cmd: Any) -> Any:
    """暫定パススルー: CommandDef または dict はそのまま返す。"""
    return cmd


def _generate_legal_commands(state: Any, card_db: Any) -> list:
    """合法コマンドを dm_ai_module.IntentGenerator.generate_legal_commands 経由で生成する。

    再発防止: generate_legal_actions（旧名）は後方互換エイリアスとして残存するが
    新規コードでは必ず generate_legal_commands を使用すること。
    """
    dm = _get_dm()
    return dm.IntentGenerator.generate_legal_commands(state, card_db)


from dm_toolkit.gui.i18n import tr

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.gui.game_controller import GameController
import threading
import time

try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


class _UiSignalBridge(QObject if _QT_AVAILABLE else object):
    """Qt シグナルブリッジ: バックグラウンドスレッドから安全にUIコールバックを呼ぶ。

    再発防止: QObject を継承したシグナルで emit すると Qt が自動的にメインスレッドの
              イベントキューへ配送するため、バックグラウンドスレッドから直接 Qt Widget を
              操作することによる 'QObject::setParent: Cannot set parent, new parent is in
              a different thread' エラーを防ぐ。
    """
    if _QT_AVAILABLE:
        ui_update_requested = pyqtSignal()


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

        # 再発防止: player_modes は C++ GameState.player_modes で管理する。
        #           Python 側のローカル dict は削除済み。set_player_mode() で
        #           gs.player_modes を直接更新すること。
        self.is_running = False
        self.is_processing = False
        self.card_db: CardDB = {}
        self.native_card_db = None  # Native C++ CardDatabase
        self.game_instance = None  # C++ GameInstance
        self.gs: Optional[GameState] = None  # Alias to game_instance.state for compatibility
        self._no_action_count = 0  # Track consecutive steps with no actions
        # Controller: encapsulate native engine interactions (responsibility split)
        # Classification: Engine/Bridge
        self.controller = GameController()
        # Async AI worker
        self._ai_thread: Optional[threading.Thread] = None
        self._ai_stop_event = threading.Event()
        # UI callback safety helper: protects against blocking or exceptions
        self._ui_callback_timeout = 0.5  # seconds to wait for UI callback before logging timeout
        # 再発防止: 緊急時のみ安全進行モード（逐次コマンド実行）を使えるようにする。
        # 既定はネイティブ本来経路（GameInstance.step/fast_forward）を使用する。
        self._safe_native_progress = os.environ.get("DM_SAFE_NATIVE_PROGRESS", "0") == "1"

        # 再発防止: Qt スレッド安全 UI 更新ブリッジ。
        #   バックグラウンドスレッドから直接 Qt Widget を操作すると
        #   'QObject::setParent: Cannot set parent, new parent is in a different thread'
        #   および 'QBasicTimer::start: QBasicTimer can only be used with threads started
        #   with QThread' が多発するため、シグナルを経由してメインスレッドのイベントキューへ
        #   配送する。_UiSignalBridge のシグナルは auto-connection でメインスレッドのスロットへ
        #   届くため、バックグラウンドからの emit でも Qt イベントループが安全に処理する。
        self._ui_bridge: Optional[_UiSignalBridge] = None
        if _QT_AVAILABLE and callable(callback_update_ui):
            try:
                self._ui_bridge = _UiSignalBridge()
                self._ui_bridge.ui_update_requested.connect(self.callback_update_ui)
            except Exception:
                self._ui_bridge = None

    def initialize_game(self, card_db: Optional[CardDB] = None, seed: int = 42) -> None:
        """Initialize game with C++ engine.
        
        Args:
            card_db: Optional Python dict card database (for backward compatibility, not used)
            seed: Random seed for game
        """
        if not dm_ai_module:
            self.callback_log(tr("Error: dm_ai_module not available"))
            return

        # Load native CardDatabase (always use C++ database)
        if self.native_card_db is None:
            if hasattr(dm_ai_module, 'JsonLoader'):
                try:
                    # 再発防止: JsonLoader にファイルパスを直接渡すと access violation が
                    # 発生するケースがあるため、EngineCompat 経由の安全ローダを使う。
                    self.native_card_db = EngineCompat.JsonLoader_load_cards("data/cards.json")
                    self.callback_log(tr("Loaded native CardDatabase via JsonLoader"))
                except Exception as e:
                    self.callback_log(tr("ERROR: JsonLoader failed - {e}").format(e=e))
                    return
            else:
                self.callback_log(tr("ERROR: JsonLoader not available"))
                return
        
        # Set card_db to native_card_db for compatibility (commands.py expects card_db)
        self.card_db = self.native_card_db
        
        # Reset no-action counter
        self._no_action_count = 0
        
        # Create GameInstance (replaces GameState)
        # Use GameController to create and hold GameInstance
        self.game_instance = self.controller.create_instance(seed, self.native_card_db)
        self.gs = self.game_instance.state  # Alias for compatibility
        self.gs.setup_test_duel()

        # Set decks
        deck0 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        deck1 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Start game via C++ PhaseManager (delegated to controller)
        try:
            # 再発防止: 初期化直後に fast_forward すると自動で盤面が進み、
            # 「読み込み直後の初期状態」ではなくなるため start_game のみに限定する。
            self.controller.start_game_only(self.gs, self.native_card_db)
            self.callback_log(tr("Game Reset"))
        except Exception as e:
            self.callback_log(tr("start_game/fast_forward failed: {e}").format(e=e))
            import traceback
            traceback.print_exc()

        self._safe_callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        """Reset game state."""
        if not dm_ai_module:
            return

        # Load native CardDatabase if not already loaded
        if self.native_card_db is None:
            if hasattr(dm_ai_module, 'JsonLoader'):
                try:
                    # 再発防止: initialize_game と同様に安全ローダ経由で読み込む。
                    self.native_card_db = EngineCompat.JsonLoader_load_cards("data/cards.json")
                    self.callback_log(tr("Loaded native CardDatabase via JsonLoader"))
                except Exception as e:
                    self.callback_log(tr("ERROR: JsonLoader failed - {e}").format(e=e))
                    self.callback_log(tr("Cannot proceed without native CardDatabase"))
                    return
            else:
                self.callback_log(tr("ERROR: JsonLoader not available in dm_ai_module"))
                return
        
        # Set card_db to native_card_db for compatibility
        self.card_db = self.native_card_db

        # Reset no-action counter
        self._no_action_count = 0

        seed = random.randint(0, 100000)
        
        # Create GameInstance (not just GameState)
        self.game_instance = self.controller.create_instance(seed, self.native_card_db)
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
            # 再発防止: reset 直後に fast_forward してしまうと、
            # デュエマ初期盤面確認前に進行してしまうため start_game のみにする。
            self.controller.start_game_only(self.gs, self.native_card_db)
        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_log(tr("Game Reset"))
        self._safe_callback_update_ui()

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
            # Check game over  
            if self.is_game_over():
                self.callback_log("Game Over")
                self._safe_callback_update_ui()
                return

            # Human players: need to check if it's their turn and wait for input
            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = self.gs.is_human_player(active_pid)

            if is_human:
                # 再発防止: _check_and_handle_input_wait はヒューマンプレイヤー専用。
                #           AI プレイヤー時に呼ぶと waiting_for_user_input=True で無限ブロックになる。
                #           game_instance.step() が SELECT_NUMBER 等を含めてすべて処理するため
                #           AI 側では _check_and_handle_input_wait を呼ばないこと。
                if self._check_and_handle_input_wait():
                    return
                # For human players, we still need to generate actions and wait
                # C++ step() is for AI only
                cmds = _generate_legal_commands(self.gs, self.card_db)
                
                if not cmds:
                    # No actions - progress automatically
                    self._fast_forward()
                    self._safe_callback_update_ui()
                    return
                
                # Human player - wait for input
                self._safe_callback_update_ui()
                return

            if self._safe_native_progress:
                # 再発防止: native step() を避け、合法コマンドを逐次実行して進行する。
                cmds = _generate_legal_commands(self.gs, self.card_db)
                if not cmds:
                    self._safe_advance_phase()
                    self._safe_callback_update_ui()
                    return

                self.execute_command(cmds[0])
                return

            # AI player - use C++ step() for complete automation
            success = self.game_instance.step()
            
            if not success:
                # Game might be over or stuck
                self._no_action_count += 1
                if self._no_action_count > 20:
                    self.callback_log(tr("ERROR: C++ step() failed {count} times. Stopping.").format(count=self._no_action_count))
                    self.is_running = False
                else:
                    # Try fast_forward as fallback
                    self._fast_forward()
            else:
                # Reset counter on success
                self._no_action_count = 0
            
            self._safe_callback_update_ui()

        except Exception as e:
            # 再発防止: step() 由来の例外が UI スレッドまで伝播すると
            # ゲーム進行中にアプリ全体が停止するため、ここで捕捉して安全停止する。
            self.callback_log(tr("ERROR: step_game failed: {e}").format(e=e))
            self.is_running = False
            try:
                self._safe_callback_update_ui()
            except Exception:
                pass

        finally:
            self.is_processing = False

    def execute_command(self, raw_action: Any):
        """
        Execute a command and update UI immediately.

        コマンド方式へ統一した実行経路（レガシーアクション不使用）:
          1. _CommandWrapper._action → game_instance.resolve_command()  (C++ 最優先)
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
            # --- 経路 1: _CommandWrapper._action (wrap_action 経由の C++ CommandDef) ---
            # 再発防止: _CommandWrapper は旧 _ActionWrapper の後継。_action 属性でネイティブオブジェクト保持。
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
            self._safe_callback_update_ui()
            return

        if not executed:
            return

        if self.callback_action_executed:
            try:
                self.callback_action_executed(cmd_dict)
            except Exception:
                pass

        # --- 実行後の進行 ---
        # 再発防止: 安全進行モードでは fast_forward を使わない。
        # step() ループ側で合法コマンドが無いときに段階的にフェーズを進める。
        if not self._safe_native_progress:
            self._fast_forward()
        if self.game_instance:
            self.gs = self.game_instance.state

        self._safe_callback_update_ui()

    # 再発防止: execute_action は削除済み。execute_command を使用すること。

    def _fast_forward(self):
        """Call C++ fast_forward to progress game until next decision point."""
        if self._safe_native_progress:
            self.callback_log("INFO: fast_forward skipped in safe native progression mode")
            return
        # Delegate to controller which guards PhaseManager calls
        if self.native_card_db is None:
            self.callback_log(tr("ERROR: Cannot call fast_forward without native CardDatabase"))
            return
        try:
            self.controller.fast_forward(self.gs, self.native_card_db)
            # Re-sync gs after C++ modifies state
            if self.game_instance:
                self.gs = self.game_instance.state
        except Exception as e:
            self.callback_log(f"ERROR executing fast_forward: {e}")
            import traceback
            self.callback_log(traceback.format_exc())

    def _safe_advance_phase(self) -> None:
        """Advance one phase without using fast_forward.

        Recurrence prevention: this is the low-risk alternative used when
        fast_forward/step are unstable on native builds.
        """
        if not dm_ai_module or self.gs is None:
            return
        if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'advance_phase'):
            try:
                dm_ai_module.PhaseManager.advance_phase(self.gs)
            except Exception as e:
                self.callback_log(f"ERROR executing advance_phase: {e}")

    def _safe_callback_update_ui(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """UI コールバックをスレッドセーフに呼び出す。

        再発防止: バックグラウンドスレッドから Qt Widget を直接操作すると
                  'QObject::setParent: Cannot set parent, new parent is in a different thread'
                  エラーが多発する。_UiSignalBridge でシグナルを emit し、Qt の auto-connection
                  でメインスレッドへ配送することでスレッド安全性を確保する。

        Args:
            wait: 現在は使用されない（後方互換のため残存）。
                  シグナル経由の非同期配送に統一したため wait/timeout は意味を持たない。
            timeout: 現在は使用されない（後方互換のため残存）。
        """
        if not callable(self.callback_update_ui):
            return

        # Qt シグナルブリッジが利用可能かつ QApplication が動作中の場合はスレッドセーフに emit する
        if self._ui_bridge is not None:
            try:
                from PyQt6.QtWidgets import QApplication
                if QApplication.instance() is not None:
                    self._ui_bridge.ui_update_requested.emit()
                    return
            except Exception:
                pass

        # フォールバック: Qt 未使用環境ではメインスレッドからの直接呼び出しのみ安全
        try:
            self.callback_update_ui()
        except Exception as e:
            try:
                self.callback_log(f"UI callback error: {e}")
            except Exception:
                pass

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

        再発防止: skip_wrapper=True は使用しない。必ず _CommandWrapper でラップして
                  execute_action が game_instance.resolve_command() 経路を使えるようにする。
        再発防止: _generate_legal_commands は strict キーワード引数を持たない。
                  strict=False 等のキーワード引数を渡すと TypeError でサイレントに [] が返る。
        """
        if not self.gs:
            return []
        try:
            cmds = _generate_legal_commands(self.gs, self.card_db) or []
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
        # If active player is AI, run worker thread to avoid blocking UI loop
        try:
            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = self.gs.is_human_player(active_pid)
        except Exception:
            is_human = False

        if not is_human:
            # Start background AI worker if not already running
            if self._ai_thread is None or not self._ai_thread.is_alive():
                self._start_ai_worker()
            return

        # Human player: perform single step and schedule next GUI callback
        self.step_game()
        if self.is_running:
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self._auto_step_loop)
            except Exception:
                self.is_running = False

    def _start_ai_worker(self) -> None:
        """Start a daemon thread that repeatedly calls engine step() for AI turns."""
        if not self.game_instance:
            return
        # Clear previous stop event
        self._ai_stop_event.clear()

        def worker():
            while not self._ai_stop_event.is_set() and not self.is_game_over():
                try:
                    # call the native step() which performs AI selection/execution
                    success = False
                    try:
                        success = self.game_instance.step()
                    except Exception:
                        success = False

                    # Always resync state and notify UI
                    if self.game_instance:
                        self.gs = self.game_instance.state
                    # Use non-blocking safe callback to avoid worker stall
                    try:
                        self._safe_callback_update_ui(wait=False)
                    except Exception:
                        pass

                    if not success:
                        # fallback fast-forward and small backoff
                        try:
                            self._fast_forward()
                        except Exception:
                            pass
                        time.sleep(0.05)
                    else:
                        time.sleep(0.01)
                except Exception:
                    # On unexpected errors, stop worker to avoid runaway thread
                    break

        t = threading.Thread(target=worker, daemon=True)
        self._ai_thread = t
        t.start()

    def _stop_ai_worker(self) -> None:
        """Signal AI worker to stop and join thread if possible."""
        try:
            self._ai_stop_event.set()
        except Exception:
            pass
        if self._ai_thread and self._ai_thread.is_alive():
            # give it a short time to exit
            self._ai_thread.join(timeout=0.2)
        self._ai_thread = None

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

        self._safe_callback_update_ui()
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
        
        self._safe_callback_update_ui()
