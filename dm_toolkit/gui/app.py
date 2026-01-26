# -*- coding: utf-8 -*-
import sys
import os
import random
import json
import csv
from pathlib import Path
from typing import Any, List, Optional, Dict, cast
from types import ModuleType

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QDockWidget, QTabWidget,
    QInputDialog, QToolBar, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit, QDialogButtonBox, QTextEdit
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer, QProcess

dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.dm_types import GameState, CardDB, Action, PlayerID, CardID
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command
from dm_toolkit.gui.deck_builder import DeckBuilder
from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.log_viewer import LogViewer
from dm_toolkit.gui.game_session import GameSession
from dm_toolkit.gui.input_handler import GameInputHandler
from dm_toolkit.gui.layout_builder import LayoutBuilder

class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("DM AI Simulator"))
        self.resize(1600, 900)

        # Ensure log list exists before any callbacks
        self.log_viewer = LogViewer()

        # Initialize GameSession (Logic)
        self.session = GameSession(
            callback_update_ui=self.update_ui,
            callback_log=self.log_message,
            callback_input_request=self.handle_user_input_request,
            callback_action_executed=self.on_action_executed
        )

        # Initialize Input Handler
        self.input_handler = GameInputHandler(self, self.session)

        # Load card database using unified robust loader
        # (Prioritizes Python dict for GUI, caches Native DB for EngineCompat wrappers)
        try:
            self.card_db = EngineCompat.load_cards_robust("data/cards.json")
            if isinstance(self.card_db, list):
                self.card_db = {card['id']: card for card in self.card_db if isinstance(card, dict) and 'id' in card}
        except Exception:
            self.card_db = {}
        
        self.p0_deck_ids: Optional[List[int]] = None
        self.p1_deck_ids: Optional[List[int]] = None
        self.last_command_index: int = 0

        # Initialize Game
        self.session.initialize_game(self.card_db)

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.session.step_phase)
        # Note: Session manages 'is_processing', App manages 'is_running' for timer
        self.is_running: bool = False

        # Initialize Layout using Builder
        self.layout_builder = LayoutBuilder(self)
        self.layout_builder.build()
        # Training status toolbar
        try:
            self._init_training_toolbar()
        except Exception:
            pass
        self.update_ui()
        # For test operation: auto-apply the 'standard_start' scenario if present
        try:
            if hasattr(self, 'scenario_tools') and getattr(self, 'scenario_tools') is not None:
                for sc in getattr(self.scenario_tools, 'scenarios', []):
                    if sc and sc.get('name') == 'standard_start':
                        # Apply scenario (this will reset game and set zones accordingly)
                        self.scenario_tools.apply_scenario(sc)
                        self.log_viewer.log_message(tr("Applied 'standard_start' scenario on startup"))
                        break
        except Exception as e:
            self.log_viewer.log_message(f"Failed to auto-apply standard scenario: {e}")
        # Load and auto-deploy a default deck on startup (shuffle then reset)
        try:
            repo_root = os.path.join(os.getcwd())
            meta_decks = os.path.join(repo_root, 'data', 'meta_decks.json')
            if os.path.exists(meta_decks):
                try:
                    with open(meta_decks, 'r', encoding='utf-8') as mf:
                        md = json.load(mf)
                    decks = md.get('decks', []) if isinstance(md, dict) else []
                    if decks:
                        first = decks[0]
                        cards = list(first.get('cards', [])) if isinstance(first, dict) else []
                        # normalize to 40 cards (truncate or repeat as needed)
                        if cards:
                            d0 = cards[:40]
                            if len(d0) < 40:
                                # repeat sequence to fill
                                while len(d0) < 40:
                                    d0.extend(cards[:(40 - len(d0))])
                            d1 = list(d0)
                            random.shuffle(d0)
                            random.shuffle(d1)
                            self.p0_deck_ids = d0
                            self.p1_deck_ids = d1
                            self.reset_game()
                            try:
                                self.log_viewer.log_message(tr('Loaded default deck and deployed (shuffled) on startup'))
                            except Exception:
                                pass
                except Exception as e:
                    try:
                        self.log_viewer.log_message(f"Failed loading default deck: {e}")
                    except Exception:
                        pass
        except Exception:
            pass
        self.showMaximized()

    @property
    def gs(self):
        return self.session.gs

    def log_message(self, msg: str) -> None:
        self.log_viewer.log_message(msg)

    def on_action_executed(self, action_dict: Dict[str, Any]) -> None:
        """Callback from GameSession when an action is executed."""
        if hasattr(self, 'loop_recorder'):
            self.loop_recorder.record_action(str(action_dict))
        if hasattr(self, 'scenario_tools'):
            self.scenario_tools.record_action(str(action_dict))

    def open_deck_builder(self) -> None:
        self.deck_builder = DeckBuilder(self.card_db)
        self.deck_builder.show()

    def open_card_editor(self) -> None:
        self.card_editor = CardEditor("data/cards.json")
        self.card_editor.data_saved.connect(self.reload_card_data)
        self.card_editor.show()

    def toggle_native_db(self) -> None:
        """Toggles the native CardDatabase usage in EngineCompat."""
        new_state = not getattr(EngineCompat, '_native_enabled', True)
        EngineCompat.set_native_enabled(new_state)

        msg = tr("Native CardDatabase Enabled") if new_state else tr("Native CardDatabase Disabled")
        self.log_viewer.log_message(msg)

        # Reload to refresh cache state if enabling
        self.reload_card_data()

        QMessageBox.information(self, tr("Native DB"), msg)

    def reload_card_data(self) -> None:
        try:
            loaded = EngineCompat.load_cards_robust("data/cards.json")
            if loaded:
                self.card_db = loaded
                # Ensure card_db is always a dict
                if isinstance(self.card_db, list):
                    self.card_db = {card['id']: card for card in self.card_db if isinstance(card, dict) and 'id' in card}

                if hasattr(self, 'deck_builder') and self.deck_builder.isVisible():
                    self.deck_builder.reload_database()
                if hasattr(self, 'scenario_tools'):
                    self.scenario_tools.set_game_state(self.gs, self.card_db)
                self.log_viewer.log_message(tr("Card Data Reloaded from Editor Save"))
            else:
                 raise Exception("load_cards_robust returned empty/None")
        except Exception as e:
            self.log_viewer.log_message(f"{tr('Error reloading cards')}: {e}")
            QMessageBox.critical(self, tr("Reload Failed"), f"{tr('Failed to reload card data')}:\n{e}")

    def toggle_scenario_mode(self, checked: bool) -> None:
        if not hasattr(self, 'scenario_tools'): return
        if checked:
            self.scenario_tools.show()
            self.log_viewer.log_message(tr("Scenario Mode Enabled"))
            if self.is_running:
                self.toggle_simulation()
        else:
            self.scenario_tools.hide()
            self.log_viewer.log_message(tr("Scenario Mode Disabled"))

    def open_simulation_dialog(self) -> None:
        self.sim_dialog = SimulationDialog(self.card_db, self, self.p0_deck_ids, self.p1_deck_ids)
        self.sim_dialog.show()

    def load_deck_p0(self) -> None:
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(self, tr("Load Deck P0"), "data/decks", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f: deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                # Shuffle deck on load so placement/draw are randomized immediately
                try:
                    random.shuffle(deck_ids)
                except Exception:
                    pass
                self.p0_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P0')}: {os.path.basename(fname)} (shuffled)")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def load_deck_p1(self) -> None:
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(self, tr("Load Deck P1"), "data/decks", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f: deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                # Shuffle deck on load so placement/draw are randomized immediately
                try:
                    random.shuffle(deck_ids)
                except Exception:
                    pass
                self.p1_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P1')}: {os.path.basename(fname)} (shuffled)")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def show_help(self) -> None:
        QMessageBox.information(self, tr("Help / Manual"), tr("Help text..."))

    def toggle_simulation(self) -> None:
        if self.is_running:
            self.timer.stop()
            if hasattr(self, 'control_panel'):
                self.control_panel.set_start_button_text(tr("Start Sim"))
            self.is_running = False
        else:
            self.timer.start(500)
            if hasattr(self, 'control_panel'):
                self.control_panel.set_start_button_text(tr("Stop Sim"))
            self.is_running = True

    def reset_game(self) -> None:
        self.timer.stop()
        self.is_running = False
        self.input_handler.selected_targets = []
        if hasattr(self, 'control_panel'):
            self.control_panel.set_confirm_button_visible(False)
            self.control_panel.set_pass_button_visible(False)
            self.control_panel.set_start_button_text(tr("Start Sim"))
        self.log_viewer.clear_logs()

        self.session.reset_game(self.p0_deck_ids, self.p1_deck_ids)

        if hasattr(self, 'scenario_tools'):
            self.scenario_tools.set_game_state(self.gs, self.card_db)
        self.last_command_index = 0
        self.update_ui()

    def pass_turn(self) -> None:
        if hasattr(self, 'current_pass_action') and self.current_pass_action:
            self.session.execute_action(self.current_pass_action)

    def confirm_selection(self) -> None:
        self.input_handler.confirm_selection()

    def on_card_clicked(self, card_id: int, instance_id: int) -> None:
        self.input_handler.on_card_clicked(card_id, instance_id)

    def on_card_double_clicked(self, card_id: int, instance_id: int) -> None:
        self.input_handler.on_card_double_clicked(card_id, instance_id)

    def on_card_hovered(self, card_id: int) -> None:
        if card_id >= 0:
            card_data = self._get_card_by_id(card_id)
            if card_data and hasattr(self, 'card_detail_panel'):
                self.card_detail_panel.update_card(card_data)

    def _get_card_by_id(self, card_id: int):
        db = self.card_db
        if db is None: return None
        try:
            if isinstance(db, dict): return db.get(card_id)
        except Exception: pass
        try:
            if hasattr(db, '__contains__') and card_id in db: return db[card_id]
        except Exception: pass
        for meth in ('get_card', 'get_by_id'):
            try:
                fn = getattr(db, meth, None)
                if callable(fn): return fn(card_id)
            except Exception: pass
        try:
            for k in getattr(db, 'keys', lambda: [])():
                if int(k) == int(card_id): return db[k]
        except Exception: pass
        return None

    def on_resolve_effect_from_stack(self, index: int) -> None:
        self.input_handler.on_resolve_effect_from_stack(index)

    def handle_user_input_request(self) -> None:
        """Called by GameSession when input is needed."""
        self.input_handler.handle_user_input_request()

    def update_ui(self) -> None:
        if self.gs is None:
            return

        # 1. Update Game Status
        if hasattr(self, 'game_status_widget'):
            self.game_status_widget.update_state(self.gs)
        
        # 2. Update Tools
        if hasattr(self, 'stack_view'):
            self.stack_view.update_state(self.gs, self.card_db)
        if hasattr(self, 'effect_debugger'):
            self.effect_debugger.update_state(self.gs, self.card_db)

        # 3. Update Logs
        history = EngineCompat.get_command_history(self.gs)
        self.last_command_index = self.log_viewer.update_from_history(
            history, self.last_command_index, self.gs, self.card_db
        )

        # 4. Generate Legal Actions & Update Game Board
        class _P:
             hand: List[Any] = []
             mana_zone: List[Any] = []
             battle_zone: List[Any] = []
             shield_zone: List[Any] = []
             graveyard: List[Any] = []
             deck: List[Any] = []

        active_pid = EngineCompat.get_active_player_id(self.gs)
        p0 = EngineCompat.get_player(self.gs, 0) or _P()
        p1 = EngineCompat.get_player(self.gs, 1) or _P()
        
        # Obtain all legal commands from engine (useful to detect PASS even
        # when UI filters actions by active player). For performance this is
        # lightweight and avoids edge cases where PASS exists but the
        # per-player filtered list is empty.
        all_legal_actions = []
        try:
            # Use EngineCompat wrapper to handle Dict->NativeDB swapping
            all_legal_actions = EngineCompat.ActionGenerator_generate_legal_commands(self.gs, self.card_db)
        except Exception:
            all_legal_actions = []

        legal_actions = []
        is_human = False
        if hasattr(self, 'control_panel'):
            is_human = self.control_panel.is_p0_human()

        if active_pid == 0 and is_human and not self.gs.game_over:
             legal_actions = all_legal_actions

        # Determine if a PASS action exists in the full engine-provided set
        # (not only the filtered per-player list). This prevents missing a
        # pass action when view/filtering logic hides other commands.
        self.current_pass_action = None
        for cmd in all_legal_actions:
            try: d = cmd.to_dict()
            except: d = {}
            if d.get('type') == 'PASS' or d.get('legacy_original_type') == 'PASS':
                self.current_pass_action = cmd
                break

        god_view = False
        if hasattr(self, 'control_panel'):
            god_view = self.control_panel.is_god_view()

        if hasattr(self, 'game_board'):
            self.game_board.update_state(p0, p1, self.card_db, legal_actions, god_view)
            if EngineCompat.is_waiting_for_user_input(self.gs):
                 self.game_board.set_selection_mode(self.input_handler.selected_targets)

        # 5. Update Control Panel
        if hasattr(self, 'control_panel'):
            pending = EngineCompat.get_pending_query(self.gs)
            self.control_panel.update_state(
                can_pass=bool(self.current_pass_action),
                is_waiting_input=EngineCompat.is_waiting_for_user_input(self.gs),
                pending_query=pending,
                selected_count=len(self.input_handler.selected_targets)
            )

    # --- Trainer integration ---
    def _init_training_toolbar(self) -> None:
        self.toolbar = QToolBar(tr("Training"))
        start_action = QAction(tr("Start Trainer"), self)
        start_action.triggered.connect(self._toggle_trainer)
        self.toolbar.addAction(start_action)
        # Watch latest match (1 game, 1 parallel)
        watch_action = QAction(tr("Watch Latest Match"), self)
        watch_action.setToolTip(tr("観戦: 最新モデル同士を1ゲームで実行して進行を表示します"))
        watch_action.triggered.connect(self._toggle_h2h_watch)
        self.toolbar.addAction(watch_action)
        # Settings action
        settings_action = QAction(tr("Training 設定"), self)
        settings_action.setToolTip(tr("トレーニングのハイパーパラメータを設定します"))
        settings_action.triggered.connect(self._show_training_settings_dialog)
        self.toolbar.addAction(settings_action)

        self.training_status_label = QLabel(tr("Trainer: stopped"))
        self.toolbar.addWidget(self.training_status_label)

        self.addToolBar(self.toolbar)

        # QProcess and buffers
        self.trainer_proc: QProcess | None = None
        self._trainer_stdout_buf: str = ""
        # head2head watch process and UI
        self.h2h_watch_proc: QProcess | None = None
        self._h2h_stdout_buf: str = ""
        self.h2h_dock: QDockWidget | None = None
        self.h2h_text: QTextEdit | None = None
        self.h2h_status_label: QLabel | None = None
        # Default trainer config (日本語ラベルに対応)
        self.trainer_config = {
            'data_path': os.path.join('data', 'transformer_training_data.npz'),
            'batch_size': 8,
            'epochs': 1,
            'lr': 1e-4,
            'log_dir': os.path.join('logs', 'transformer'),
            'checkpoint_dir': os.path.join('checkpoints', 'transformer'),
            'checkpoint_freq': 5000,
            # head2head eval defaults
            'eval_every_steps': 0,
            'eval_games': 50,
            'eval_parallel': 8,
            'eval_use_pytorch': False,
            'eval_baseline': ''
        }

    def _toggle_trainer(self) -> None:
        if self.trainer_proc is None or self.trainer_proc.state() == QProcess.ProcessState.NotRunning:
            self._start_trainer()
        else:
            self._stop_trainer()

    def _start_trainer(self) -> None:
        script_path = os.path.join(os.getcwd(), "python", "training", "train_transformer_phase4.py")
        if not os.path.exists(script_path):
            self.log_viewer.log_message(f"Trainer script not found: {script_path}")
            return

        self.trainer_proc = QProcess(self)
        self.trainer_proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.trainer_proc.readyReadStandardOutput.connect(self._on_trainer_stdout)
        self.trainer_proc.readyReadStandardError.connect(self._on_trainer_stderr)
        # Build args from trainer_config
        cfg = self.trainer_config
        # Try to merge SimulationDialog saved settings if available
        try:
            repo_root = os.path.join(os.getcwd())
            sim_settings_path = os.path.join(repo_root, 'data', 'sim_settings.json')
            if os.path.exists(sim_settings_path):
                try:
                    with open(sim_settings_path, 'r', encoding='utf-8') as f:
                        sim_s = json.load(f)
                        # map sim settings to trainer eval config if present
                        if 'h2h_games' in sim_s:
                            cfg['eval_games'] = int(sim_s.get('h2h_games', cfg.get('eval_games')))
                        if 'h2h_parallel' in sim_s:
                            cfg['eval_parallel'] = int(sim_s.get('h2h_parallel', cfg.get('eval_parallel')))
                        if 'h2h_use_pytorch' in sim_s:
                            cfg['eval_use_pytorch'] = bool(sim_s.get('h2h_use_pytorch', cfg.get('eval_use_pytorch')))
                        if 'h2h_baseline' in sim_s and sim_s.get('h2h_baseline'):
                            cfg['eval_baseline'] = str(sim_s.get('h2h_baseline'))
                        # auto-eval settings from SimulationDialog
                        if 'eval_every_steps' in sim_s:
                            cfg['eval_every_steps'] = int(sim_s.get('eval_every_steps', cfg.get('eval_every_steps', 0)))
                        if 'eval_games' in sim_s:
                            cfg['eval_games'] = int(sim_s.get('eval_games', cfg.get('eval_games', 50)))
                        if 'eval_parallel' in sim_s:
                            cfg['eval_parallel'] = int(sim_s.get('eval_parallel', cfg.get('eval_parallel', 8)))
                        if 'eval_use_pytorch' in sim_s:
                            cfg['eval_use_pytorch'] = bool(sim_s.get('eval_use_pytorch', cfg.get('eval_use_pytorch', False)))
                except Exception:
                    pass
        except Exception:
            pass
        args = [script_path,
                '--data_path', str(cfg.get('data_path', 'data/transformer_training_data.npz')),
                '--batch_size', str(cfg.get('batch_size', 8)),
                '--epochs', str(cfg.get('epochs', 1)),
                '--lr', str(cfg.get('lr', 1e-4)),
                '--log_dir', str(cfg.get('log_dir', 'logs/transformer')),
                '--checkpoint_dir', str(cfg.get('checkpoint_dir', 'checkpoints/transformer')),
                '--checkpoint_freq', str(cfg.get('checkpoint_freq', 5000))]
        # Start using same python executable
        self.trainer_proc.start(sys.executable, args)
        self.training_status_label.setText(tr("Trainer: starting"))
        self.log_viewer.log_message(tr("Trainer started."))

    def _stop_trainer(self) -> None:
        if not self.trainer_proc:
            return
        try:
            self.trainer_proc.terminate()
        except Exception:
            try:
                self.trainer_proc.kill()
            except Exception:
                pass
        self.training_status_label.setText(tr("Trainer: stopped"))
        self.log_viewer.log_message(tr("Trainer stopped."))
        self.trainer_proc = None

    def _on_trainer_stderr(self) -> None:
        if not self.trainer_proc:
            return
        data = bytes(self.trainer_proc.readAllStandardError()).decode('utf-8', errors='replace')
        for line in data.splitlines():
            self.log_viewer.log_message(f"TRAINER_ERR: {line}")

    def _on_trainer_stdout(self) -> None:
        if not self.trainer_proc:
            return
        data = bytes(self.trainer_proc.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not data:
            return
        self._trainer_stdout_buf += data
        lines = self._trainer_stdout_buf.splitlines(keepends=True)
        complete_lines = []
        # Extract complete lines (ending with newline)
        for ln in lines:
            if ln.endswith('\n') or ln.endswith('\r'):
                complete_lines.append(ln)
            else:
                # partial line - keep in buffer
                pass
        if complete_lines:
            # remove processed part from buffer
            processed = ''.join(complete_lines)
            self._trainer_stdout_buf = self._trainer_stdout_buf[len(processed):]
            for ln in complete_lines:
                text = ln.strip()
                self.log_viewer.log_message(f"TRAINER_OUT: {text}")
                # Detect TRAIN_JSON lines
                marker = 'TRAIN_JSON:'
                if marker in text:
                    try:
                        idx = text.index(marker) + len(marker)
                        payload = text[idx:].strip()
                        hb = json.loads(payload)
                        # Update status label concisely
                        em = f"Epoch {hb.get('epoch')} Step {hb.get('global_step')} ETA {hb.get('eta_minutes')}m"
                        self.training_status_label.setText(em)
                    except Exception as e:
                        self.log_viewer.log_message(f"Failed parsing TRAIN_JSON: {e}")

        # --- head2head watch ---
        def _toggle_h2h_watch(self) -> None:
            if self.h2h_watch_proc is None or self.h2h_watch_proc.state() == QProcess.ProcessState.NotRunning:
                self._start_h2h_watch()
            else:
                self._stop_h2h_watch()

        def _start_h2h_watch(self) -> None:
            repo_root = os.path.join(os.getcwd())
            h2h_script = os.path.join(repo_root, 'training', 'head2head.py')
            if not os.path.exists(h2h_script):
                self.log_viewer.log_message(f"head2head script not found: {h2h_script}")
                QMessageBox.warning(self, tr("Error"), tr("head2head script not found."))
                return

            # find two latest models in checkpoints/transformer
            ck_dir = os.path.join(repo_root, 'checkpoints', 'transformer')
            models = []
            try:
                if os.path.exists(ck_dir):
                    for fn in os.listdir(ck_dir):
                        if fn.endswith('.pth') or fn.endswith('.onnx'):
                            if 'step_' in fn:
                                models.append(os.path.join(ck_dir, fn))
            except Exception:
                models = []

            def step_key(fn: str):
                try:
                    parts = Path(fn).stem.split('_')
                    if 'step' in parts:
                        i = parts.index('step')
                        return int(parts[i+1])
                except Exception:
                    pass
                return 0

            models = sorted(models, key=step_key, reverse=True)
            if len(models) < 2:
                self.log_viewer.log_message(tr('Not enough models found in checkpoints/transformer to watch latest match'))
                QMessageBox.information(self, tr('Info'), tr('最新チェックポイントが2つ以上必要です (checkpoints/transformer)'))
                return

            model_a = models[0]
            model_b = models[1]

            # create dock UI
            if self.h2h_dock is None:
                dock = QDockWidget(tr("H2H Watch"), self)
                w = QWidget()
                v = QVBoxLayout(w)
                status = QLabel(tr("Connecting..."))
                te = QTextEdit()
                te.setReadOnly(True)
                # Allow text selection and copying from the H2H watch output
                try:
                    from PyQt6.QtCore import Qt
                    te.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
                except Exception:
                    # Fallback: if Qt flags not available, leave as read-only which still allows copy via context menu
                    pass
                v.addWidget(status)
                v.addWidget(te)
                w.setLayout(v)
                dock.setWidget(w)
                self.addDockWidget(Qt.RightDockWidgetArea, dock)
                self.h2h_dock = dock
                self.h2h_text = te
                self.h2h_status_label = status

            cmd = [sys.executable, h2h_script, model_a, model_b, '--games', '1', '--parallel', '1', '--pass-penalty', '0.2']
            self.h2h_watch_proc = QProcess(self)
            self.h2h_watch_proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            self.h2h_watch_proc.readyReadStandardOutput.connect(self._on_h2h_stdout)
            self.h2h_watch_proc.readyReadStandardError.connect(self._on_h2h_stderr)
            try:
                self.h2h_watch_proc.start(sys.executable, cmd[1:])
            except Exception as e:
                self.log_viewer.log_message(f"Failed to start head2head watch: {e}")
                QMessageBox.critical(self, tr('Error'), tr('Failed to start head2head process'))
                return
            self.log_viewer.log_message(tr('Started head2head watch:') + ' ' + ' '.join(cmd))
            if self.h2h_status_label:
                self.h2h_status_label.setText(tr('Running'))

        def _stop_h2h_watch(self) -> None:
            if not self.h2h_watch_proc:
                return
            try:
                self.h2h_watch_proc.terminate()
            except Exception:
                try:
                    self.h2h_watch_proc.kill()
                except Exception:
                    pass
            self.h2h_watch_proc = None
            if self.h2h_status_label:
                self.h2h_status_label.setText(tr('Stopped'))

        def _on_h2h_stderr(self) -> None:
            if not self.h2h_watch_proc:
                return
            data = bytes(self.h2h_watch_proc.readAllStandardError()).decode('utf-8', errors='replace')
            for line in data.splitlines():
                self.log_viewer.log_message(f"H2H_ERR: {line}")
                if self.h2h_text:
                    self.h2h_text.append(line)

        def _on_h2h_stdout(self) -> None:
            if not self.h2h_watch_proc:
                return
            data = bytes(self.h2h_watch_proc.readAllStandardOutput()).decode('utf-8', errors='replace')
            if not data:
                return
            self._h2h_stdout_buf += data
            lines = self._h2h_stdout_buf.splitlines(keepends=True)
            complete_lines = []
            for ln in lines:
                if ln.endswith('\n') or ln.endswith('\r'):
                    complete_lines.append(ln)
                else:
                    pass
            if complete_lines:
                processed = ''.join(complete_lines)
                self._h2h_stdout_buf = self._h2h_stdout_buf[len(processed):]
                for ln in complete_lines:
                    text = ln.strip()
                    # append raw
                    if self.h2h_text:
                        self.h2h_text.append(text)
                    # parse H2H_JSON messages
                    marker = 'H2H_JSON:'
                    if marker in text:
                        try:
                            idx = text.index(marker) + len(marker)
                            payload = text[idx:].strip()
                            obj = json.loads(payload)
                            ev = obj.get('event')
                            if ev == 'initial_state':
                                players = obj.get('players', [])
                                if self.h2h_status_label:
                                    self.h2h_status_label.setText(tr('Initial') + f" p0:{players[0].get('hand', '?')}h/{players[0].get('deck','?')}d p1:{players[1].get('hand','?')}h/{players[1].get('deck','?')}d")
                            elif ev == 'legal_map':
                                # show legal actions for current turn
                                if self.h2h_text:
                                    try:
                                        self.h2h_text.append('\n' + tr('Legal actions:') + '\n' + json.dumps(obj.get('legal_map', []), ensure_ascii=False))
                                    except Exception:
                                        pass
                            elif ev == 'progress' or ev == 'summary':
                                if self.h2h_status_label:
                                    try:
                                        wins = obj.get('wins', 0)
                                        losses = obj.get('losses', 0)
                                        draws = obj.get('draws', 0)
                                        self.h2h_status_label.setText(tr('Score') + f" {wins}-{losses}-{draws}")
                                    except Exception:
                                        pass
                        except Exception:
                            pass

    def _show_training_settings_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("トレーニング設定"))
        form = QFormLayout(dlg)

        # batch_size
        sb_batch = QSpinBox(dlg)
        sb_batch.setRange(1, 1024)
        sb_batch.setValue(int(self.trainer_config.get('batch_size', 8)))
        sb_batch.setToolTip(tr("ミニバッチサイズ。大きいほどGPU効率が上がりますがメモリを消費します。"))
        form.addRow(tr("バッチサイズ:"), sb_batch)

        # epochs
        sb_epochs = QSpinBox(dlg)
        sb_epochs.setRange(1, 100000)
        sb_epochs.setValue(int(self.trainer_config.get('epochs', 1)))
        sb_epochs.setToolTip(tr("学習エポック数。長時間学習で値を大きくします。"))
        form.addRow(tr("エポック数:"), sb_epochs)

        # lr
        ds_lr = QDoubleSpinBox(dlg)
        ds_lr.setDecimals(8)
        ds_lr.setRange(1e-8, 1.0)
        ds_lr.setSingleStep(1e-5)
        ds_lr.setValue(float(self.trainer_config.get('lr', 1e-4)))
        ds_lr.setToolTip(tr("学習率。小さくすると安定しますが学習が遅くなります。"))
        form.addRow(tr("学習率 (lr):"), ds_lr)

        # checkpoint_freq
        sb_ck = QSpinBox(dlg)
        sb_ck.setRange(1, 10000000)
        sb_ck.setValue(int(self.trainer_config.get('checkpoint_freq', 5000)))
        sb_ck.setToolTip(tr("チェックポイントを保存するステップ間隔。頻繁すぎるとI/O負荷が増えます。"))
        form.addRow(tr("チェックポイント頻度:"), sb_ck)

        # log_dir
        le_log = QLineEdit(dlg)
        le_log.setText(str(self.trainer_config.get('log_dir', 'logs/transformer')))
        le_log.setToolTip(tr("TensorBoard ログの保存先ディレクトリ。"))
        btn_log = QPushButton(tr("参照..."), dlg)
        def pick_log():
            d = QFileDialog.getExistingDirectory(self, tr("ログディレクトリを選択"), os.getcwd())
            if d:
                le_log.setText(d)
        btn_log.clicked.connect(pick_log)
        h_log = QHBoxLayout()
        h_log.addWidget(le_log)
        h_log.addWidget(btn_log)
        w_log = QWidget(dlg)
        w_log.setLayout(h_log)
        form.addRow(tr("ログディレクトリ:"), w_log)

        # checkpoint_dir
        le_ck = QLineEdit(dlg)
        le_ck.setText(str(self.trainer_config.get('checkpoint_dir', 'checkpoints/transformer')))
        le_ck.setToolTip(tr("チェックポイント保存先ディレクトリ。"))
        btn_ck = QPushButton(tr("参照..."), dlg)
        def pick_ck():
            d = QFileDialog.getExistingDirectory(self, tr("チェックポイントディレクトリを選択"), os.getcwd())
            if d:
                le_ck.setText(d)
        btn_ck.clicked.connect(pick_ck)
        h_ck = QHBoxLayout()
        h_ck.addWidget(le_ck)
        h_ck.addWidget(btn_ck)
        w_ck = QWidget(dlg)
        w_ck.setLayout(h_ck)
        form.addRow(tr("チェックポイントディレクトリ:"), w_ck)

        # --- head2head evaluation settings ---
        sb_eval_steps = QSpinBox(dlg)
        sb_eval_steps.setRange(0, 100000000)
        sb_eval_steps.setValue(int(self.trainer_config.get('eval_every_steps', 0)))
        sb_eval_steps.setToolTip(tr("0=無効。指定ステップ毎に軽量 head2head を自動実行します。"))
        form.addRow(tr("評価頻度（ステップ毎）:"), sb_eval_steps)

        sb_eval_games = QSpinBox(dlg)
        sb_eval_games.setRange(1, 10000)
        sb_eval_games.setValue(int(self.trainer_config.get('eval_games', 50)))
        sb_eval_games.setToolTip(tr("各評価で実行するゲーム数（軽量評価は 20-100 推奨）。"))
        form.addRow(tr("評価ゲーム数:"), sb_eval_games)

        sb_eval_parallel = QSpinBox(dlg)
        sb_eval_parallel.setRange(1, 256)
        sb_eval_parallel.setValue(int(self.trainer_config.get('eval_parallel', 8)))
        sb_eval_parallel.setToolTip(tr("head2head の parallel 値（バッチ評価サイズ）。"))
        form.addRow(tr("評価 parallel:"), sb_eval_parallel)

        cb_eval_pytorch = QCheckBox(dlg)
        cb_eval_pytorch.setChecked(bool(self.trainer_config.get('eval_use_pytorch', False)))
        cb_eval_pytorch.setToolTip(tr("PyTorch チェックポイント（.pth）を直接読み込んで評価する場合はチェック。"))
        form.addRow(tr("PyTorch 評価を使用:"), cb_eval_pytorch)

        le_baseline = QLineEdit(dlg)
        le_baseline.setText(str(self.trainer_config.get('eval_baseline', '')))
        le_baseline.setToolTip(tr("対戦相手（ベースライン）モデルのパス。ONNX または .pth を指定。空ならランダム/デフォルト。"))
        btn_base = QPushButton(tr("参照..."), dlg)
        def pick_base():
            f, _ = QFileDialog.getOpenFileName(self, tr("ベースラインモデルを選択"), os.getcwd(), "Model Files (*.onnx *.pth);;All Files (*)")
            if f:
                le_baseline.setText(f)
        btn_base.clicked.connect(pick_base)
        h_base = QHBoxLayout()
        h_base.addWidget(le_baseline)
        h_base.addWidget(btn_base)
        w_base = QWidget(dlg)
        w_base.setLayout(h_base)
        form.addRow(tr("ベースラインモデル:"), w_base)

        # Buttons
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        def on_accept():
            self.trainer_config['batch_size'] = int(sb_batch.value())
            self.trainer_config['epochs'] = int(sb_epochs.value())
            self.trainer_config['lr'] = float(ds_lr.value())
            self.trainer_config['checkpoint_freq'] = int(sb_ck.value())
            self.trainer_config['log_dir'] = str(le_log.text())
            self.trainer_config['checkpoint_dir'] = str(le_ck.text())
            # head2head settings
            self.trainer_config['eval_every_steps'] = int(sb_eval_steps.value())
            self.trainer_config['eval_games'] = int(sb_eval_games.value())
            self.trainer_config['eval_parallel'] = int(sb_eval_parallel.value())
            self.trainer_config['eval_use_pytorch'] = bool(cb_eval_pytorch.isChecked())
            self.trainer_config['eval_baseline'] = str(le_baseline.text())
            dlg.accept()
        def on_reject():
            dlg.reject()
        bb.accepted.connect(on_accept)
        bb.rejected.connect(on_reject)
        form.addRow(bb)

        dlg.setLayout(form)
        dlg.exec()

def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Creating QApplication...")
    app = QApplication(sys.argv)

    if dm_ai_module is None:
        QMessageBox.critical(None, tr("Error"),
            tr("dm_ai_module not found. Please build the C++ extension."))
        sys.exit(1)

    print("Creating GameWindow...")
    try:
        window = GameWindow()
        print("GameWindow created successfully!")
        window.show()
        print("Window shown, entering event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error creating or showing window: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
