# -*- coding: utf-8 -*-
import sys
import os
import random
import json
import csv
from typing import Any, List, Optional, Dict, cast
from types import ModuleType

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QDockWidget, QTabWidget,
    QInputDialog, QToolBar
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer

dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.types import GameState, CardDB, Action, PlayerID, CardID
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.localization import tr, describe_command
from dm_toolkit.gui.deck_builder import DeckBuilder
from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.widgets.scenario_tools import ScenarioToolsDock
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.stack_view import StackViewWidget
from dm_toolkit.gui.widgets.loop_recorder import LoopRecorderWidget
from dm_toolkit.gui.widgets.card_effect_debugger import CardEffectDebugger
from dm_toolkit.gui.dialogs.selection_dialog import CardSelectionDialog

# New Components
from dm_toolkit.gui.widgets.log_viewer import LogViewer
from dm_toolkit.gui.widgets.control_panel import ControlPanel
from dm_toolkit.gui.widgets.game_board import GameBoard
from dm_toolkit.gui.game_session import GameSession
from dm_toolkit.gui.input_handler import GameInputHandler

class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("DM AI Simulator"))
        self.resize(1600, 900)

        # Initialize GameSession (Logic)
        self.session = GameSession(
            callback_update_ui=self.update_ui,
            callback_log=self.log_message,
            callback_input_request=self.handle_user_input_request,
            callback_action_executed=self.on_action_executed
        )

        # Initialize Input Handler
        self.input_handler = GameInputHandler(self, self.session)

        # Ensure log list exists before any callbacks
        self.log_viewer = LogViewer()

        # Load card database
        self.card_db: CardDB = {}
        loaded_db = EngineCompat.JsonLoader_load_cards("data/cards.json")
        if loaded_db:
             self.card_db = loaded_db
        else:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                json_path = os.path.join(base_dir, 'data', 'cards.json')
                with open(json_path, 'r', encoding='utf-8') as _f:
                    self.card_db = json.load(_f)
            except Exception:
                self.card_db = {}
        
        # Initialize Game
        self.session.initialize_game(self.card_db)

        self.p0_deck_ids: Optional[List[int]] = None
        self.p1_deck_ids: Optional[List[int]] = None
        self.last_command_index: int = 0

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.session.step_phase)
        # Note: Session manages 'is_processing', App manages 'is_running' for timer
        self.is_running: bool = False

        # Toolbar
        self.init_toolbar()

        # UI Setup
        self.init_ui()

        self.update_ui()
        self.showMaximized()

    @property
    def gs(self):
        return self.session.gs

    def init_toolbar(self):
        self.toolbar = QToolBar(tr("Main Toolbar"), self)
        self.toolbar.setObjectName("MainToolbar")
        self.addToolBar(self.toolbar)

        deck_act = QAction(tr("Deck Builder"), self)
        deck_act.triggered.connect(self.open_deck_builder)
        self.toolbar.addAction(deck_act)

        card_act = QAction(tr("Card Editor"), self)
        card_act.triggered.connect(self.open_card_editor)
        self.toolbar.addAction(card_act)

        self.scen_act = QAction(tr("Scenario Mode"), self)
        self.scen_act.setCheckable(True)
        self.scen_act.triggered.connect(self.toggle_scenario_mode)
        self.toolbar.addAction(self.scen_act)

        sim_act = QAction(tr("Batch Simulation"), self)
        sim_act.triggered.connect(self.open_simulation_dialog)
        self.toolbar.addAction(sim_act)

        ai_act = QAction(tr("AI Analysis"), self)
        ai_act.triggered.connect(lambda: self.mcts_dock.setVisible(not self.mcts_dock.isVisible()))
        self.toolbar.addAction(ai_act)

        loop_act = QAction(tr("Loop Recorder"), self)
        loop_act.triggered.connect(lambda: self.loop_dock.setVisible(not self.loop_dock.isVisible()))
        self.toolbar.addAction(loop_act)

        debug_act = QAction(tr("Effect Debugger"), self)
        debug_act.triggered.connect(lambda: self.debugger_dock.setVisible(not self.debugger_dock.isVisible()))
        self.toolbar.addAction(debug_act)

        log_act = QAction(tr("Logs"), self)
        log_act.triggered.connect(lambda: self.log_dock.setVisible(not self.log_dock.isVisible()))
        self.toolbar.addAction(log_act)

    def init_ui(self):
        # AI Tools Dock (Left)
        self.ai_tools_dock = QDockWidget(tr("AI & Tools"), self)
        self.ai_tools_dock.setObjectName("AIToolsDock")
        self.ai_tools_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.ai_tools_panel = QWidget()
        self.ai_tools_panel.setMinimumWidth(300)
        self.ai_tools_layout = QVBoxLayout(self.ai_tools_panel)
        self.ai_tools_dock.setWidget(self.ai_tools_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ai_tools_dock)

        # Game Status Dock (Right)
        self.status_dock = QDockWidget(tr("Game Status & Operations"), self)
        self.status_dock.setObjectName("StatusDock")
        self.status_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        self.status_panel = QWidget()
        self.status_panel.setMinimumWidth(300)
        self.status_layout_main = QVBoxLayout(self.status_panel)
        self.status_dock.setWidget(self.status_panel)
        
        self.top_section_group = QGroupBox(tr("Game Status"))
        top_layout = QVBoxLayout()
        
        status_layout = QHBoxLayout()
        self.turn_label = QLabel(tr("Turn: {turn}").format(turn=1))
        self.turn_label.setStyleSheet("font-weight: bold;")
        self.phase_label = QLabel(tr("Phase: {phase}").format(phase=tr("Start Phase")))
        self.active_label = QLabel(tr("Active: P{player_id}").format(player_id=0))
        status_layout.addWidget(self.turn_label)
        status_layout.addWidget(self.phase_label)
        status_layout.addWidget(self.active_label)
        top_layout.addLayout(status_layout)

        self.card_detail_panel = CardDetailPanel()
        top_layout.addWidget(self.card_detail_panel)
        
        self.top_section_group.setLayout(top_layout)
        self.status_layout_main.addWidget(self.top_section_group)
        
        # Control Panel
        self.control_panel = ControlPanel()
        self.control_panel.start_simulation_clicked.connect(self.toggle_simulation)
        self.control_panel.step_clicked.connect(self.session.step_phase)
        self.control_panel.pass_clicked.connect(self.pass_turn)
        self.control_panel.confirm_clicked.connect(self.confirm_selection)
        self.control_panel.reset_clicked.connect(self.reset_game)
        
        # Tool connections
        self.control_panel.deck_builder_clicked.connect(self.open_deck_builder)
        self.control_panel.card_editor_clicked.connect(self.open_card_editor)
        self.control_panel.batch_sim_clicked.connect(self.open_simulation_dialog)
        self.control_panel.load_deck_p0_clicked.connect(self.load_deck_p0)
        self.control_panel.load_deck_p1_clicked.connect(self.load_deck_p1)
        self.control_panel.god_view_toggled.connect(self.update_ui)
        self.control_panel.help_clicked.connect(self.show_help)
        
        # Mode update connections
        self.control_panel.p0_human_radio.toggled.connect(lambda c: self.session.set_player_mode(0, 'Human' if c else 'AI'))
        self.control_panel.p1_human_radio.toggled.connect(lambda c: self.session.set_player_mode(1, 'Human' if c else 'AI'))

        self.ai_tools_layout.addWidget(self.control_panel)
        self.ai_tools_layout.addStretch()
        
        self.status_layout_main.addStretch()

        # Board Panel
        self.game_board = GameBoard()
        self.game_board.action_triggered.connect(self.session.execute_action)
        self.game_board.card_clicked.connect(self.on_card_clicked)
        self.game_board.card_double_clicked.connect(self.on_card_double_clicked)
        self.game_board.card_hovered.connect(self.on_card_hovered)
        
        self.setCentralWidget(self.game_board)
        
        # Docks
        self.stack_dock = QDockWidget(tr("Pending Effects"), self)
        self.stack_dock.setObjectName("StackDock")
        self.stack_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.stack_view = StackViewWidget()
        self.stack_view.effect_resolved.connect(self.on_resolve_effect_from_stack)
        self.stack_dock.setWidget(self.stack_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stack_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.status_dock)
        self.splitDockWidget(self.stack_dock, self.status_dock, Qt.Orientation.Vertical)
        
        self.log_dock = QDockWidget(tr("Logs"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_dock.setWidget(self.log_viewer)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.log_dock.hide()

        self.mcts_dock: QDockWidget = QDockWidget(tr("MCTS Analysis"), self)
        self.mcts_dock.setObjectName("MCTSDock")
        self.mcts_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.mcts_view = MCTSView()
        self.mcts_dock.setWidget(self.mcts_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.mcts_dock)
        self.mcts_dock.hide()

        self.loop_dock: QDockWidget = QDockWidget(tr("Loop Recorder"), self)
        self.loop_dock.setObjectName("LoopDock")
        self.loop_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.loop_recorder = LoopRecorderWidget(lambda: self.gs)
        self.loop_dock.setWidget(self.loop_recorder)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.loop_dock)
        self.loop_dock.hide()

        self.debugger_dock = QDockWidget(tr("Card Effect Debugger"), self)
        self.debugger_dock.setObjectName("DebuggerDock")
        self.debugger_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.effect_debugger = CardEffectDebugger()
        self.debugger_dock.setWidget(self.effect_debugger)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.debugger_dock)
        self.debugger_dock.hide()

        self.scenario_tools = ScenarioToolsDock(self, self.gs, self.card_db)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.scenario_tools)
        self.scenario_tools.hide()
        
    def log_message(self, msg: str) -> None:
        self.log_viewer.log_message(msg)

    def on_action_executed(self, action_dict: Dict[str, Any]) -> None:
        """Callback from GameSession when an action is executed."""
        self.loop_recorder.record_action(str(action_dict))
        self.scenario_tools.record_action(str(action_dict))

    def open_deck_builder(self) -> None:
        self.deck_builder = DeckBuilder(self.card_db)
        self.deck_builder.show()

    def open_card_editor(self) -> None:
        self.card_editor = CardEditor("data/cards.json")
        self.card_editor.data_saved.connect(self.reload_card_data)
        self.card_editor.show()

    def reload_card_data(self) -> None:
        try:
            loaded = EngineCompat.JsonLoader_load_cards("data/cards.json")
            if loaded is not None:
                self.card_db = loaded
            if hasattr(self, 'deck_builder') and self.deck_builder.isVisible():
                self.deck_builder.reload_database()
            self.scenario_tools.set_game_state(self.gs, self.card_db)
            self.log_viewer.log_message(tr("Card Data Reloaded from Editor Save"))
        except Exception as e:
            self.log_viewer.log_message(f"{tr('Error reloading cards')}: {e}")

    def toggle_scenario_mode(self, checked: bool) -> None:
        if checked:
            self.scenario_tools.show()
            self.log_viewer.log_message(tr("Scenario Mode Enabled"))
            if self.is_running:
                self.toggle_simulation()
        else:
            self.scenario_tools.hide()
            self.log_viewer.log_message(tr("Scenario Mode Disabled"))

    def open_simulation_dialog(self) -> None:
        self.sim_dialog = SimulationDialog(self.card_db, self)
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
                self.p0_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P0')}: {os.path.basename(fname)}")
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
                self.p1_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P1')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def show_help(self) -> None:
        QMessageBox.information(self, tr("Help / Manual"), tr("Help text..."))

    def toggle_simulation(self) -> None:
        if self.is_running:
            self.timer.stop()
            self.control_panel.set_start_button_text(tr("Start Sim"))
            self.is_running = False
        else:
            self.timer.start(500)
            self.control_panel.set_start_button_text(tr("Stop Sim"))
            self.is_running = True

    def reset_game(self) -> None:
        self.timer.stop()
        self.is_running = False
        self.input_handler.selected_targets = []
        self.control_panel.set_confirm_button_visible(False)
        self.control_panel.set_pass_button_visible(False)
        self.control_panel.set_start_button_text(tr("Start Sim"))
        self.log_viewer.clear_logs()

        self.session.reset_game(self.p0_deck_ids, self.p1_deck_ids)

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
            if card_data:
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
        if not hasattr(self, 'turn_label'):
            return

        turn_number = EngineCompat.get_turn_number(self.gs)
        current_phase = EngineCompat.get_current_phase(self.gs)

        phase_map = {
            "START": "Start Phase",
            "DRAW": "Draw Phase",
            "MANA": "Mana Phase",
            "MAIN": "Main Phase",
            "ATTACK": "Attack Phase",
            "BLOCK": "Block Phase",
            "END": "End Phase"
        }
        phase_key = phase_map.get(str(current_phase), str(current_phase))

        active_pid = EngineCompat.get_active_player_id(self.gs)
        self.turn_label.setText(tr("Turn: {turn}").format(turn=turn_number))
        self.phase_label.setText(tr("Phase: {phase}").format(phase=tr(phase_key)))
        self.active_label.setText(tr("Active: P{player_id}").format(player_id=active_pid))
        
        self.stack_view.update_state(self.gs, self.card_db)
        self.effect_debugger.update_state(self.gs, self.card_db)

        history = EngineCompat.get_command_history(self.gs)
        try: current_len = len(history)
        except: current_len = 0
        if current_len > self.last_command_index:
            for i in range(self.last_command_index, current_len):
                cmd = history[i]
                try: desc = describe_command(cmd, self.gs, self.card_db)
                except: desc = str(cmd)
                self.log_viewer.log_message(desc)
            self.last_command_index = current_len

        class _P:
             hand: List[Any] = []
             mana_zone: List[Any] = []
             battle_zone: List[Any] = []
             shield_zone: List[Any] = []
             graveyard: List[Any] = []
             deck: List[Any] = []

        p0 = EngineCompat.get_player(self.gs, 0) or _P()
        p1 = EngineCompat.get_player(self.gs, 1) or _P()
        
        legal_actions = []
        if active_pid == 0 and self.control_panel.is_p0_human() and not self.gs.game_over:
             from dm_toolkit.commands import generate_legal_commands
             legal_actions = generate_legal_commands(self.gs, self.card_db)

        self.current_pass_action = None
        for cmd in legal_actions:
            try: d = cmd.to_dict()
            except: d = {}
            if d.get('type') == 'PASS' or d.get('legacy_original_type') == 'PASS':
                self.current_pass_action = cmd
                break

        self.control_panel.set_pass_button_visible(bool(self.current_pass_action))

        god_view = self.control_panel.is_god_view()
        self.game_board.update_state(p0, p1, self.card_db, legal_actions, god_view)

        pending = EngineCompat.get_pending_query(self.gs)
        if EngineCompat.is_waiting_for_user_input(self.gs) and pending is not None and getattr(pending, 'query_type', '') == "SELECT_TARGET":
            params = getattr(pending, 'params', {})
            min_targets = params.get('min', 1) if hasattr(params, 'get') else 1
            max_targets = params.get('max', 99) if hasattr(params, 'get') else 99

            current = len(self.input_handler.selected_targets)
            self.control_panel.set_confirm_button_text(f"{tr('Confirm')} ({current}/{min_targets}-{max_targets})")
            self.control_panel.set_confirm_button_visible(True)
            self.control_panel.set_confirm_button_enabled(current >= min_targets)

            self.game_board.set_selection_mode(self.input_handler.selected_targets)
        else:
            self.control_panel.set_confirm_button_visible(False)

def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    if dm_ai_module is None:
        QMessageBox.critical(None, tr("Error"),
            tr("dm_ai_module not found. Please build the C++ extension."))
        sys.exit(1)

    window = GameWindow()
    window.show()
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
