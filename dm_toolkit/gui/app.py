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
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.stack_view import StackViewWidget
from dm_toolkit.gui.widgets.loop_recorder import LoopRecorderWidget
from dm_toolkit.gui.widgets.card_effect_debugger import CardEffectDebugger
from dm_toolkit.gui.dialogs.selection_dialog import CardSelectionDialog

# Import Phase 7 requirements: Action Wrapper and Mapper
from dm_toolkit.commands import wrap_action
# USE UNIFIED EXECUTION PIPELINE (also exposes deprecated action_mapper for backward compatibility if needed)
from dm_toolkit.unified_execution import ensure_executable_command
# Restore legacy mapper import if needed or use unified
# from dm_toolkit.action_to_command import map_action  <-- REPLACED BY unified pipeline

# Import GameController
from dm_toolkit.gui.game_controller import GameController

class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("DM AI Simulator"))
        self.resize(1600, 900)

        # Initialize GameController
        self.controller = GameController(self.update_ui, self.log_message)

        # Ensure log list exists before any callbacks (e.g., controller init) attempt to log.
        self.log_list = QListWidget()

        # Ensure GameState attribute exists before controller init triggers update_ui callback.
        self.gs: Optional[GameState] = None

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
        
        # Initialize Game via Controller
        self.controller.initialize_game(self.card_db)
        self.gs = self.controller.gs  # Link for UI binding

        self.p0_deck_ids: Optional[List[int]] = None
        self.p1_deck_ids: Optional[List[int]] = None
        self.last_action: Optional[Action] = None
        self.selected_targets: List[int] = []
        self.last_command_index: int = 0

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_phase)
        self.is_running: bool = False
        self.is_processing: bool = False

        # Toolbar
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

        # UI Setup
        # AI Tools Dock (Left)
        self.ai_tools_dock = QDockWidget(tr("AI & Tools"), self)
        self.ai_tools_dock.setObjectName("AIToolsDock")
        self.ai_tools_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.ai_tools_panel = QWidget()
        self.ai_tools_panel.setMinimumWidth(300)
        self.ai_tools_layout = QVBoxLayout(self.ai_tools_panel)
        self.ai_tools_dock.setWidget(self.ai_tools_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ai_tools_dock)

        # Game Status Dock (Right) - Separate from AI Tools
        self.status_dock = QDockWidget(tr("Game Status & Operations"), self)
        self.status_dock.setObjectName("StatusDock")
        self.status_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        self.status_panel = QWidget()
        self.status_panel.setMinimumWidth(300)
        self.status_layout_main = QVBoxLayout(self.status_panel)
        self.status_dock.setWidget(self.status_panel)
        
        self.top_section_group = QGroupBox(tr("Game Status & Operations"))
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
        
        game_ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start Sim"))
        self.start_btn.setShortcut("F5")
        self.start_btn.clicked.connect(self.toggle_simulation)
        game_ctrl_layout.addWidget(self.start_btn)

        self.step_button = QPushButton(tr("Step"))
        self.step_button.setShortcut("Space")
        self.step_button.clicked.connect(self.step_phase)
        game_ctrl_layout.addWidget(self.step_button)

        self.pass_btn = QPushButton(tr("Pass / End Turn"))
        self.pass_btn.setShortcut("Ctrl+E")
        self.pass_btn.clicked.connect(self.pass_turn)
        self.pass_btn.setVisible(False)
        self.pass_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        game_ctrl_layout.addWidget(self.pass_btn)

        self.confirm_btn = QPushButton(tr("Confirm Selection"))
        self.confirm_btn.setShortcut("Return")
        self.confirm_btn.clicked.connect(self.confirm_selection)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        game_ctrl_layout.addWidget(self.confirm_btn)

        self.reset_btn = QPushButton(tr("Reset"))
        self.reset_btn.setShortcut("Ctrl+R")
        self.reset_btn.clicked.connect(self.reset_game)
        game_ctrl_layout.addWidget(self.reset_btn)
        top_layout.addLayout(game_ctrl_layout)

        self.top_section_group.setLayout(top_layout)
        self.status_layout_main.addWidget(self.top_section_group)
        self.status_layout_main.addStretch()

        # Bottom Section (AI & Tools) - now in ai_tools_dock
        self.bottom_section_group = QGroupBox(tr("AI & Tools"))
        bottom_layout = QVBoxLayout()

        mode_group = QGroupBox(tr("Player Mode"))
        mode_layout = QVBoxLayout()

        self.p0_human_radio = QRadioButton(tr("P0 (Self): Human"))
        self.p0_ai_radio = QRadioButton(tr("P0 (Self): AI"))
        self.p0_ai_radio.setChecked(True)
        self.p0_group = QButtonGroup()
        self.p0_group.addButton(self.p0_human_radio)
        self.p0_group.addButton(self.p0_ai_radio)
        self.p0_human_radio.toggled.connect(lambda _: self._update_p0_controls_visibility())
        self.p0_ai_radio.toggled.connect(lambda _: self._update_p0_controls_visibility())
        
        mode_layout.addWidget(self.p0_human_radio)
        mode_layout.addWidget(self.p0_ai_radio)
        
        self.p1_human_radio = QRadioButton(tr("P1 (Opp): Human"))
        self.p1_ai_radio = QRadioButton(tr("P1 (Opp): AI"))
        self.p1_ai_radio.setChecked(True)
        self.p1_group = QButtonGroup()
        self.p1_group.addButton(self.p1_human_radio)
        self.p1_group.addButton(self.p1_ai_radio)
        
        mode_layout.addWidget(self.p1_human_radio)
        mode_layout.addWidget(self.p1_ai_radio)
        mode_group.setLayout(mode_layout)
        bottom_layout.addWidget(mode_group)
        
        tools_layout = QVBoxLayout()
        self.deck_builder_button = QPushButton(tr("Deck Builder"))
        self.deck_builder_button.clicked.connect(self.open_deck_builder)
        tools_layout.addWidget(self.deck_builder_button)

        self.card_editor_button = QPushButton(tr("Card Editor"))
        self.card_editor_button.clicked.connect(self.open_card_editor)
        tools_layout.addWidget(self.card_editor_button)

        self.sim_dialog_button = QPushButton(tr("Batch Simulation"))
        self.sim_dialog_button.clicked.connect(self.open_simulation_dialog)
        tools_layout.addWidget(self.sim_dialog_button)
        bottom_layout.addLayout(tools_layout)

        deck_group = QGroupBox(tr("Deck Management"))
        deck_layout = QVBoxLayout()
        self.load_deck_btn = QPushButton(tr("Load Deck P0"))
        self.load_deck_btn.clicked.connect(self.load_deck_p0)
        deck_layout.addWidget(self.load_deck_btn)

        self.load_deck_p1_btn = QPushButton(tr("Load Deck P1"))
        self.load_deck_p1_btn.clicked.connect(self.load_deck_p1)
        deck_layout.addWidget(self.load_deck_p1_btn)
        deck_group.setLayout(deck_layout)
        bottom_layout.addWidget(deck_group)
        
        self.god_view_check = QCheckBox(tr("God View"))
        self.god_view_check.setChecked(False)
        self.god_view_check.stateChanged.connect(self.update_ui)
        bottom_layout.addWidget(self.god_view_check)

        self.help_btn = QPushButton(tr("Help / Manual"))
        self.help_btn.clicked.connect(self.show_help)
        bottom_layout.addWidget(self.help_btn)

        self.bottom_section_group.setLayout(bottom_layout)
        self.ai_tools_layout.addWidget(self.bottom_section_group)

        # P0 human control panel (hidden unless P0 is human)
        self.p0_control_group = QGroupBox(tr("P0 Controls"))
        p0_ctrl_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.p0_ctrl_start = QPushButton(tr("Start"))
        self.p0_ctrl_start.clicked.connect(self.toggle_simulation)
        row1.addWidget(self.p0_ctrl_start)

        self.p0_ctrl_step = QPushButton(tr("Step"))
        self.p0_ctrl_step.clicked.connect(self.step_phase)
        row1.addWidget(self.p0_ctrl_step)
        p0_ctrl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.p0_ctrl_pass = QPushButton(tr("Pass / End"))
        self.p0_ctrl_pass.clicked.connect(self.pass_turn)
        row2.addWidget(self.p0_ctrl_pass)

        self.p0_ctrl_confirm = QPushButton(tr("Confirm"))
        self.p0_ctrl_confirm.clicked.connect(self.confirm_selection)
        row2.addWidget(self.p0_ctrl_confirm)
        p0_ctrl_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.p0_ctrl_reset = QPushButton(tr("Reset"))
        self.p0_ctrl_reset.clicked.connect(self.reset_game)
        row3.addWidget(self.p0_ctrl_reset)
        p0_ctrl_layout.addLayout(row3)

        self.p0_control_group.setLayout(p0_ctrl_layout)
        self.ai_tools_layout.addWidget(self.p0_control_group)

        self.ai_tools_layout.addStretch()
        
        # Board Panel
        self.board_panel = QWidget()
        self.board_layout = QVBoxLayout(self.board_panel)
        self.board_layout.setContentsMargins(0, 0, 0, 0)
        
        self.p1_zones = QWidget()
        self.p1_layout = QVBoxLayout(self.p1_zones)
        self.p1_hand = ZoneWidget(tr("P1 Hand"))
        self.p1_mana = ZoneWidget(tr("P1 Mana"))
        self.p1_graveyard = ZoneWidget(tr("P1 Graveyard"))
        self.p1_battle = ZoneWidget(tr("P1 Battle Zone"))
        self.p1_shield = ZoneWidget(tr("P1 Shield Zone"))
        self.p1_deck_zone = ZoneWidget(tr("P1 Deck"))
        
        self.p1_layout.addWidget(self.p1_hand)
        p1_row2 = QHBoxLayout()
        p1_row2.addWidget(self.p1_mana, stretch=3)
        p1_row2.addWidget(self.p1_shield, stretch=2)
        p1_row2.addWidget(self.p1_graveyard, stretch=1)
        self.p1_layout.addLayout(p1_row2)

        p1_battle_row = QHBoxLayout()
        p1_battle_row.addWidget(self.p1_battle, stretch=5)
        p1_battle_row.addWidget(self.p1_deck_zone, stretch=1)
        self.p1_layout.addLayout(p1_battle_row)
        
        self.p0_zones = QWidget()
        self.p0_layout = QVBoxLayout(self.p0_zones)
        self.p0_battle = ZoneWidget(tr("P0 Battle Zone"))
        self.p0_deck_zone = ZoneWidget(tr("P0 Deck"))
        self.p0_shield = ZoneWidget(tr("P0 Shield Zone"))
        self.p0_mana = ZoneWidget(tr("P0 Mana"))
        self.p0_graveyard = ZoneWidget(tr("P0 Graveyard"))
        self.p0_hand = ZoneWidget(tr("P0 Hand"))
        
        # Connect
        self.p0_hand.action_triggered.connect(self.execute_action)
        self.p0_mana.action_triggered.connect(self.execute_action)
        self.p0_battle.action_triggered.connect(self.execute_action)
        self.p0_graveyard.action_triggered.connect(self.execute_action)

        self.p0_hand.card_clicked.connect(self.on_card_clicked)
        self.p0_mana.card_clicked.connect(self.on_card_clicked)
        self.p0_battle.card_clicked.connect(self.on_card_clicked)
        self.p0_graveyard.card_clicked.connect(self.on_card_clicked)
        
        # Connect double-click handlers for quick play
        self.p0_hand.card_double_clicked.connect(self.on_card_double_clicked)
        self.p0_mana.card_double_clicked.connect(self.on_card_double_clicked)
        self.p0_battle.card_double_clicked.connect(self.on_card_double_clicked)
        self.p0_graveyard.card_double_clicked.connect(self.on_card_double_clicked)
        
        self.p0_hand.card_hovered.connect(self.on_card_hovered)
        self.p0_mana.card_hovered.connect(self.on_card_hovered)
        self.p0_battle.card_hovered.connect(self.on_card_hovered)
        self.p0_shield.card_hovered.connect(self.on_card_hovered)
        self.p0_graveyard.card_hovered.connect(self.on_card_hovered)
        
        self.p1_hand.card_hovered.connect(self.on_card_hovered)
        self.p1_mana.card_hovered.connect(self.on_card_hovered)
        self.p1_battle.card_hovered.connect(self.on_card_hovered)
        self.p1_shield.card_hovered.connect(self.on_card_hovered)
        self.p1_graveyard.card_hovered.connect(self.on_card_hovered)

        # Ensure initial visibility
        self._update_p0_controls_visibility()
        
        p0_battle_row = QHBoxLayout()
        p0_battle_row.addWidget(self.p0_battle, stretch=5)
        p0_battle_row.addWidget(self.p0_deck_zone, stretch=1)
        self.p0_layout.addLayout(p0_battle_row)

        p0_row2 = QHBoxLayout()
        p0_row2.addWidget(self.p0_mana, stretch=3)
        p0_row2.addWidget(self.p0_shield, stretch=2)
        p0_row2.addWidget(self.p0_graveyard, stretch=1)
        self.p0_layout.addLayout(p0_row2)
        self.p0_layout.addWidget(self.p0_hand)
        
        self.board_splitter = QSplitter(Qt.Orientation.Vertical)
        self.board_splitter.addWidget(self.p1_zones)
        self.board_splitter.addWidget(self.p0_zones)
        self.board_layout.addWidget(self.board_splitter)
        
        self.setCentralWidget(self.board_panel)
        
        # Docks
        self.stack_dock = QDockWidget(tr("Pending Effects"), self)
        self.stack_dock.setObjectName("StackDock")
        self.stack_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.stack_view = StackViewWidget()
        self.stack_view.effect_resolved.connect(self.on_resolve_effect_from_stack)
        self.stack_dock.setWidget(self.stack_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stack_dock)

        # Add Status Dock to right side (where log was)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.status_dock)
        self.splitDockWidget(self.stack_dock, self.status_dock, Qt.Orientation.Vertical)
        
        self.log_dock = QDockWidget(tr("Logs"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_dock.setWidget(self.log_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.log_dock.hide()  # Hide by default

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

        self.update_ui()
        self.showMaximized()
        
    def log_message(self, msg: str) -> None:
        self.log_list.addItem(msg)
        self.log_list.scrollToBottom()

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
            self.civ_map = self.build_civ_map()
            if hasattr(self, 'deck_builder') and self.deck_builder.isVisible():
                self.deck_builder.reload_database()
            self.scenario_tools.set_game_state(self.gs, self.card_db)
            self.log_list.addItem(tr("Card Data Reloaded from Editor Save"))
        except Exception as e:
            self.log_list.addItem(f"{tr('Error reloading cards')}: {e}")

    def build_civ_map(self) -> Dict[str, Any]:
        return {}

    def toggle_scenario_mode(self, checked: bool) -> None:
        if checked:
            self.scenario_tools.show()
            self.log_list.addItem(tr("Scenario Mode Enabled"))
            if self.is_running:
                self.toggle_simulation()
        else:
            self.scenario_tools.hide()
            self.log_list.addItem(tr("Scenario Mode Disabled"))

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
                self.log_list.addItem(f"{tr('Loaded Deck for P0')}: {os.path.basename(fname)}")
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
                self.log_list.addItem(f"{tr('Loaded Deck for P1')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def show_help(self) -> None:
        QMessageBox.information(self, tr("Help / Manual"), tr("Help text..."))

    def toggle_simulation(self) -> None:
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText(tr("Start Sim"))
            self.is_running = False
        else:
            self.timer.start(500)
            self.start_btn.setText(tr("Stop Sim"))
            self.is_running = True

    def reset_game(self) -> None:
        self.timer.stop()
        self.is_running = False
        self.selected_targets = []
        self.confirm_btn.setVisible(False)
        self.pass_btn.setVisible(False)
        self.start_btn.setText(tr("Start Sim"))
        self.log_list.clear()

        # Delegate logic to controller
        self.controller.reset_game(self.p0_deck_ids, self.p1_deck_ids)
        self.gs = self.controller.gs # Sync back reference

        self.scenario_tools.set_game_state(self.gs, self.card_db)
        self.last_command_index = 0
        self.update_ui()

    def pass_turn(self) -> None:
        if hasattr(self, 'current_pass_action') and self.current_pass_action:
            self.execute_action(self.current_pass_action)

    def confirm_selection(self) -> None:
        if not EngineCompat.is_waiting_for_user_input(self.gs): return
        query = EngineCompat.get_pending_query(self.gs)
        min_targets = query.params.get('min', 1)
        if len(self.selected_targets) < min_targets:
            # Converted to localized message box:
            msg = tr("Please select at least {min_targets} target(s).").format(min_targets=min_targets)
            QMessageBox.warning(self, tr("Invalid Selection"), msg)
            return
        targets = list(self.selected_targets)
        self.selected_targets = []
        self.confirm_btn.setVisible(False)
        EngineCompat.EffectResolver_resume(self.gs, self.card_db, targets)
        self.step_phase()

    def on_card_clicked(self, card_id: int, instance_id: int) -> None:
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.p0_human_radio.isChecked(): return

        if EngineCompat.is_waiting_for_user_input(self.gs):
             pending = EngineCompat.get_pending_query(self.gs)
             if pending.query_type == "SELECT_TARGET":
                 valid_targets = pending.valid_targets
                 if instance_id in valid_targets:
                     if instance_id in self.selected_targets:
                         self.selected_targets.remove(instance_id)
                     else:
                         query_max = pending.params.get('max', 99)
                         if len(self.selected_targets) < query_max:
                             self.selected_targets.append(instance_id)
                         else:
                             return
                     self.update_ui()
             return

        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
            if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
                relevant_cmds.append(c)

        if not relevant_cmds: return

        # If multiple actions are available for the same card, let the user choose
        # or rely on the context menu (which CardWidget handles).
        # If we just execute the first one, we might pick the wrong one (e.g. Play vs Mana Charge).
        if len(relevant_cmds) > 1:
            # We could show a dialog, but CardWidget already has a context menu on right-click.
            # To avoid confusion on left-click, we can trigger the context menu logic programmatically
            # or show a simple selection dialog here.

            # Construct items for selection
            items = []
            for cmd in relevant_cmds:
                d = cmd.to_dict()
                desc = describe_command(d, self.gs, self.card_db)
                items.append({'description': desc, 'command': cmd})

            # Use CardSelectionDialog-like logic but for actions
            options = [item['description'] for item in items]
            item, ok = QInputDialog.getItem(self, tr("Select Action"), tr("Choose action to perform:"), options, 0, False)
            if ok and item:
                idx = options.index(item)
                self.execute_action(items[idx]['command'])
            return

        self.execute_action(relevant_cmds[0])

    def on_card_double_clicked(self, card_id: int, instance_id: int) -> None:
        """Handle double-click to quickly play the most common action (Play or Mana Charge)."""
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.p0_human_radio.isChecked(): 
            return

        # Don't handle during input wait
        if EngineCompat.is_waiting_for_user_input(self.gs):
            return

        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
            if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
                relevant_cmds.append((c, d))

        if not relevant_cmds: 
            return

        # Prioritize PLAY_CARD over MANA_CHARGE for hand cards
        play_cmd = None
        mana_cmd = None
        attack_cmd = None
        other_cmd = None

        for cmd, d in relevant_cmds:
            cmd_type = d.get('type', '')
            if cmd_type == 'PLAY_CARD':
                play_cmd = cmd
            elif cmd_type == 'MANA_CHARGE':
                mana_cmd = cmd
            elif cmd_type == 'ATTACK':
                attack_cmd = cmd
            elif not other_cmd:
                other_cmd = cmd

        # Priority: Play > Attack > Other > Mana Charge
        if play_cmd:
            self.execute_action(play_cmd)
        elif attack_cmd:
            self.execute_action(attack_cmd)
        elif other_cmd:
            self.execute_action(other_cmd)
        elif mana_cmd:
            self.execute_action(mana_cmd)

    def on_card_hovered(self, card_id: int) -> None:
        if card_id >= 0:
            card_data = self._get_card_by_id(card_id)
            if card_data:
                self.card_detail_panel.update_card(card_data)

    def _update_p0_controls_visibility(self):
        """Show human control panel only when P0 is human."""
        show = self.p0_human_radio.isChecked()
        self.p0_control_group.setVisible(show)

    def _get_card_by_id(self, card_id: int):
        """Retrieve card object from various possible CardDB implementations.
        Supports dict-like and native dm_ai_module CardDatabase.
        """
        db = self.card_db
        if db is None:
            return None
        # Standard dict
        try:
            if isinstance(db, dict):
                return db.get(card_id)
        except Exception:
            pass
        # Mapping-like
        try:
            if hasattr(db, '__contains__') and card_id in db:
                return db[card_id]
        except Exception:
            pass
        # Native wrappers
        for meth in ('get_card', 'get_by_id'):
            try:
                fn = getattr(db, meth, None)
                if callable(fn):
                    return fn(card_id)
            except Exception:
                pass
        # Fallback: iterate keys if supported
        try:
            for k in getattr(db, 'keys', lambda: [])():
                if int(k) == int(card_id):
                    return db[k]
        except Exception:
            pass
        return None

    def execute_action(self, action: Any) -> None:
        """
        Phase 7.1: Receiver Input Normalization
        Accepts Action, CommandDict, or ICommand and unifies execution.
        """
        self.last_action = action
        
        # 1. Wrap into ICommand (Phase 7 Requirement: "execute_action ... wraps to cmd_dict")
        # Ensure conversion to dict using unified pipeline first
        try:
             cmd_dict = ensure_executable_command(action)
        except:
             cmd_dict = None

        if cmd_dict:
            command = wrap_action(cmd_dict)
        else:
            # Fallback for raw action objects if ensure_executable_command didn't handle them
            command = wrap_action(action)

        if command:
            try:
                # 2. Execute via Command interface (delegates to EngineCompat if needed)
                try:
                    command.execute(self.gs)
                except Exception:
                    # Use EngineCompat when direct execute is unavailable
                    EngineCompat.ExecuteCommand(self.gs, command)

                # 3. Log using Command Dict (Phase 7.2)
                final_dict = command.to_dict()
                log_str = f"P0 {tr('Action')}: {final_dict.get('type', 'UNKNOWN')}"
                if 'to_zone' in final_dict:
                    log_str += f" -> {final_dict['to_zone']}"
                self.log_list.addItem(log_str)

                self.loop_recorder.record_action(str(final_dict))
                self.scenario_tools.record_action(str(final_dict))
            except Exception as e:
                self.log_list.addItem(tr("Execution Error: {error}").format(error=e))
        else:
            # Unified fallback: convert raw action to command and execute
            try:
                from dm_toolkit.unified_execution import ensure_executable_command
                cmd = ensure_executable_command(action)
                EngineCompat.ExecuteCommand(self.gs, cmd, self.card_db)
            except Exception:
                 pass

        if self.check_and_handle_input_wait(): return

        if dm_ai_module:
            pending_count = self.gs.get_pending_effect_count()
            # Basic checks for auto-pass
            act_type = getattr(action, 'type', None)
            if (act_type == dm_ai_module.ActionType.PASS or act_type == dm_ai_module.ActionType.MANA_CHARGE) and pending_count == 0:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            
        self.update_ui()

    def on_resolve_effect_from_stack(self, index: int) -> None:
        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        resolve_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
            if d.get('type') == 'RESOLVE_EFFECT':
                resolve_cmds.append((c, d))

        target_cmd = None
        for c, d in resolve_cmds:
            if d.get('slot_index') == index:
                target_cmd = c
                break
        if target_cmd:
            self.execute_action(target_cmd)
        elif len(resolve_cmds) == 1:
            self.execute_action(resolve_cmds[0][0])

    def check_and_handle_input_wait(self) -> bool:
        if not self.gs.waiting_for_user_input: return False
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.start_btn.setText(tr("Start Sim"))
        self.handle_user_input_request()
        self.update_ui()
        return True

    def handle_user_input_request(self) -> None:
        query = EngineCompat.get_pending_query(self.gs)
        if query.query_type == "SELECT_OPTION":
             options = query.options
             item, ok = QInputDialog.getItem(self, tr("Select Option"), tr("Choose an option:"), options, 0, False)
             if ok and item:
                 idx = options.index(item)
                 EngineCompat.EffectResolver_resume(self.gs, self.card_db, idx)
                 self.step_phase()
        elif query.query_type == "SELECT_TARGET":
             valid_targets = query.valid_targets
             if not valid_targets: return
             first_target_id = valid_targets[0]
             in_buffer = False
             buffer_cards = EngineCompat.get_effect_buffer(self.gs)
             for c in buffer_cards:
                 if c.instance_id == first_target_id:
                     in_buffer = True
                     break
             if in_buffer:
                 items = []
                 for tid in valid_targets:
                     found = next((c for c in buffer_cards if c.instance_id == tid), None)
                     if found: items.append(found)
                 min_sel = query.params.get('min', 1)
                 max_sel = query.params.get('max', 99)
                 dialog = CardSelectionDialog(tr("Select Cards"), tr("Please select cards:"), items, min_sel, max_sel, self, self.card_db)
                 if dialog.exec():
                     indices = dialog.get_selected_indices()
                     selected_instance_ids = [items[i].instance_id for i in indices]
                     EngineCompat.EffectResolver_resume(self.gs, self.card_db, cast(Any, selected_instance_ids))
                     self.step_phase()
                 return

    def step_phase(self) -> None:
        if self.is_processing: return
        self.is_processing = True
        try:
            if self.check_and_handle_input_wait(): return
            if self.gs.game_over:
                self.timer.stop()
                self.is_running = False
                self.start_btn.setText(tr("Start Sim"))
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (active_pid == 0 and self.p0_human_radio.isChecked()) or \
                       (active_pid == 1 and self.p1_human_radio.isChecked())

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
                    pending_info = EngineCompat.get_pending_effects_info(self.gs)
                    items = []
                    valid_cmds = []
                    for c, d in resolve_cmds:
                        idx = d.get('slot_index', -1)
                        if 0 <= idx < len(pending_info):
                            p_type, p_source, p_ctrl = pending_info[idx]
                            inst = None
                            try:
                                inst = self.gs.get_card_instance(p_source)
                                c_def = self.card_db.get(inst.card_id)
                                source_name = c_def.name if c_def else "Unknown"
                            except:
                                source_name = f"Instance {p_source}"
                            desc = f"Trigger: {p_type} (Controller: P{p_ctrl})"
                            items.append({'source_name': source_name, 'description': desc, 'card_id': inst.card_id if inst else -1})
                            valid_cmds.append(c)
                    if items:
                         dialog = CardSelectionDialog(tr("Select Trigger"), tr("Select effect to resolve:"), items, 1, 1, self, self.card_db)
                         if dialog.exec():
                             indices = dialog.get_selected_indices()
                             if indices:
                                 self.execute_action(valid_cmds[indices[0]])
                                 return
                if not cmds:
                    EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                    self.update_ui()
                return

            if not cmds:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            else:
                best_cmd = cmds[0]
                if best_cmd:
                    self.execute_action(best_cmd)
            self.update_ui()
        finally:
            self.is_processing = False
        
    def update_ui(self) -> None:
        if self.gs is None:
            return
        # During early startup, controller.initialize_game may call update_ui before widgets exist.
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
                try:
                    desc = describe_command(cmd, self.gs, self.card_db)
                except:
                    desc = str(cmd)
                self.log_list.addItem(desc)
            self.log_list.scrollToBottom()
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
        
        # Generate legal actions for the active human player
        legal_actions = []
        if active_pid == 0 and self.p0_human_radio.isChecked() and not self.gs.game_over:
             from dm_toolkit.commands import generate_legal_commands
             legal_actions = generate_legal_commands(self.gs, self.card_db)

        # Check for PASS action to enable Pass Button
        self.current_pass_action = None
        for cmd in legal_actions:
            try: d = cmd.to_dict()
            except: d = {}
            if d.get('type') == 'PASS' or d.get('legacy_original_type') == 'PASS':
                self.current_pass_action = cmd
                break

        if self.current_pass_action:
            self.pass_btn.setVisible(True)
            # self.pass_btn.setText(tr("Pass Turn")) # Optionally update text
        else:
            self.pass_btn.setVisible(False)

        def convert_zone(zone_cards: List[Any], hide: bool=False) -> List[Dict[str, Any]]:
            if hide: return [{'id': -1, 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]
            return [{'id': getattr(c, 'card_id', -1), 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]
            
        god_view = self.god_view_check.isChecked()
        self.p0_hand.update_cards(convert_zone(p0.hand), self.card_db, legal_actions=legal_actions)
        self.p0_mana.update_cards(convert_zone(p0.mana_zone), self.card_db, legal_actions=legal_actions)
        self.p0_battle.update_cards(convert_zone(p0.battle_zone), self.card_db, legal_actions=legal_actions)
        self.p0_shield.update_cards(convert_zone(p0.shield_zone), self.card_db, legal_actions=legal_actions)
        self.p0_graveyard.update_cards(convert_zone(p0.graveyard), self.card_db, legal_actions=legal_actions)
        self.p0_deck_zone.update_cards(convert_zone(p0.deck, hide=True), self.card_db, legal_actions=legal_actions)
        
        self.p1_hand.update_cards(convert_zone(p1.hand, hide=not god_view), self.card_db)
        self.p1_mana.update_cards(convert_zone(p1.mana_zone), self.card_db)
        self.p1_battle.update_cards(convert_zone(p1.battle_zone), self.card_db)
        self.p1_shield.update_cards(convert_zone(p1.shield_zone, hide=not god_view), self.card_db)
        self.p1_graveyard.update_cards(convert_zone(p1.graveyard), self.card_db)
        self.p1_deck_zone.update_cards(convert_zone(p1.deck, hide=True), self.card_db)

        pending = EngineCompat.get_pending_query(self.gs)
        if EngineCompat.is_waiting_for_user_input(self.gs) and pending is not None and getattr(pending, 'query_type', '') == "SELECT_TARGET":
            valid_targets = getattr(pending, 'valid_targets', [])
            params = getattr(pending, 'params', {})
            min_targets = params.get('min', 1) if hasattr(params, 'get') else 1
            max_targets = params.get('max', 99) if hasattr(params, 'get') else 99

            current = len(self.selected_targets)
            self.confirm_btn.setText(f"{tr('Confirm')} ({current}/{min_targets}-{max_targets})")
            self.confirm_btn.setVisible(True)
            self.confirm_btn.setEnabled(current >= min_targets)

            zones = [
                self.p0_hand, self.p0_mana, self.p0_battle, self.p0_shield, self.p0_graveyard,
                self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard
            ]
            for zone in zones:
                for target_id in self.selected_targets:
                    zone.set_card_selected(target_id, True)
        else:
            self.confirm_btn.setVisible(False)

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
