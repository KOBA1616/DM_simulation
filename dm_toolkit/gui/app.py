# -*- coding: utf-8 -*-
import sys
import os
import random
import json
import csv

# Ensure bin is in path for dm_ai_module
current_dir = os.path.dirname(os.path.abspath(__file__))
bin_dir = os.path.join(current_dir, "../../../bin")
if os.path.exists(bin_dir):
    sys.path.append(bin_dir)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QDockWidget, QTabWidget,
    QInputDialog, QToolBar
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer
import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.localization import tr, describe_command
from dm_toolkit.gui.deck_builder import DeckBuilder
from dm_toolkit.gui.card_editor import CardEditor
from dm_toolkit.gui.widgets.scenario_tools import ScenarioToolsDock
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.stack_view import StackViewWidget
from dm_toolkit.gui.widgets.loop_recorder import LoopRecorderWidget
from dm_toolkit.gui.dialogs.selection_dialog import CardSelectionDialog

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM AI Simulator")
        self.resize(1600, 900)
        
        # Game State
        self.gs = dm_ai_module.GameState(42)
        self.gs.setup_test_duel()
        # Load card database: prefer C++ JsonLoader if available, otherwise fallback to Python JSON
        self.card_db = EngineCompat.JsonLoader_load_cards("data/cards.json")
        if self.card_db:
             EngineCompat.PhaseManager_start_game(self.gs, self.card_db)
        else:
            try:
                with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cards.json'), 'r', encoding='utf-8') as _f:
                    self.card_db = json.load(_f)
            except Exception:
                # Fallback to empty DB to allow UI to start; functionality will be limited
                self.card_db = []
        
        self.p0_deck_ids = None
        self.p1_deck_ids = None
        self.last_action = None
        self.selected_targets = []
        self.last_command_index = 0  # For incremental command logging

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_phase)
        self.is_running = False
        self.is_processing = False

        # Toolbar Setup
        self.toolbar = QToolBar(tr("Main Toolbar"), self)
        self.toolbar.setObjectName("MainToolbar")
        self.addToolBar(self.toolbar)

        deck_act = QAction(tr("Deck Builder"), self)
        deck_act.setToolTip(tr("Open the Deck Builder tool"))
        deck_act.triggered.connect(self.open_deck_builder)
        self.toolbar.addAction(deck_act)

        card_act = QAction(tr("Card Editor"), self)
        card_act.setToolTip(tr("Open the Card Editor tool"))
        card_act.triggered.connect(self.open_card_editor)
        self.toolbar.addAction(card_act)

        self.scen_act = QAction(tr("Scenario Mode"), self)
        self.scen_act.setToolTip(tr("Toggle Scenario Mode for testing specific game states"))
        self.scen_act.setCheckable(True)
        self.scen_act.triggered.connect(self.toggle_scenario_mode)
        self.toolbar.addAction(self.scen_act)

        sim_act = QAction(tr("Batch Simulation"), self)
        sim_act.setToolTip(tr("Run multiple games for statistical analysis"))
        sim_act.triggered.connect(self.open_simulation_dialog)
        self.toolbar.addAction(sim_act)

        # AI Tools Button (MCTS)
        ai_act = QAction(tr("AI Analysis"), self)
        ai_act.setToolTip(tr("Toggle the MCTS Analysis dock"))
        ai_act.triggered.connect(lambda: self.mcts_dock.setVisible(not self.mcts_dock.isVisible()))
        self.toolbar.addAction(ai_act)

        # Loop Recorder Button
        loop_act = QAction(tr("Loop Recorder"), self)
        loop_act.setToolTip(tr("Toggle the Loop Recorder dock"))
        loop_act.triggered.connect(lambda: self.loop_dock.setVisible(not self.loop_dock.isVisible()))
        self.toolbar.addAction(loop_act)

        # UI Setup
        self.info_dock = QDockWidget(tr("Game Info & Controls"), self)
        self.info_dock.setObjectName("InfoDock")
        self.info_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.info_panel = QWidget()
        self.info_panel.setMinimumWidth(300)
        self.info_layout = QVBoxLayout(self.info_panel)
        self.info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.info_dock)

        # 1. Top Section
        self.top_section_group = QGroupBox(tr("Game Status & Operations"))
        top_layout = QVBoxLayout()
        
        status_layout = QHBoxLayout()
        self.turn_label = QLabel(f"{tr('Turn')}: 1")
        self.turn_label.setStyleSheet("font-weight: bold;")
        self.phase_label = QLabel(f"{tr('Phase')}: START")
        self.active_label = QLabel(f"{tr('Active')}: P0")
        status_layout.addWidget(self.turn_label)
        status_layout.addWidget(self.phase_label)
        status_layout.addWidget(self.active_label)
        top_layout.addLayout(status_layout)

        self.card_detail_panel = CardDetailPanel()
        top_layout.addWidget(self.card_detail_panel)
        
        game_ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start Sim"))
        self.start_btn.setToolTip(f"{tr('Start/Stop continuous simulation')} (F5)")
        self.start_btn.setShortcut("F5")
        self.start_btn.clicked.connect(self.toggle_simulation)
        game_ctrl_layout.addWidget(self.start_btn)

        self.step_button = QPushButton(tr("Step"))
        self.step_button.setToolTip(f"{tr('Advance game by one step')} (Space)")
        self.step_button.setShortcut("Space")
        self.step_button.clicked.connect(self.step_phase)
        game_ctrl_layout.addWidget(self.step_button)

        self.confirm_btn = QPushButton(tr("Confirm Selection"))
        self.confirm_btn.setToolTip(f"{tr('Confirm target selection')} (Enter)")
        self.confirm_btn.setShortcut("Return")
        self.confirm_btn.clicked.connect(self.confirm_selection)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        game_ctrl_layout.addWidget(self.confirm_btn)

        self.reset_btn = QPushButton(tr("Reset"))
        self.reset_btn.setToolTip(f"{tr('Reset the game state')} (Ctrl+R)")
        self.reset_btn.setShortcut("Ctrl+R")
        self.reset_btn.clicked.connect(self.reset_game)
        game_ctrl_layout.addWidget(self.reset_btn)
        top_layout.addLayout(game_ctrl_layout)

        self.top_section_group.setLayout(top_layout)
        self.info_layout.addWidget(self.top_section_group)

        # 2. Bottom Section
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

        self.scenario_editor_button = QPushButton(tr("Scenario Editor"))
        self.scenario_editor_button.clicked.connect(self.open_scenario_editor)
        tools_layout.addWidget(self.scenario_editor_button)

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
        self.info_layout.addWidget(self.bottom_section_group)
        
        self.info_layout.addStretch()
        
        # Board Panel
        self.board_panel = QWidget()
        self.board_layout = QVBoxLayout(self.board_panel)
        self.board_layout.setContentsMargins(0, 0, 0, 0)
        
        self.p1_zones = QWidget()
        self.p1_layout = QVBoxLayout(self.p1_zones)
        self.p1_hand = ZoneWidget("P1 手札")
        self.p1_mana = ZoneWidget("P1 マナ")
        self.p1_graveyard = ZoneWidget("P1 墓地")
        self.p1_battle = ZoneWidget("P1 バトルゾーン")
        self.p1_shield = ZoneWidget("P1 シールド")
        self.p1_deck_zone = ZoneWidget("P1 デッキ")
        
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
        self.p0_battle = ZoneWidget("P0 バトルゾーン")
        self.p0_deck_zone = ZoneWidget("P0 デッキ")
        self.p0_shield = ZoneWidget("P0 シールド")
        self.p0_mana = ZoneWidget("P0 マナ")
        self.p0_graveyard = ZoneWidget("P0 墓地")
        self.p0_hand = ZoneWidget("P0 手札")
        
        # Connect Context Menu Signals
        self.p0_hand.action_triggered.connect(self.execute_action)
        self.p0_mana.action_triggered.connect(self.execute_action)
        self.p0_battle.action_triggered.connect(self.execute_action)
        self.p0_graveyard.action_triggered.connect(self.execute_action)

        self.p0_hand.card_clicked.connect(self.on_card_clicked)
        self.p0_mana.card_clicked.connect(self.on_card_clicked)
        self.p0_battle.card_clicked.connect(self.on_card_clicked)
        self.p0_graveyard.card_clicked.connect(self.on_card_clicked)
        
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
        
        # 1. Pending Stack Dock (Right) - Created first to be top
        self.stack_dock = QDockWidget(tr("Pending Effects"), self)
        self.stack_dock.setObjectName("StackDock")
        self.stack_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.stack_view = StackViewWidget()
        self.stack_view.effect_resolved.connect(self.on_resolve_effect_from_stack)
        self.stack_dock.setWidget(self.stack_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stack_dock)

        # 2. Log Dock (Right)
        self.log_dock = QDockWidget(tr("Logs"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_list = QListWidget()
        self.log_dock.setWidget(self.log_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        
        # Stack above Log
        self.splitDockWidget(self.stack_dock, self.log_dock, Qt.Orientation.Vertical)

        # 3. MCTS Dock (Right) - Default Hidden
        self.mcts_dock = QDockWidget(tr("MCTS Analysis"), self)
        self.mcts_dock.setObjectName("MCTSDock")
        self.mcts_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.mcts_view = MCTSView()
        self.mcts_dock.setWidget(self.mcts_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.mcts_dock)
        self.mcts_dock.hide()

        # 4. Loop Recorder Dock (Left) - Default Hidden
        self.loop_dock = QDockWidget(tr("Loop Recorder"), self)
        self.loop_dock.setObjectName("LoopDock")
        self.loop_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.loop_recorder = LoopRecorderWidget(lambda: self.gs)
        self.loop_dock.setWidget(self.loop_recorder)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.loop_dock)
        self.loop_dock.hide()

        # 5. Scenario Tools Dock (Bottom/Right) - Default Hidden
        self.scenario_tools = ScenarioToolsDock(self, self.gs, self.card_db)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.scenario_tools)
        self.scenario_tools.hide()

        self.update_ui()
        self.showMaximized()
        
    def open_deck_builder(self):
        self.deck_builder = DeckBuilder(self.card_db)
        self.deck_builder.show()

    def open_card_editor(self):
        self.card_editor = CardEditor("data/cards.json")
        self.card_editor.data_saved.connect(self.reload_card_data)
        self.card_editor.show()

    def open_scenario_editor(self):
        """Open the Scenario Editor if available; otherwise show info."""
        try:
            from dm_toolkit.gui import scenario_editor
            # Expect ScenarioEditor to be a dialog/class taking card_db
            if hasattr(scenario_editor, 'ScenarioEditor'):
                self.scenario_editor = scenario_editor.ScenarioEditor(self.card_db, parent=self)
                self.scenario_editor.show()
                return
        except Exception:
            pass
        QMessageBox.information(self, tr("Scenario Editor"), tr("Scenario Editor not available in this environment."))

    def reload_card_data(self):
        try:
            self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
            self.civ_map = self.build_civ_map()

            # Refresh Deck Builder if open
            if hasattr(self, 'deck_builder') and self.deck_builder.isVisible():
                self.deck_builder.reload_database()

            # Update Scenario Tools DB
            self.scenario_tools.set_game_state(self.gs, self.card_db)

            self.log_list.addItem(tr("Card Data Reloaded from Editor Save"))
        except Exception as e:
            self.log_list.addItem(f"{tr('Error reloading cards')}: {e}")

    def toggle_scenario_mode(self, checked):
        if checked:
            self.scenario_tools.show()
            self.log_list.addItem(tr("Scenario Mode Enabled"))
            # Stop any simulation
            if self.is_running:
                self.toggle_simulation()
        else:
            self.scenario_tools.hide()
            self.log_list.addItem(tr("Scenario Mode Disabled"))

    def open_simulation_dialog(self):
        self.sim_dialog = SimulationDialog(self.card_db, self)
        self.sim_dialog.show()

    def load_deck_p0(self):
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self, tr("Load Deck P0"), "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                self.p0_deck_ids = deck_ids
                self.reset_game()
                self.log_list.addItem(f"{tr('Loaded Deck for P0')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def load_deck_p1(self):
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self, tr("Load Deck P1"), "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                self.p1_deck_ids = deck_ids
                self.reset_game()
                self.log_list.addItem(f"{tr('Loaded Deck for P1')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def show_help(self):
        QMessageBox.information(self, tr("Help / Manual"), "Help text...")

    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText(tr("Start Sim"))
            self.is_running = False
        else:
            self.timer.start(500)
            self.start_btn.setText(tr("Stop Sim"))
            self.is_running = True

    def reset_game(self):
        self.timer.stop()
        self.is_running = False
        self.selected_targets = []
        self.confirm_btn.setVisible(False)
        self.start_btn.setText(tr("Start Sim"))
        self.gs = dm_ai_module.GameState(random.randint(0, 10000))
        self.gs.setup_test_duel()
        if self.p0_deck_ids: self.gs.set_deck(0, self.p0_deck_ids)
        if self.p1_deck_ids: self.gs.set_deck(1, self.p1_deck_ids)
        EngineCompat.PhaseManager_start_game(self.gs, self.card_db)
        self.scenario_tools.set_game_state(self.gs, self.card_db)
        self.log_list.clear()
        self.log_list.addItem(tr("Game Reset"))
        self.last_command_index = 0 # Reset command index
        self.update_ui()

    def confirm_selection(self):
        if not EngineCompat.is_waiting_for_user_input(self.gs): return

        query = EngineCompat.get_pending_query(self.gs)
        min_targets = query.params.get('min', 1)

        if len(self.selected_targets) < min_targets:
            QMessageBox.warning(self, "Invalid Selection", f"Please select at least {min_targets} target(s).")
            return

        targets = list(self.selected_targets)
        self.selected_targets = []
        self.confirm_btn.setVisible(False)

        EngineCompat.EffectResolver_resume(self.gs, self.card_db, targets)
        # self.log_list.addItem(f"Resumed with targets: {targets}")
        self.step_phase()

    def on_card_clicked(self, card_id, instance_id):
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.p0_human_radio.isChecked():
            return

        if EngineCompat.is_waiting_for_user_input(self.gs):
             pending = EngineCompat.get_pending_query(self.gs)
             if pending.query_type == "SELECT_TARGET":
                 valid_targets = pending.valid_targets

                 # Check if we need to show a popup (Searching from Buffer/Stack)
                 # If valid_targets are in BUFFER or STACK, usually they aren't clickable on board widgets
                 # UNLESS the board widget (e.g. StackView) emits this signal.

                 if instance_id in valid_targets:
                     if instance_id in self.selected_targets:
                         self.selected_targets.remove(instance_id)
                     else:
                         query_max = pending.params.get('max', 99)
                         if len(self.selected_targets) < query_max:
                             self.selected_targets.append(instance_id)
                         else:
                             # self.log_list.addItem(f"Max targets reached ({query_max})")
                             return
                     self.update_ui()
                 else:
                     pass # self.log_list.addItem("Invalid target selected.")
             return

        actions = EngineCompat.ActionGenerator_generate_legal_actions(
            self.gs, self.card_db
        )
        relevant_actions = [a for a in actions if EngineCompat.get_action_source_id(a) == instance_id]

        if not relevant_actions:
            # self.log_list.addItem(f"{tr('No actions for card')} {card_id} (Inst: {instance_id})")
            return

        if len(relevant_actions) == 1:
            self.execute_action(relevant_actions[0])
        else:
            # self.log_list.addItem(tr("Multiple actions found. Executing first."))
            self.execute_action(relevant_actions[0])

    def on_card_hovered(self, card_id):
        if card_id >= 0:
            card_data = self.card_db.get(card_id)
            if card_data:
                self.card_detail_panel.update_card(card_data)

    def execute_action(self, action):
        self.last_action = action
        EngineCompat.EffectResolver_resolve_action(
            self.gs, action, self.card_db
        )
        # self.log_list.addItem(f"P0 {tr('Action')}: {action.to_string()}")
        self.loop_recorder.record_action(action.to_string())
        self.scenario_tools.record_action(action.to_string())
        
        if EngineCompat.is_waiting_for_user_input(self.gs):
            self.handle_user_input_request()
            return

        # Refactored phase logic: Check pending effects
        pending_count = self.gs.get_pending_effect_count()
        if (action.type == dm_ai_module.ActionType.PASS or action.type == dm_ai_module.ActionType.MANA_CHARGE) and pending_count == 0:
            EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            
        self.update_ui()

    def on_resolve_effect_from_stack(self, index):
        # Generate actions and find RESOLVE_EFFECT with matching slot index?
        # OR usually RESOLVE_EFFECT doesn't expose slot_index in ActionGenerator if it only generates ONE action (resolve top).
        # We need to check if we can arbitrarily resolve.
        # If PendingEffect list is exposed, we can assume we want to resolve the one at 'index'.

        # NOTE: C++ engine might strict about resolving TOP effect (LIFO).
        # But for simultaneous triggers, we can choose order.
        # If we are in a state where we can choose, ActionGenerator should return multiple RESOLVE_EFFECT actions.

        actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
        resolve_actions = [a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]

        target_action = None
        for a in resolve_actions:
            # Check if action corresponds to our index.
            # Assuming slot_index holds the index in pending_effects vector.
            if EngineCompat.get_action_slot_index(a) == index:
                target_action = a
                break

        if target_action:
            self.execute_action(target_action)
        else:
             # If exact index match fail, maybe it's implicitly the only one available?
             if len(resolve_actions) == 1:
                  self.execute_action(resolve_actions[0])
             else:
                  pass # self.log_list.addItem(f"Cannot resolve effect at index {index}. (Not in legal actions)")

    def handle_user_input_request(self):
        query = EngineCompat.get_pending_query(self.gs)

        if query.query_type == "SELECT_OPTION":
             options = query.options
             item, ok = QInputDialog.getItem(self, "Select Option", "Choose an option:", options, 0, False)
             if ok and item:
                 idx = options.index(item)
                 EngineCompat.EffectResolver_resume(self.gs, self.card_db, idx)
                 self.step_phase()

        elif query.query_type == "SELECT_TARGET":
             # CHECK FOR BUFFER/STACK SELECTION (POPUP REQUIRED)
             valid_targets = query.valid_targets
             if not valid_targets:
                 # Should not happen, but safe guard
                 return

             # Inspect first target to guess location
             first_target_id = valid_targets[0]
             # We need to know if this card is in Buffer/Stack.
             # Use GameState helper or check effect_buffer.

             # Check if targets are in effect buffer
             in_buffer = False
             buffer_cards = EngineCompat.get_effect_buffer(self.gs)
             for c in buffer_cards:
                 if c.instance_id == first_target_id:
                     in_buffer = True
                     break

             if in_buffer:
                 # Show Popup
                 items = []
                 for tid in valid_targets:
                     # Find card in buffer
                     found = next((c for c in buffer_cards if c.instance_id == tid), None)
                     if found:
                         items.append(found) # CardInstance object (has card_id)

                 min_sel = query.params.get('min', 1)
                 max_sel = query.params.get('max', 99)

                 dialog = CardSelectionDialog("Select Cards", "Please select cards:", items, min_sel, max_sel, self, self.card_db)
                 if dialog.exec():
                     indices = dialog.get_selected_indices()
                     selected_instance_ids = [items[i].instance_id for i in indices]
                     EngineCompat.EffectResolver_resume(self.gs, self.card_db, selected_instance_ids)
                     self.step_phase()
                 return

             # Otherwise, standard UI selection (Hand/Battle/Mana)
             # self.log_list.addItem(f"Please select {query.params['min']} target(s).")
             self.update_ui()

    def step_phase(self):
        if self.is_processing: return
        self.is_processing = True
        
        try:
            if EngineCompat.is_waiting_for_user_input(self.gs):
                self.handle_user_input_request()
                return

            if self.gs.game_over:
                self.timer.stop()
                self.is_running = False
                self.start_btn.setText(tr("Start Sim"))
                winner = self.gs.winner
                # self.log_list.addItem(f"{tr('Game Over! Winner')}: P{winner}")
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (active_pid == 0 and self.p0_human_radio.isChecked()) or \
                       (active_pid == 1 and self.p1_human_radio.isChecked())

            if is_human:
                actions = EngineCompat.ActionGenerator_generate_legal_actions(
                    self.gs, self.card_db
                )

                # CHECK FOR TRIGGER SELECTION (Multiple RESOLVE_EFFECT)
                resolve_actions = [a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]
                if len(resolve_actions) > 1:
                    # Show Popup for Trigger Order/Selection
                    # We need details about pending effects.
                    # Use get_pending_effects_info(state) -> list of (type, source_id, controller)
                    # This might not map 1:1 if multiple effects are from same source?
                    # But RESOLVE_EFFECT usually maps to the top of stack OR specific index if implemented.

                    # Assuming we can match actions to pending effects via slot_index
                    pending_info = EngineCompat.get_pending_effects_info(self.gs)

                    # Map actions to descriptions
                    items = []
                    valid_actions = []

                    for act in resolve_actions:
                        idx = EngineCompat.get_action_slot_index(act)
                        if 0 <= idx < len(pending_info):
                            p_type, p_source, p_ctrl = pending_info[idx]
                            c_def = self.card_db.get(p_source, None) # p_source is instance_id? No, usually card_id or we need look up.
                            # Memory says: "get_pending_effects_info(state) returns a list of tuples (type, source_instance_id, controller)"

                            source_name = "Unknown"
                            if c_def: # If it was card_id
                                source_name = c_def.name
                            else:
                                # Try to look up instance
                                try:
                                    inst = self.gs.get_card_instance(p_source)
                                    c_def = self.card_db.get(inst.card_id)
                                    source_name = c_def.name
                                except:
                                    source_name = f"Instance {p_source}"

                            desc = f"Trigger: {p_type} (Controller: P{p_ctrl})"
                            items.append({'source_name': source_name, 'description': desc, 'card_id': inst.card_id if 'inst' in locals() else -1})
                            valid_actions.append(act)

                    if items:
                         dialog = CardSelectionDialog("Select Trigger", "Select effect to resolve:", items, 1, 1, self, self.card_db)
                         if dialog.exec():
                             indices = dialog.get_selected_indices()
                             if indices:
                                 self.execute_action(valid_actions[indices[0]])
                                 return # Action executed, loop will continue

                if not actions:
                    EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                    # self.log_list.addItem(f"P{active_pid} {tr('Auto-Pass')}")
                    self.update_ui()
                return

            actions = EngineCompat.ActionGenerator_generate_legal_actions(
                self.gs, self.card_db
            )

            if not actions:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                # self.log_list.addItem(f"P{active_pid} {tr('Auto-Pass')}")
            else:
                best_action = actions[0] # Fallback
                
                if best_action:
                    self.last_action = best_action
                    EngineCompat.EffectResolver_resolve_action(
                        self.gs, best_action, self.card_db
                    )
                    # self.log_list.addItem(f"P{active_pid} {tr('AI Action')}: {best_action.to_string()}")
                    self.loop_recorder.record_action(best_action.to_string())
                    self.scenario_tools.record_action(best_action.to_string())

                    if EngineCompat.is_waiting_for_user_input(self.gs):
                         # self.log_list.addItem("AI Paused for Input (Not Implemented). Stopping Sim.")
                         self.timer.stop()
                         self.is_running = False
                         self.start_btn.setText(tr("Start Sim"))
                         return

                    # Refactored Phase Logic
                    pending_count = self.gs.get_pending_effect_count()
                    if (best_action.type == dm_ai_module.ActionType.PASS or best_action.type == dm_ai_module.ActionType.MANA_CHARGE) and pending_count == 0:
                        EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)

            self.update_ui()
        finally:
            self.is_processing = False
        
    def update_ui(self):
        # Guard attributes on GameState that may not exist in minimal dm_ai_module builds
        turn_number = EngineCompat.get_turn_number(self.gs)
        current_phase = EngineCompat.get_current_phase(self.gs)
        active_pid = EngineCompat.get_active_player_id(self.gs)
        self.turn_label.setText(f"{tr('Turn')}: {turn_number}")
        self.phase_label.setText(f"{tr('Phase')}: {current_phase}")
        self.active_label.setText(f"{tr('Active')}: P{active_pid}")
        
        # Update Stack View
        self.stack_view.update_state(self.gs, self.card_db)

        # ---------------------------------------------------------------------
        # COMMAND LOG UPDATE (safe: GameState may not expose command_history)
        # ---------------------------------------------------------------------
        history = EngineCompat.get_command_history(self.gs)
        try:
            current_len = len(history)
        except Exception:
            current_len = 0
        if current_len > self.last_command_index:
            for i in range(self.last_command_index, current_len):
                cmd = history[i]
                try:
                    desc = describe_command(cmd, self.gs, self.card_db)
                except Exception:
                    desc = str(cmd)
                self.log_list.addItem(desc)
            self.log_list.scrollToBottom()
            self.last_command_index = current_len

        # Use EngineCompat to get players or fallback
        class _P: pass
        p0 = EngineCompat.get_player(self.gs, 0) or _P()
        p1 = EngineCompat.get_player(self.gs, 1) or _P()
        
        # Calculate legal actions for context menus
        # active_pid may have been determined earlier from turn labels
        active_pid = active_pid if 'active_pid' in locals() else EngineCompat.get_active_player_id(self.gs)
        legal_actions = []
        if active_pid == 0 and self.p0_human_radio.isChecked():
             legal_actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)

        def convert_zone(zone_cards, hide=False):
            if hide:
                return [{'id': -1, 'tapped': c.is_tapped, 'instance_id': c.instance_id} for c in zone_cards]
            return [{'id': c.card_id, 'tapped': c.is_tapped, 'instance_id': c.instance_id} for c in zone_cards]
            
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

        # Pending user input may not exist on all GameState builds
        pending = EngineCompat.get_pending_query(self.gs)
        if EngineCompat.is_waiting_for_user_input(self.gs) and pending is not None and getattr(pending, 'query_type', '') == "SELECT_TARGET":
            valid_targets = getattr(pending, 'valid_targets', [])
            min_targets = getattr(getattr(pending, 'params', {}), 'get', lambda k, d=None: d)('min', 1) if hasattr(getattr(pending, 'params', None), 'get') else 1
            max_targets = pending.params.get('max', 99)

            # Update button text with count
            current = len(self.selected_targets)
            self.confirm_btn.setText(f"{tr('Confirm')} ({current}/{min_targets}-{max_targets})")
            self.confirm_btn.setVisible(True)
            self.confirm_btn.setEnabled(current >= min_targets)

            # Apply selections visually
            zones = [
                self.p0_hand, self.p0_mana, self.p0_battle, self.p0_shield, self.p0_graveyard,
                self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard
            ]
            for zone in zones:
                for target_id in self.selected_targets:
                    zone.set_card_selected(target_id, True)
        else:
            self.confirm_btn.setVisible(False)

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
