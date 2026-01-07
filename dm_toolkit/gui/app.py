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
from dm_toolkit.gui.card_editor import CardEditor
from dm_toolkit.gui.widgets.scenario_tools import ScenarioToolsDock
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.stack_view import StackViewWidget
from dm_toolkit.gui.widgets.loop_recorder import LoopRecorderWidget
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

        # UI Setup
        self.info_dock = QDockWidget(tr("Game Info & Controls"), self)
        self.info_dock.setObjectName("InfoDock")
        self.info_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.info_panel = QWidget()
        self.info_panel.setMinimumWidth(300)
        self.info_layout = QVBoxLayout(self.info_panel)
        self.info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.info_dock)

        # Top Section
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
        self.start_btn.setShortcut("F5")
        self.start_btn.clicked.connect(self.toggle_simulation)
        game_ctrl_layout.addWidget(self.start_btn)

        self.step_button = QPushButton(tr("Step"))
        self.step_button.setShortcut("Space")
        self.step_button.clicked.connect(self.step_phase)
        game_ctrl_layout.addWidget(self.step_button)

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
        self.info_layout.addWidget(self.top_section_group)

        # Bottom Section
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
        self.p1_hand = ZoneWidget(tr("P1 Hand"))
        self.p1_mana = ZoneWidget(tr("P1 Mana"))
        self.p1_graveyard = ZoneWidget(tr("P1 Graveyard"))
        self.p1_battle = ZoneWidget(tr("P1 Battle Zone"))
        self.p1_shield = ZoneWidget(tr("P1 Shield"))
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
        self.p0_shield = ZoneWidget(tr("P0 Shield"))
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
        
        # Docks
        self.stack_dock = QDockWidget(tr("Pending Effects"), self)
        self.stack_dock.setObjectName("StackDock")
        self.stack_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.stack_view = StackViewWidget()
        self.stack_view.effect_resolved.connect(self.on_resolve_effect_from_stack)
        self.stack_dock.setWidget(self.stack_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stack_dock)

        self.log_dock = QDockWidget(tr("Logs"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_dock.setWidget(self.log_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        
        self.splitDockWidget(self.stack_dock, self.log_dock, Qt.Orientation.Vertical)

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

    def open_scenario_editor(self) -> None:
        try:
            from dm_toolkit.gui.editor import scenario_editor
            if hasattr(scenario_editor, 'ScenarioEditor'):
                self.scenario_editor = scenario_editor.ScenarioEditor(self.card_db, parent=self)
                self.scenario_editor.show()
                return
        except Exception:
            pass
        QMessageBox.information(self, tr("Scenario Editor"), tr("Scenario Editor not available."))

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
        self.start_btn.setText(tr("Start Sim"))
        self.log_list.clear()

        # Delegate logic to controller
        self.controller.reset_game(self.p0_deck_ids, self.p1_deck_ids)
        self.gs = self.controller.gs # Sync back reference

        self.scenario_tools.set_game_state(self.gs, self.card_db)
        self.last_command_index = 0
        self.update_ui()

    def confirm_selection(self) -> None:
        if not EngineCompat.is_waiting_for_user_input(self.gs): return
        query = EngineCompat.get_pending_query(self.gs)
        min_targets = query.params.get('min', 1)
        if len(self.selected_targets) < min_targets:
            # QMessageBox.warning(self, "Invalid Selection", f"Please select at least {min_targets} target(s).")
            # Converted to:
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

        actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
        relevant_actions = [a for a in actions if EngineCompat.get_action_source_id(a) == instance_id]

        if not relevant_actions: return
        self.execute_action(relevant_actions[0])

    def on_card_hovered(self, card_id: int) -> None:
        if card_id >= 0:
            card_data = self.card_db.get(card_id)
            if card_data: self.card_detail_panel.update_card(card_data)

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
                command.execute(self.gs)

                # 3. Log using Command Dict (Phase 7.2)
                final_dict = command.to_dict()
                log_str = f"P0 {tr('Action')}: {final_dict.get('type', 'UNKNOWN')}"
                if 'to_zone' in final_dict:
                    log_str += f" -> {final_dict['to_zone']}"
                self.log_list.addItem(log_str)

                self.loop_recorder.record_action(str(final_dict))
                self.scenario_tools.record_action(str(final_dict))
            except Exception as e:
                self.log_list.addItem(f"Execution Error: {e}")
                # Fallback?
                try:
                    EngineCompat.EffectResolver_resolve_action(self.gs, action, self.card_db)
                except:
                    pass
        else:
            # Fallback legacy path
            try:
                 EngineCompat.ExecuteCommand(self.gs, action, self.card_db)
            except:
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
        actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
        resolve_actions = []
        if dm_ai_module:
            resolve_actions = [a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]

        target_action = None
        for a in resolve_actions:
            if EngineCompat.get_action_slot_index(a) == index:
                target_action = a
                break
        if target_action: self.execute_action(target_action)
        elif len(resolve_actions) == 1: self.execute_action(resolve_actions[0])

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

            if is_human:
                actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
                resolve_actions = []
                if dm_ai_module:
                    resolve_actions = [a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]

                if len(resolve_actions) > 1:
                    pending_info = EngineCompat.get_pending_effects_info(self.gs)
                    items = []
                    valid_actions = []
                    for act in resolve_actions:
                        idx = EngineCompat.get_action_slot_index(act)
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
                            valid_actions.append(act)
                    if items:
                         dialog = CardSelectionDialog(tr("Select Trigger"), tr("Select effect to resolve:"), items, 1, 1, self, self.card_db)
                         if dialog.exec():
                             indices = dialog.get_selected_indices()
                             if indices:
                                 self.execute_action(valid_actions[indices[0]])
                                 return
                if not actions:
                    EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
                    self.update_ui()
                return

            actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)
            if not actions:
                EngineCompat.PhaseManager_next_phase(self.gs, self.card_db)
            else:
                best_action = actions[0]
                if best_action:
                    self.execute_action(best_action)
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
        active_pid = EngineCompat.get_active_player_id(self.gs)
        self.turn_label.setText(f"{tr('Turn')}: {turn_number}")
        self.phase_label.setText(f"{tr('Phase')}: {current_phase}")
        self.active_label.setText(f"{tr('Active')}: P{active_pid}")
        
        self.stack_view.update_state(self.gs, self.card_db)

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
        
        legal_actions = []
        if active_pid == 0 and self.p0_human_radio.isChecked():
             legal_actions = EngineCompat.ActionGenerator_generate_legal_actions(self.gs, self.card_db)

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
