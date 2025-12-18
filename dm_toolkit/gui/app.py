import sys
import os
import random
import json
import csv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QDockWidget, QTabWidget,
    QInputDialog
)
from PyQt6.QtCore import Qt, QTimer
import dm_ai_module
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.deck_builder import DeckBuilder
from dm_toolkit.gui.card_editor import CardEditor
from dm_toolkit.gui.editor.scenario_editor import ScenarioEditor
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.simulation_dialog import SimulationDialog

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM AI Simulator")
        self.resize(1400, 900)
        
        # Game State
        self.gs = dm_ai_module.GameState(42)
        self.gs.setup_test_duel()
        self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        self.civ_map = self.load_civilizations_from_json("data/cards.json")
        
        self.p0_deck_ids = None
        self.p1_deck_ids = None
        self.last_action = None

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_phase)
        self.is_running = False
        self.is_processing = False

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
        self.start_btn.clicked.connect(self.toggle_simulation)
        game_ctrl_layout.addWidget(self.start_btn)

        self.step_button = QPushButton(tr("Step"))
        self.step_button.clicked.connect(self.step_phase)
        game_ctrl_layout.addWidget(self.step_button)

        self.reset_btn = QPushButton(tr("Reset"))
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
        
        self.mcts_dock = QDockWidget(tr("MCTS Analysis"), self)
        self.mcts_dock.setObjectName("MCTSDock")
        self.mcts_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.mcts_view = MCTSView()
        self.mcts_dock.setWidget(self.mcts_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.mcts_dock)
        
        self.log_dock = QDockWidget(tr("Logs"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_list = QListWidget()
        self.log_dock.setWidget(self.log_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        
        self.update_ui()
        self.showMaximized()
        
    def load_civilizations_from_json(self, filepath):
        civ_map = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cards = json.load(f)
                for card in cards:
                    cid = card.get('id')
                    civ = card.get('civilization')
                    if cid is not None and civ is not None:
                        civ_map[cid] = civ
        except Exception as e:
            print(f"Error loading civilizations from json: {e}")
        return civ_map

    def open_deck_builder(self):
        self.deck_builder = DeckBuilder(self.card_db, self.civ_map)
        self.deck_builder.show()

    def open_card_editor(self):
        self.card_editor = CardEditor("data/cards.json")
        self.card_editor.show()

    def open_scenario_editor(self):
        self.scenario_editor = ScenarioEditor(self)
        self.scenario_editor.show()

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
        self.start_btn.setText(tr("Start Sim"))
        self.gs = dm_ai_module.GameState(random.randint(0, 10000))
        self.gs.setup_test_duel()
        if self.p0_deck_ids: self.gs.set_deck(0, self.p0_deck_ids)
        if self.p1_deck_ids: self.gs.set_deck(1, self.p1_deck_ids)
        dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        self.log_list.clear()
        self.log_list.addItem(tr("Game Reset"))
        self.update_ui()

    def on_card_clicked(self, card_id, instance_id):
        if self.gs.active_player_id != 0 or not self.p0_human_radio.isChecked():
            return

        if self.gs.waiting_for_user_input:
             if self.gs.pending_query.query_type == "SELECT_TARGET":
                 valid_targets = self.gs.pending_query.valid_targets
                 if instance_id in valid_targets:
                     dm_ai_module.EffectResolver.resume(self.gs, self.card_db, [instance_id])
                     self.log_list.addItem(f"Resumed with target: {instance_id}")
                     self.step_phase()
                     return
                 else:
                     self.log_list.addItem("Invalid target selected.")
             return

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(
            self.gs, self.card_db
        )
        relevant_actions = [a for a in actions if a.source_instance_id == instance_id]

        if not relevant_actions:
            self.log_list.addItem(f"{tr('No actions for card')} {card_id} (Inst: {instance_id})")
            return

        if len(relevant_actions) == 1:
            self.execute_action(relevant_actions[0])
        else:
            self.log_list.addItem(tr("Multiple actions found. Executing first."))
            self.execute_action(relevant_actions[0])

    def on_card_hovered(self, card_id):
        if card_id >= 0:
            card_data = self.card_db.get(card_id)
            if card_data:
                self.card_detail_panel.update_card(card_data, self.civ_map)

    def execute_action(self, action):
        self.last_action = action
        dm_ai_module.EffectResolver.resolve_action(
            self.gs, action, self.card_db
        )
        self.log_list.addItem(f"P0 {tr('Action')}: {action.to_string()}")
        
        if self.gs.waiting_for_user_input:
            self.handle_user_input_request()
            return

        if action.type == dm_ai_module.ActionType.PASS or action.type == dm_ai_module.ActionType.MANA_CHARGE:
            dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
            
        self.update_ui()

    def handle_user_input_request(self):
        query = self.gs.pending_query

        if query.query_type == "SELECT_OPTION":
             options = query.options
             item, ok = QInputDialog.getItem(self, "Select Option", "Choose an option:", options, 0, False)
             if ok and item:
                 idx = options.index(item)
                 dm_ai_module.EffectResolver.resume(self.gs, self.card_db, idx)
                 self.step_phase()

        elif query.query_type == "SELECT_TARGET":
             self.log_list.addItem(f"Please select {query.params['min']} target(s).")
             self.update_ui()

    def step_phase(self):
        if self.is_processing: return
        self.is_processing = True
        was_running_at_start = self.is_running
        
        try:
            if self.gs.waiting_for_user_input:
                self.handle_user_input_request()
                return

            is_over, result = dm_ai_module.PhaseManager.check_game_over(self.gs)
            if is_over:
                self.timer.stop()
                self.is_running = False
                self.start_btn.setText(tr("Start Sim"))
                self.log_list.addItem(f"{tr('Game Over! Result')}: {result}")
                return

            active_pid = self.gs.active_player_id
            is_human = (active_pid == 0 and self.p0_human_radio.isChecked()) or \
                       (active_pid == 1 and self.p1_human_radio.isChecked())

            if is_human:
                actions = dm_ai_module.ActionGenerator.generate_legal_actions(
                    self.gs, self.card_db
                )
                if not actions:
                    dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
                    self.log_list.addItem(f"P{active_pid} {tr('Auto-Pass')}")
                    self.update_ui()
                return

            actions = dm_ai_module.ActionGenerator.generate_legal_actions(
                self.gs, self.card_db
            )

            if not actions:
                dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
                self.log_list.addItem(f"P{active_pid} {tr('Auto-Pass')}")
            else:
                best_action = actions[0] # Fallback
                
                if best_action:
                    self.last_action = best_action
                    dm_ai_module.EffectResolver.resolve_action(
                        self.gs, best_action, self.card_db
                    )
                    self.log_list.addItem(f"P{active_pid} {tr('AI Action')}: {best_action.to_string()}")

                    if self.gs.waiting_for_user_input:
                         self.log_list.addItem("AI Paused for Input (Not Implemented). Stopping Sim.")
                         self.timer.stop()
                         self.is_running = False
                         self.start_btn.setText(tr("Start Sim"))
                         return

                    if best_action.type == dm_ai_module.ActionType.PASS or best_action.type == dm_ai_module.ActionType.MANA_CHARGE:
                        dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)

            self.update_ui()
        finally:
            self.is_processing = False
        
    def update_ui(self):
        self.turn_label.setText(f"{tr('Turn')}: {self.gs.turn_number}")
        self.phase_label.setText(f"{tr('Phase')}: {self.gs.current_phase}")
        self.active_label.setText(f"{tr('Active')}: P{self.gs.active_player_id}")
        
        p0 = self.gs.players[0]
        p1 = self.gs.players[1]
        
        def convert_zone(zone_cards, hide=False):
            if hide:
                return [{'id': -1, 'tapped': c.is_tapped} for c in zone_cards]
            return [{'id': c.card_id, 'tapped': c.is_tapped} for c in zone_cards]
            
        god_view = self.god_view_check.isChecked()
        
        self.p0_hand.update_cards(convert_zone(p0.hand), self.card_db, self.civ_map)
        self.p0_mana.update_cards(convert_zone(p0.mana_zone), self.card_db, self.civ_map)
        self.p0_battle.update_cards(convert_zone(p0.battle_zone), self.card_db, self.civ_map)
        self.p0_shield.update_cards(convert_zone(p0.shield_zone), self.card_db, self.civ_map)
        self.p0_graveyard.update_cards(convert_zone(p0.graveyard), self.card_db, self.civ_map)
        self.p0_deck_zone.update_cards(convert_zone(p0.deck, hide=True), self.card_db, self.civ_map)
        
        self.p1_hand.update_cards(convert_zone(p1.hand, hide=not god_view), self.card_db, self.civ_map)
        self.p1_mana.update_cards(convert_zone(p1.mana_zone), self.card_db, self.civ_map)
        self.p1_battle.update_cards(convert_zone(p1.battle_zone), self.card_db, self.civ_map)
        self.p1_shield.update_cards(convert_zone(p1.shield_zone, hide=not god_view), self.card_db, self.civ_map)
        self.p1_graveyard.update_cards(convert_zone(p1.graveyard), self.card_db, self.civ_map)
        self.p1_deck_zone.update_cards(convert_zone(p1.deck, hide=True), self.card_db, self.civ_map)

        if self.gs.waiting_for_user_input and self.gs.pending_query.query_type == "SELECT_TARGET":
            valid_targets = self.gs.pending_query.valid_targets

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
