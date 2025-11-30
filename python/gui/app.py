import sys
import os
import random
import json
import csv

# Add python/ and root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer
import dm_ai_module
from gui.deck_builder import DeckBuilder
from gui.card_editor import CardEditor
from gui.widgets.zone_widget import ZoneWidget
from gui.widgets.mcts_view import MCTSView
from gui.widgets.card_detail_panel import CardDetailPanel
# from gui.ai.mcts_python import PythonMCTS # Removed


class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM AI Simulator")
        self.resize(1400, 900)
        
        # Game State
        self.gs = dm_ai_module.GameState(42)
        self.gs.setup_test_duel()
        self.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
        dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        self.civ_map = self.load_civilizations("data/cards.csv")
        
        # Default Decks (from setup_test_duel logic)
        self.p0_deck_ids = [i for i in range(40)] # Placeholder, actual logic in C++ setup_test_duel is different
        self.p1_deck_ids = [i + 100 for i in range(40)] # Placeholder
        
        # We should probably capture the deck from the initial state if possible, 
        # but C++ setup_test_duel hardcodes it. 
        # Let's just rely on reset_game using these if they are set, 
        # or default setup_test_duel if not.
        self.p0_deck_ids = None
        self.p1_deck_ids = None

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_phase)
        self.is_running = False
        self.is_processing = False

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left Panel (Game Info & Controls)
        self.info_panel = QWidget()
        self.info_panel.setFixedWidth(280)
        self.info_layout = QVBoxLayout(self.info_panel)
        
        self.turn_label = QLabel("Turn: 1")
        self.turn_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.phase_label = QLabel("Phase: START")
        self.active_label = QLabel("Active: P0")
        self.info_layout.addWidget(self.turn_label)
        self.info_layout.addWidget(self.phase_label)
        self.info_layout.addWidget(self.active_label)
        
        # Card Detail Panel
        self.card_detail_panel = CardDetailPanel()
        self.info_layout.addWidget(self.card_detail_panel)
        
        # Controls Group
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self.toggle_simulation)
        ctrl_layout.addWidget(self.start_btn)
        
        self.step_button = QPushButton("Step Phase")
        self.step_button.clicked.connect(self.step_phase)
        ctrl_layout.addWidget(self.step_button)
        
        self.reset_btn = QPushButton("Reset Game")
        self.reset_btn.clicked.connect(self.reset_game)
        ctrl_layout.addWidget(self.reset_btn)
        
        ctrl_group.setLayout(ctrl_layout)
        self.info_layout.addWidget(ctrl_group)
        
        # Player Mode Group
        mode_group = QGroupBox("Player Modes")
        mode_layout = QVBoxLayout()
        
        self.p0_human_radio = QRadioButton("P0: Human")
        self.p0_ai_radio = QRadioButton("P0: AI")
        self.p0_ai_radio.setChecked(True)
        self.p0_group = QButtonGroup()
        self.p0_group.addButton(self.p0_human_radio)
        self.p0_group.addButton(self.p0_ai_radio)
        
        mode_layout.addWidget(self.p0_human_radio)
        mode_layout.addWidget(self.p0_ai_radio)
        
        self.p1_human_radio = QRadioButton("P1: Human")
        self.p1_ai_radio = QRadioButton("P1: AI")
        self.p1_ai_radio.setChecked(True)
        self.p1_group = QButtonGroup()
        self.p1_group.addButton(self.p1_human_radio)
        self.p1_group.addButton(self.p1_ai_radio)
        
        mode_layout.addWidget(self.p1_human_radio)
        mode_layout.addWidget(self.p1_ai_radio)
        
        mode_group.setLayout(mode_layout)
        self.info_layout.addWidget(mode_group)
        
        self.deck_builder_button = QPushButton("Deck Builder")
        self.deck_builder_button.clicked.connect(self.open_deck_builder)
        self.info_layout.addWidget(self.deck_builder_button)

        self.card_editor_button = QPushButton("Card Editor")
        self.card_editor_button.clicked.connect(self.open_card_editor)
        self.info_layout.addWidget(self.card_editor_button)

        # Deck Loading Controls
        deck_group = QGroupBox("Deck Management")
        deck_layout = QVBoxLayout()

        self.load_deck_btn = QPushButton("Load Deck P0 (Player)")
        self.load_deck_btn.clicked.connect(self.load_deck_p0)
        deck_layout.addWidget(self.load_deck_btn)

        self.load_deck_p1_btn = QPushButton("Load Deck P1 (Opponent)")
        self.load_deck_p1_btn.clicked.connect(self.load_deck_p1)
        deck_layout.addWidget(self.load_deck_p1_btn)

        deck_group.setLayout(deck_layout)
        self.info_layout.addWidget(deck_group)
        
        self.god_view_check = QCheckBox("God View (Show Opponent Hand)")
        self.god_view_check.setChecked(False)
        self.god_view_check.stateChanged.connect(self.update_ui)
        self.info_layout.addWidget(self.god_view_check)

        # Help Button
        self.help_btn = QPushButton("Help / Instructions")
        self.help_btn.setStyleSheet("background-color: #e1f5fe; color: #0277bd; font-weight: bold;")
        self.help_btn.clicked.connect(self.show_help)
        self.info_layout.addWidget(self.help_btn)
        
        self.info_layout.addStretch()
        
        # MCTS View
        self.mcts_view = MCTSView()
        self.info_layout.addWidget(self.mcts_view)
        
        self.main_layout.addWidget(self.info_panel)
        
        # Center Panel (Board)
        self.board_panel = QWidget()
        self.board_layout = QVBoxLayout(self.board_panel)
        
        # P1 (Opponent) Zones
        self.p1_zones = QWidget()
        self.p1_layout = QVBoxLayout(self.p1_zones)
        self.p1_hand = ZoneWidget("P1 Hand")
        self.p1_mana = ZoneWidget("P1 Mana")
        self.p1_battle = ZoneWidget("P1 Battle")
        self.p1_shield = ZoneWidget("P1 Shield")
        
        self.p1_layout.addWidget(self.p1_hand)
        self.p1_layout.addWidget(self.p1_mana)
        self.p1_layout.addWidget(self.p1_battle)
        self.p1_layout.addWidget(self.p1_shield)
        
        # P0 (Player) Zones
        self.p0_zones = QWidget()
        self.p0_layout = QVBoxLayout(self.p0_zones)
        self.p0_battle = ZoneWidget("P0 Battle")
        self.p0_shield = ZoneWidget("P0 Shield")
        self.p0_mana = ZoneWidget("P0 Mana")
        self.p0_hand = ZoneWidget("P0 Hand")
        
        # Connect signals
        self.p0_hand.card_clicked.connect(self.on_card_clicked)
        self.p0_mana.card_clicked.connect(self.on_card_clicked)
        self.p0_battle.card_clicked.connect(self.on_card_clicked)
        
        self.p0_hand.card_hovered.connect(self.on_card_hovered)
        self.p0_mana.card_hovered.connect(self.on_card_hovered)
        self.p0_battle.card_hovered.connect(self.on_card_hovered)
        self.p0_shield.card_hovered.connect(self.on_card_hovered)
        
        self.p1_hand.card_hovered.connect(self.on_card_hovered)
        self.p1_mana.card_hovered.connect(self.on_card_hovered)
        self.p1_battle.card_hovered.connect(self.on_card_hovered)
        self.p1_shield.card_hovered.connect(self.on_card_hovered)
        
        self.p0_layout.addWidget(self.p0_battle)
        self.p0_layout.addWidget(self.p0_shield)
        self.p0_layout.addWidget(self.p0_mana)
        self.p0_layout.addWidget(self.p0_hand)
        
        # Splitter for P1 and P0 areas
        self.board_splitter = QSplitter(Qt.Orientation.Vertical)
        self.board_splitter.addWidget(self.p1_zones)
        self.board_splitter.addWidget(self.p0_zones)
        self.board_layout.addWidget(self.board_splitter)
        
        self.main_layout.addWidget(self.board_panel, stretch=2)
        
        # Right Panel (Logs)
        self.log_list = QListWidget()
        self.log_list.setFixedWidth(250)
        self.main_layout.addWidget(self.log_list)
        
        self.update_ui()
        
    def load_civilizations(self, filepath):
        civ_map = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cid = int(row['ID'])
                        civ = row['Civilization']
                        civ_map[cid] = civ
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Error loading civilizations: {e}")
        return civ_map

    def open_deck_builder(self):
        self.deck_builder = DeckBuilder(self.card_db, self.civ_map)
        self.deck_builder.show()

    def open_card_editor(self):
        self.card_editor = CardEditor("data/cards.csv")
        self.card_editor.show()

    def load_deck_p0(self):
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Deck P0", "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    deck_ids = json.load(f)
                
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, "Invalid Deck", "Deck must have 40 cards.")
                    return

                self.p0_deck_ids = deck_ids
                self.reset_game()
                
                self.log_list.addItem(f"Loaded Deck for P0: {os.path.basename(fname)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load deck: {e}")

    def load_deck_p1(self):
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Deck P1", "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    deck_ids = json.load(f)
                
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, "Invalid Deck", "Deck must have 40 cards.")
                    return

                self.p1_deck_ids = deck_ids
                self.reset_game()
                
                self.log_list.addItem(f"Loaded Deck for P1: {os.path.basename(fname)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load deck: {e}")

    def show_help(self):
        help_text = """
        <h2>DM AI Simulator Guide</h2>
        <p><b>Basic Controls:</b></p>
        <ul>
            <li><b>Start Simulation:</b> Starts the AI vs AI or AI vs Human game loop.</li>
            <li><b>Step Phase:</b> Advances the game by one phase or action manually.</li>
            <li><b>Reset Game:</b> Restarts the game with current decks.</li>
        </ul>
        <p><b>Player Modes:</b></p>
        <ul>
            <li><b>Human:</b> You control the actions. Click cards to play/use them.</li>
            <li><b>AI:</b> The AI plays automatically using MCTS.</li>
        </ul>
        <p><b>Deck Management:</b></p>
        <ul>
            <li><b>Deck Builder:</b> Create and save custom decks.</li>
            <li><b>Load Deck P0/P1:</b> Load a saved JSON deck for Player (P0) or Opponent (P1).</li>
        </ul>
        <p><b>Views:</b></p>
        <ul>
            <li><b>God View:</b> Check this to see the opponent's hand and shields.</li>
            <li><b>MCTS View:</b> Shows the AI's thought process (search tree).</li>
        </ul>
        """
        QMessageBox.information(self, "Help / Instructions", help_text)
        
    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText("Start Simulation")
            self.is_running = False
        else:
            self.timer.start(500) # 500ms per step
            self.start_btn.setText("Stop Simulation")
            self.is_running = True

    def reset_game(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("Start Simulation")
        
        self.gs = dm_ai_module.GameState(random.randint(0, 10000))
        self.gs.setup_test_duel()
        
        # Apply custom decks if loaded
        if self.p0_deck_ids:
            self.gs.set_deck(0, self.p0_deck_ids)
        if self.p1_deck_ids:
            self.gs.set_deck(1, self.p1_deck_ids)
            
        dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        
        self.log_list.clear()
        self.log_list.addItem("Game Reset")
        self.update_ui()

    def on_card_clicked(self, card_id, instance_id):
        # Only handle clicks if it's P0's turn and P0 is Human
        if self.gs.active_player_id != 0 or not self.p0_human_radio.isChecked():
            return

        # Get legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(
            self.gs, self.card_db
        )
        
        # Filter actions for this card
        # Note: instance_id might be -1 if unknown, but here we expect valid instance_id
        relevant_actions = [
            a for a in actions 
            if a.source_instance_id == instance_id or (a.card_id == card_id and a.type == dm_ai_module.ActionType.PLAY_CARD)
        ]
        # Note: PLAY_CARD actions usually have source_instance_id set correctly by generator.
        # Let's rely on source_instance_id.
        relevant_actions = [a for a in actions if a.source_instance_id == instance_id]

        if not relevant_actions:
            self.log_list.addItem(f"No actions for card {card_id} (Inst: {instance_id})")
            return

        if len(relevant_actions) == 1:
            action = relevant_actions[0]
            self.execute_action(action)
        else:
            # Multiple actions (e.g. Attack Player vs Creature)
            # For now, just pick first or show dialog.
            # Let's pick Attack Player if available, else random.
            # Or simple dialog?
            # Let's just log and pick first for MVP.
            self.log_list.addItem(f"Multiple actions found. Executing first.")
            self.execute_action(relevant_actions[0])

    def on_card_hovered(self, card_id):
        if card_id == -1:
            # Hidden card or mouse left
            pass
        
        if card_id >= 0:
            card_data = self.card_db.get(card_id)
            if card_data:
                self.card_detail_panel.update_card(card_data, self.civ_map)
        else:
            # Maybe clear?
            pass

    def execute_action(self, action):
        dm_ai_module.EffectResolver.resolve_action(
            self.gs, action, self.card_db
        )
        self.log_list.addItem(f"P0 Action: {action.to_string()}")
        
        if action.type == dm_ai_module.ActionType.PASS or action.type == dm_ai_module.ActionType.MANA_CHARGE:
            dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
            
        self.update_ui()

    def step_phase(self):
        if self.is_processing:
            return
        self.is_processing = True
        was_running_at_start = self.is_running
        
        try:
            # Check Game Over
            is_over, result = dm_ai_module.PhaseManager.check_game_over(self.gs)
            if is_over:
                self.timer.stop()
                self.is_running = False
                self.start_btn.setText("Start Simulation")
                self.log_list.addItem(f"Game Over! Result: {result}")
                return

            active_pid = self.gs.active_player_id
            is_human = (active_pid == 0 and self.p0_human_radio.isChecked()) or \
                       (active_pid == 1 and self.p1_human_radio.isChecked())

            if is_human:
                # If human turn, we wait for input.
                # Unless there are no actions (Auto-Pass)
                actions = dm_ai_module.ActionGenerator.generate_legal_actions(
                    self.gs, self.card_db
                )
                if not actions:
                    dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
                    self.log_list.addItem(f"P{active_pid} Auto-Pass")
                    self.update_ui()
                return

            # AI Turn
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(
                self.gs, self.card_db
            )

            if not actions:
                dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
                self.log_list.addItem(f"P{active_pid} Auto-Pass")
            else:
                # Determinize for AI
                # Clone state
                search_state = self.gs.clone()
                # Shuffle opponent's hidden zones
                dm_ai_module.Determinizer.determinize(search_state, active_pid)

                # Use C++ MCTS
                mcts = dm_ai_module.MCTS(self.card_db, 1.0, 0.3, 0.25, 1) # Batch size 1 for GUI (simple)
                
                # Use C++ Heuristic Evaluator
                evaluator = dm_ai_module.HeuristicEvaluator(self.card_db)

                # Search using C++ evaluator (no Python callback overhead)
                policy = mcts.search_with_heuristic(search_state, 50, evaluator, True, 1.0)
                
                # Select Action (Argmax of policy)
                best_idx = -1
                best_prob = -1.0
                for i, p in enumerate(policy):
                    if p > best_prob:
                        best_prob = p
                        best_idx = i
                
                best_action = None
                if best_idx >= 0:
                    # Find action object matching index
                    # We need to regenerate actions to find the object
                    # Or we can use the tree root children
                    root = mcts.get_last_root()
                    if root:
                        # Find child with highest visits (usually matches policy if temp -> 0, but here temp=1.0)
                        # Actually policy is returned based on visits.
                        # So we can just pick action from legal actions that matches index?
                        # But multiple actions might map to same index? (Hopefully not)
                        # Better to use tree children.
                        best_child = None
                        max_visits = -1
                        for child in root.children:
                            if child.visit_count > max_visits:
                                max_visits = child.visit_count
                                best_child = child
                        
                        if best_child:
                            best_action = best_child.action
                
                # Update MCTS View
                def convert_tree_data(node):
                    if not node: return None
                    
                    # Determine name
                    name = node.action.to_string()
                    if node.action.type == dm_ai_module.ActionType.PASS:
                        name = "PASS"
                    elif not name: # Fallback
                        name = "Unknown"

                    data = {
                        "name": name,
                        "visits": node.visit_count,
                        "value": node.value,
                        "children": []
                    }

                    # Limit depth for view?
                    # MCTSView handles recursion.
                    # But C++ tree might be deep.
                    # Let's just do 2 levels.
                    if node.visit_count > 1: # Only expand visited nodes
                         for child in node.children:
                             data["children"].append(convert_tree_data(child))
                    return data

                root = mcts.get_last_root()
                if root:
                    # Root action is dummy.
                    tree_data = {
                        "name": "Root",
                        "visits": root.visit_count,
                        "value": root.value,
                        "children": []
                    }
                    for child in root.children:
                        tree_data["children"].append(convert_tree_data(child))
                    
                    self.mcts_view.update_from_data(tree_data)
                
                if was_running_at_start and not self.is_running:
                    self.log_list.addItem("Simulation stopped.")
                    return

                if best_action:
                    dm_ai_module.EffectResolver.resolve_action(
                        self.gs, best_action, self.card_db
                    )
                    self.log_list.addItem(f"P{active_pid} AI Action: {best_action.to_string()}")

                    if best_action.type == dm_ai_module.ActionType.PASS or best_action.type == dm_ai_module.ActionType.MANA_CHARGE:
                        dm_ai_module.PhaseManager.next_phase(self.gs, self.card_db)
                else:
                    self.log_list.addItem("Error: MCTS returned None")

            self.update_ui()
        finally:
            self.is_processing = False
        
    def update_ui(self):
        self.turn_label.setText(f"Turn: {self.gs.turn_number}")
        self.phase_label.setText(f"Phase: {self.gs.current_phase}")
        self.active_label.setText(f"Active: P{self.gs.active_player_id}")
        
        # Update Zones
        p0 = self.gs.players[0]
        p1 = self.gs.players[1]
        
        # Helper to convert C++ vector to list of dicts
        def convert_zone(zone_cards, hide=False):
            if hide:
                return [{'id': -1, 'tapped': c.is_tapped} for c in zone_cards] # -1 for hidden
            return [{'id': c.card_id, 'tapped': c.is_tapped} for c in zone_cards]
            
        # P0 is Human (usually), P1 is Opponent
        # If God View is OFF, hide P1 Hand and Shield
        god_view = self.god_view_check.isChecked()
        
        self.p0_hand.update_cards(convert_zone(p0.hand), self.card_db, self.civ_map)
        self.p0_mana.update_cards(convert_zone(p0.mana_zone), self.card_db, self.civ_map)
        self.p0_battle.update_cards(convert_zone(p0.battle_zone), self.card_db, self.civ_map)
        self.p0_shield.update_cards(convert_zone(p0.shield_zone), self.card_db, self.civ_map)
        
        self.p1_hand.update_cards(convert_zone(p1.hand, hide=not god_view), self.card_db, self.civ_map)
        self.p1_mana.update_cards(convert_zone(p1.mana_zone), self.card_db, self.civ_map)
        self.p1_battle.update_cards(convert_zone(p1.battle_zone), self.card_db, self.civ_map)
        self.p1_shield.update_cards(convert_zone(p1.shield_zone, hide=not god_view), self.card_db, self.civ_map)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec())
