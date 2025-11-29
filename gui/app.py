import sys
import os
import random
import json
import csv

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt
import dm_ai_module
from gui.deck_builder import DeckBuilder
from gui.widgets.zone_widget import ZoneWidget
from gui.widgets.mcts_view import MCTSView
from gui.ai.mcts_python import PythonMCTS


class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM AI Simulator")
        self.resize(1400, 900)
        
        # Game State
        self.gs = dm_ai_module.GameState(42)
        self.gs.setup_test_duel()
        dm_ai_module.PhaseManager.start_game(self.gs)
        self.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
        self.civ_map = self.load_civilizations("data/cards.csv")
        
        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left Panel (Game Info & Controls)
        self.info_panel = QWidget()
        self.info_panel.setFixedWidth(250)
        self.info_layout = QVBoxLayout(self.info_panel)
        
        self.turn_label = QLabel("Turn: 1")
        self.turn_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.phase_label = QLabel("Phase: START")
        self.active_label = QLabel("Active: P0")
        self.info_layout.addWidget(self.turn_label)
        self.info_layout.addWidget(self.phase_label)
        self.info_layout.addWidget(self.active_label)
        
        self.step_button = QPushButton("Step Phase")
        self.step_button.clicked.connect(self.step_phase)
        self.info_layout.addWidget(self.step_button)
        
        self.deck_builder_button = QPushButton("Deck Builder")
        self.deck_builder_button.clicked.connect(self.open_deck_builder)
        self.info_layout.addWidget(self.deck_builder_button)

        self.load_deck_btn = QPushButton("Load Deck P0")
        self.load_deck_btn.clicked.connect(self.load_deck_p0)
        self.info_layout.addWidget(self.load_deck_btn)
        
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

                new_gs = dm_ai_module.GameState(random.randint(0, 10000))
                new_gs.setup_test_duel()
                new_gs.set_deck(0, deck_ids)
                dm_ai_module.PhaseManager.start_game(new_gs)
                self.gs = new_gs
                
                self.log_list.clear()
                self.log_list.addItem(f"Loaded Deck for P0: {os.path.basename(fname)}")
                self.update_ui()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load deck: {e}")
        
    def step_phase(self):
        # Check Game Over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(self.gs)
        if is_over:
            self.log_list.addItem(f"Game Over! Result: {result}")
            return

        # Generate Actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(
            self.gs, self.card_db
        )

        if not actions:
            dm_ai_module.PhaseManager.next_phase(self.gs)
            self.log_list.addItem("Auto-Pass Phase")
        else:
            # Use MCTS to decide action
            mcts = PythonMCTS(self.card_db, simulations=50) # 50 simulations for responsiveness
            best_action = mcts.search(self.gs)
            
            # Update MCTS View
            tree_data = mcts.get_tree_data()
            self.mcts_view.update_from_data(tree_data)
            
            if best_action:
                dm_ai_module.EffectResolver.resolve_action(
                    self.gs, best_action, self.card_db
                )
                self.log_list.addItem(f"Action: {best_action.to_string()}")

                if best_action.type == dm_ai_module.ActionType.PASS:
                    dm_ai_module.PhaseManager.next_phase(self.gs)
            else:
                # Should not happen if actions is not empty
                self.log_list.addItem("Error: MCTS returned None")

        self.update_ui()
        
    def update_ui(self):
        self.turn_label.setText(f"Turn: {self.gs.turn_number}")
        self.phase_label.setText(f"Phase: {self.gs.current_phase}")
        self.active_label.setText(f"Active: P{self.gs.active_player_id}")
        
        # Update Zones
        p0 = self.gs.players[0]
        p1 = self.gs.players[1]
        
        # Helper to convert C++ vector to list of dicts
        def convert_zone(zone_cards):
            return [{'id': c.card_id, 'tapped': c.is_tapped} for c in zone_cards]
            
        self.p0_hand.update_cards(convert_zone(p0.hand), self.card_db, self.civ_map)
        self.p0_mana.update_cards(convert_zone(p0.mana_zone), self.card_db, self.civ_map)
        self.p0_battle.update_cards(convert_zone(p0.battle_zone), self.card_db, self.civ_map)
        self.p0_shield.update_cards(convert_zone(p0.shield_zone), self.card_db, self.civ_map)
        
        self.p1_hand.update_cards(convert_zone(p1.hand), self.card_db, self.civ_map)
        self.p1_mana.update_cards(convert_zone(p1.mana_zone), self.card_db, self.civ_map)
        self.p1_battle.update_cards(convert_zone(p1.battle_zone), self.card_db, self.civ_map)
        self.p1_shield.update_cards(convert_zone(p1.shield_zone), self.card_db, self.civ_map)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec())
