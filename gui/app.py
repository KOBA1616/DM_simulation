import sys
import os
import random
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox
)
import dm_ai_module
from gui.deck_builder import DeckBuilder


class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM AI Simulator")
        self.resize(1200, 800)
        
        # Game State
        self.gs = dm_ai_module.GameState(42)
        self.gs.setup_test_duel()
        dm_ai_module.PhaseManager.start_game(self.gs)
        self.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
        
        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left Panel (Game Info)
        self.info_panel = QWidget()
        self.info_layout = QVBoxLayout(self.info_panel)
        self.turn_label = QLabel("Turn: 1")
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
        
        self.main_layout.addWidget(self.info_panel)
        
        # Center Panel (Board)
        self.board_panel = QWidget()
        self.board_layout = QVBoxLayout(self.board_panel)
        
        self.p1_area = QLabel("P1 Area")
        self.p1_area.setStyleSheet("background-color: #ffcccc; padding: 10px;")
        self.board_layout.addWidget(self.p1_area)
        
        self.p0_area = QLabel("P0 Area")
        self.p0_area.setStyleSheet("background-color: #ccffcc; padding: 10px;")
        self.board_layout.addWidget(self.p0_area)
        
        self.main_layout.addWidget(self.board_panel)
        
        # Right Panel (Logs/Actions)
        self.log_list = QListWidget()
        self.main_layout.addWidget(self.log_list)
        
        self.update_ui()
        
    def open_deck_builder(self):
        self.deck_builder = DeckBuilder(self.card_db)
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

                # Set deck for P0
                self.gs.set_deck(0, deck_ids)
                
                # Restart game to apply changes (or just reset state)
                # For now, let's just reset the game with new deck
                # But we need P1 deck too. Let's assume P1 keeps default or random.
                # Actually, setup_test_duel sets both.
                # If we call set_deck, it updates the deck vector.
                # But we need to restart the game to draw hands etc.
                
                # Re-initialize game
                # self.gs = dm_ai_module.GameState(random.randint(0, 10000))
                # self.gs.setup_test_duel() # Sets default decks
                # self.gs.set_deck(0, deck_ids) # Override P0
                # dm_ai_module.PhaseManager.start_game(self.gs)
                
                # Wait, GameState is bound to C++.
                # If I create new GameState, I need to update self.gs
                
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
            # For now, just pick random action to step forward
            # In real GUI, we would let user click or AI choose
            action = random.choice(actions)
            dm_ai_module.EffectResolver.resolve_action(
                self.gs, action, self.card_db
            )
            self.log_list.addItem(f"Action: {action.to_string()}")

            if action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(self.gs)

        self.update_ui()
        
    def update_ui(self):
        self.turn_label.setText(f"Turn: {self.gs.turn_number}")
        self.phase_label.setText(f"Phase: {self.gs.current_phase}")
        self.active_label.setText(f"Active: P{self.gs.active_player_id}")
        
        # Update Board Text
        p0 = self.gs.players[0]
        p1 = self.gs.players[1]
        
        self.p0_area.setText(
            f"P0 (Active: {self.gs.active_player_id == 0})\n"
            f"Hand: {len(p0.hand)} | Mana: {len(p0.mana_zone)} | Shields: {len(p0.shield_zone)}\n"
            f"Battle: {len(p0.battle_zone)}"
        )
        
        self.p1_area.setText(
            f"P1 (Active: {self.gs.active_player_id == 1})\n"
            f"Hand: {len(p1.hand)} | Mana: {len(p1.mana_zone)} | Shields: {len(p1.shield_zone)}\n"
            f"Battle: {len(p1.battle_zone)}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec())
