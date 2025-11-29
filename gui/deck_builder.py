import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QPushButton, QLabel, QLineEdit, QMessageBox, QFileDialog
)
import dm_ai_module
from gui.card_editor import CardEditor


class DeckBuilder(QWidget):
    def __init__(self, card_db):
        super().__init__()
        self.card_db = card_db
        self.current_deck = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Deck Builder")
        self.resize(800, 600)
        layout = QHBoxLayout(self)

        # Left: Card Database
        left_panel = QVBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.filter_cards)
        left_panel.addWidget(self.search_bar)

        self.card_list = QListWidget()
        self.card_list.itemDoubleClicked.connect(self.add_card)
        left_panel.addWidget(self.card_list)
        
        # Add New Card Button
        new_card_btn = QPushButton("New Card")
        new_card_btn.clicked.connect(self.open_card_editor)
        left_panel.addWidget(new_card_btn)

        layout.addLayout(left_panel)

        # Right: Current Deck
        right_panel = QVBoxLayout()
        self.deck_count_label = QLabel("Deck: 0/40")
        right_panel.addWidget(self.deck_count_label)

        self.deck_list = QListWidget()
        self.deck_list.itemDoubleClicked.connect(self.remove_card)
        right_panel.addWidget(self.deck_list)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Deck")
        save_btn.clicked.connect(self.save_deck)
        load_btn = QPushButton("Load Deck")
        load_btn.clicked.connect(self.load_deck)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(load_btn)

        right_panel.addLayout(btn_layout)
        layout.addLayout(right_panel)

        self.populate_card_list()

    def open_card_editor(self):
        editor = CardEditor("data/cards.csv", self)
        if editor.exec():
            # Reload DB
            self.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
            self.populate_card_list()

    def populate_card_list(self):
        self.card_list.clear()
        for cid, card in self.card_db.items():
            self.card_list.addItem(f"[{cid}] {card.name} (Cost: {card.cost})")

    def filter_cards(self, text):
        # Simple filter
        self.card_list.clear()
        for cid, card in self.card_db.items():
            if text.lower() in card.name.lower():
                self.card_list.addItem(
                    f"[{cid}] {card.name} (Cost: {card.cost})"
                )

    def add_card(self, item):
        if len(self.current_deck) >= 40:
            return

        text = item.text()
        try:
            cid = int(text.split(']')[0][1:])
            self.current_deck.append(cid)
            self.update_deck_list()
        except (ValueError, IndexError):
            pass

    def remove_card(self, item):
        row = self.deck_list.row(item)
        if row >= 0:
            self.current_deck.pop(row)
            self.update_deck_list()

    def update_deck_list(self):
        self.deck_list.clear()
        self.current_deck.sort()
        for cid in self.current_deck:
            if cid in self.card_db:
                card = self.card_db[cid]
                self.deck_list.addItem(f"[{cid}] {card.name}")
            else:
                self.deck_list.addItem(f"[{cid}] Unknown")
        self.deck_count_label.setText(f"Deck: {len(self.current_deck)}/40")

    def save_deck(self):
        if len(self.current_deck) != 40:
            QMessageBox.warning(
                self, "Invalid Deck", "Deck must have exactly 40 cards."
            )
            return

        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Deck", "data/decks", "JSON Files (*.json)"
        )
        if fname:
            with open(fname, 'w') as f:
                json.dump(self.current_deck, f)
            QMessageBox.information(self, "Success", "Deck saved!")

    def load_deck(self):
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Deck", "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    self.current_deck = json.load(f)
                self.update_deck_list()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load deck: {e}"
                )
