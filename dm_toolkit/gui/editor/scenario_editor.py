# -*- coding: cp932 -*-
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QSplitter, QWidget,
                             QTabWidget, QFormLayout, QSpinBox, QCheckBox)
from PyQt6.QtCore import Qt
import json
import os
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.parts.card_search import CardSearchWidget
from dm_toolkit.gui.editor.forms.parts.draggable_list import DraggableListWidget

class ScenarioEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scenario Editor"))
        self.resize(1200, 700)
        self.scenarios = []
        self.current_index = -1
        self.is_updating_ui = False
        self.all_cards = []
        self.setup_ui()
        self.load_data()
        self.load_card_db()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left: Scenario List
        left_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_selection_changed)
        left_layout.addWidget(QLabel(tr("Scenarios")))
        left_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton(tr("New"))
        self.btn_new.clicked.connect(self.on_new)
        self.btn_delete = QPushButton(tr("Delete"))
        self.btn_delete.clicked.connect(self.on_delete)
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_delete)
        left_layout.addLayout(btn_layout)

        # Center: Editor
        center_layout = QVBoxLayout()

        # Header Info
        header_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self.save_header_to_memory)
        header_layout.addRow(tr("Name (ID):"), self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.editingFinished.connect(self.save_header_to_memory)
        header_layout.addRow(tr("Description:"), self.desc_edit)
        center_layout.addLayout(header_layout)

        # Tabs for Config
        self.tabs = QTabWidget()
        center_layout.addWidget(self.tabs)

        # Tab 1: General Settings
        self.tab_general = QWidget()
        self.form_general = QFormLayout()
        self.spin_my_mana = QSpinBox()
        self.spin_my_mana.setRange(0, 99)
        self.form_general.addRow(tr("My Mana (Available):"), self.spin_my_mana)

        self.spin_enemy_shields = QSpinBox()
        self.spin_enemy_shields.setRange(0, 99)
        self.form_general.addRow(tr("Enemy Shields (Count):"), self.spin_enemy_shields)

        self.check_enemy_trigger = QCheckBox(tr("Enemy Can Use Trigger"))
        self.form_general.addRow("", self.check_enemy_trigger)

        self.check_loop_proof = QCheckBox(tr("Loop Proof Mode"))
        self.form_general.addRow("", self.check_loop_proof)

        self.tab_general.setLayout(self.form_general)
        self.tabs.addTab(self.tab_general, tr("General"))

        # Zone Tabs
        self.zone_lists = {}
        self.create_zone_tab("my_hand_cards", tr("My Hand"))
        self.create_zone_tab("my_battle_zone", tr("My Battle Zone"))
        self.create_zone_tab("my_mana_zone", tr("My Mana Zone"))
        self.create_zone_tab("my_grave_yard", tr("My Graveyard"))
        self.create_zone_tab("my_shields", tr("My Shields"))
        self.create_zone_tab("my_deck", tr("My Deck"))
        self.create_zone_tab("enemy_battle_zone", tr("Enemy Battle Zone"))
        self.create_zone_tab("enemy_deck", tr("Enemy Deck"))

        # Connect signals
        self.spin_my_mana.valueChanged.connect(self.save_config_to_memory)
        self.spin_enemy_shields.valueChanged.connect(self.save_config_to_memory)
        self.check_enemy_trigger.stateChanged.connect(self.save_config_to_memory)
        self.check_loop_proof.stateChanged.connect(self.save_config_to_memory)

        self.btn_save = QPushButton(tr("Save All to File"))
        self.btn_save.clicked.connect(self.save_to_file)
        center_layout.addWidget(self.btn_save)

        # Right: Card Search
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel(tr("Card Search")))
        self.card_search = CardSearchWidget()
        right_layout.addWidget(self.card_search)

        # Splitter Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        layout.addWidget(splitter)

        self.enable_inputs(False)

    def create_zone_tab(self, key, title):
        tab = QWidget()
        vbox = QVBoxLayout()

        # Tools Layout (Load Deck, etc.)
        tools_layout = QHBoxLayout()
        lbl = QLabel(tr("Drag cards here or press Delete to remove:"))
        tools_layout.addWidget(lbl)

        # Add 'Load Deck' button only for deck tabs
        if "deck" in key:
            btn_load_deck = QPushButton(tr("Load Deck JSON"))
            btn_load_deck.clicked.connect(lambda: self.load_deck_into_zone(key))
            tools_layout.addWidget(btn_load_deck)

            # Add 'Move to Zone' button
            btn_move = QPushButton(tr("Search & Move to Zone..."))
            btn_move.clicked.connect(lambda: self.search_and_move_from_deck(key))
            tools_layout.addWidget(btn_move)

        vbox.addLayout(tools_layout)

        # Use DraggableListWidget instead of QTextEdit
        list_widget = DraggableListWidget()
        # Connect internal model change isn't trivial for 'items changed',
        # so we rely on save/load flow or could connect model signal.
        # simpler: update memory when switching tabs or saving.

        # Shortcut for delete
        # Note: QShortcut requires window context, simplified via keyPressEvent override in widget or manual check
        list_widget.keyPressEvent = lambda event: self.handle_list_key_press(event, list_widget)

        vbox.addWidget(list_widget)
        tab.setLayout(vbox)
        self.tabs.addTab(tab, title)
        self.zone_lists[key] = list_widget

    def load_deck_into_zone(self, key):
        list_widget = self.zone_lists.get(key)
        if not list_widget: return

        fname, _ = QFileDialog.getOpenFileName(
            self, tr("Load Deck"), "data/decks", "JSON Files (*.json)"
        )
        if fname:
            try:
                with open(fname, 'r') as f:
                    deck_ids = json.load(f)

                list_widget.clear()
                for cid in deck_ids:
                    list_widget.addItem(str(cid))

                QMessageBox.information(self, tr("Success"), tr(f"Loaded {len(deck_ids)} cards."))
                self.save_config_to_memory()
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"Failed to load deck: {e}")

    def search_and_move_from_deck(self, deck_key):
        deck_list = self.zone_lists.get(deck_key)
        if not deck_list or deck_list.count() == 0:
            QMessageBox.warning(self, tr("Warning"), tr("Deck is empty."))
            return

        # Create a dialog to select a card from the deck
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Select Card to Move"))
        layout = QVBoxLayout(dlg)

        card_list = QListWidget()
        # Populate with cards currently in deck list
        temp_deck_items = []
        for i in range(deck_list.count()):
            txt = deck_list.item(i).text()
            temp_deck_items.append(txt) # Keep as string ID
            card_list.addItem(txt)

        layout.addWidget(QLabel(tr("Select a card to move from Deck:")))
        layout.addWidget(card_list)

        # Target Zone Selection
        layout.addWidget(QLabel(tr("To Zone:")))
        zone_combo = QTabWidget() # Use tab-like selection or combo? simpler: Combo
        from PyQt6.QtWidgets import QComboBox
        combo = QComboBox()

        # Determine target zones based on player (my/enemy)
        prefix = "my_" if "my_" in deck_key else "enemy_"
        possible_zones = [k for k in self.zone_lists.keys() if k.startswith(prefix) and "deck" not in k]

        zone_map = {}
        for k in possible_zones:
            name = k.replace(prefix, "").replace("_", " ").title()
            combo.addItem(name, k)

        layout.addWidget(combo)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton(tr("Move"))
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn = QPushButton(tr("Cancel"))
        cancel_btn.clicked.connect(dlg.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        if dlg.exec():
            # Perform Move
            selected_items = card_list.selectedItems()
            if not selected_items: return

            selected_text = selected_items[0].text()
            target_zone_key = combo.currentData()

            # Remove from deck list (find first occurrence)
            for i in range(deck_list.count()):
                if deck_list.item(i).text() == selected_text:
                    deck_list.takeItem(i)
                    break

            # Add to target zone
            target_list = self.zone_lists.get(target_zone_key)
            if target_list:
                target_list.addItem(selected_text)

            self.save_config_to_memory()

    def handle_list_key_press(self, event, list_widget):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            for item in list_widget.selectedItems():
                list_widget.takeItem(list_widget.row(item))
        else:
            QListWidget.keyPressEvent(list_widget, event)

    def load_card_db(self):
        path = "data/cards.json"
        if not os.path.exists(path) and os.path.exists("../data/cards.json"):
            path = "../data/cards.json"

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.all_cards = json.load(f)
                self.card_search.load_cards(self.all_cards)
            except Exception as e:
                print(f"Failed to load cards: {e}")

    def load_data(self):
        path = "data/scenarios.json"
        if not os.path.exists(path) and os.path.exists("../data/scenarios.json"):
            path = "../data/scenarios.json"

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.scenarios = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, tr("Error"), f"Failed to load scenarios: {e}")
                self.scenarios = []

        self.refresh_list()

    def refresh_list(self):
        self.list_widget.clear()
        for item in self.scenarios:
            self.list_widget.addItem(item.get("name", "Unnamed"))

    def on_selection_changed(self, row):
        if self.current_index >= 0 and self.current_index < len(self.scenarios):
             self.save_all_to_memory(self.current_index)

        self.current_index = row
        if row >= 0:
            item = self.scenarios[row]
            self.is_updating_ui = True

            self.name_edit.setText(item.get("name", ""))
            self.desc_edit.setText(item.get("description", ""))

            config = item.get("config", {})
            self.spin_my_mana.setValue(config.get("my_mana", 0))
            self.spin_enemy_shields.setValue(config.get("enemy_shield_count", 5))
            self.check_enemy_trigger.setChecked(config.get("enemy_can_use_trigger", False))
            self.check_loop_proof.setChecked(config.get("loop_proof_mode", False))

            for key, lst in self.zone_lists.items():
                lst.clear()
                val = config.get(key, [])
                if isinstance(val, list):
                    for x in val:
                        lst.addItem(str(x))

            self.is_updating_ui = False
            self.enable_inputs(True)
        else:
            self.clear_inputs()
            self.enable_inputs(False)

    def save_header_to_memory(self):
        if self.current_index < 0 or self.is_updating_ui: return
        item = self.scenarios[self.current_index]
        item["name"] = self.name_edit.text()
        item["description"] = self.desc_edit.text()
        self.list_widget.item(self.current_index).setText(item["name"])

    def save_config_to_memory(self):
        if self.current_index < 0 or self.is_updating_ui: return
        self.save_all_to_memory(self.current_index)

    def save_all_to_memory(self, index):
        if index < 0 or index >= len(self.scenarios): return

        item = self.scenarios[index]
        item["name"] = self.name_edit.text()
        item["description"] = self.desc_edit.text()

        config = item.get("config", {})
        config["my_mana"] = self.spin_my_mana.value()
        config["enemy_shield_count"] = self.spin_enemy_shields.value()
        config["enemy_can_use_trigger"] = self.check_enemy_trigger.isChecked()
        config["loop_proof_mode"] = self.check_loop_proof.isChecked()

        # Parse Zone Lists
        for key, lst in self.zone_lists.items():
            parsed_list = []
            for i in range(lst.count()):
                text = lst.item(i).text()
                # Try to parse as int (ID), else keep as string (Name)
                try:
                    parsed_list.append(int(text))
                except ValueError:
                    parsed_list.append(text)
            config[key] = parsed_list

        item["config"] = config

    def on_new(self):
        new_item = {
            "name": "new_scenario",
            "description": "",
            "config": {
                "my_mana": 0,
                "my_hand_cards": [],
                "my_battle_zone": [],
                "my_mana_zone": [],
                "my_shields": [],
                "enemy_shield_count": 5
            }
        }
        self.scenarios.append(new_item)
        self.refresh_list()
        self.list_widget.setCurrentRow(len(self.scenarios) - 1)

    def on_delete(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            del self.scenarios[row]
            self.current_index = -1
            self.refresh_list()
            self.clear_inputs()

    def save_to_file(self):
        if self.current_index >= 0:
            self.save_all_to_memory(self.current_index)

        path = "data/scenarios.json"
        if not os.path.exists("data") and os.path.exists("../data"):
            path = "../data/scenarios.json"

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.scenarios, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, tr("Success"), tr("Scenarios saved successfully!"))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to save: {e}")

    def clear_inputs(self):
        self.is_updating_ui = True
        self.name_edit.clear()
        self.desc_edit.clear()
        self.spin_my_mana.setValue(0)
        self.spin_enemy_shields.setValue(5)
        self.check_enemy_trigger.setChecked(False)
        self.check_loop_proof.setChecked(False)
        for lst in self.zone_lists.values():
            lst.clear()
        self.is_updating_ui = False

    def enable_inputs(self, enable):
        self.name_edit.setEnabled(enable)
        self.desc_edit.setEnabled(enable)
        self.tabs.setEnabled(enable)
