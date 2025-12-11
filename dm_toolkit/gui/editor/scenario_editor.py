from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QSplitter, QWidget,
                             QTabWidget, QFormLayout, QSpinBox, QCheckBox)
from PyQt6.QtCore import Qt
import json
import os
from dm_toolkit.gui.localization import tr

class ScenarioEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scenario Editor"))
        self.resize(1000, 700)
        self.scenarios = []
        self.current_index = -1
        self.is_updating_ui = False
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left: List
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

        # Right: Editor
        right_layout = QVBoxLayout()

        # Header Info
        header_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self.save_header_to_memory)
        header_layout.addRow(tr("Name (ID):"), self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.editingFinished.connect(self.save_header_to_memory)
        header_layout.addRow(tr("Description:"), self.desc_edit)
        right_layout.addLayout(header_layout)

        # Tabs for Config
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

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

        # Zone Tabs helper
        self.zone_edits = {}
        self.create_zone_tab("my_hand_cards", tr("My Hand"))
        self.create_zone_tab("my_battle_zone", tr("My Battle Zone"))
        self.create_zone_tab("my_mana_zone", tr("My Mana Zone"))
        self.create_zone_tab("my_grave_yard", tr("My Graveyard"))
        self.create_zone_tab("my_shields", tr("My Shields")) # Distinct from count
        self.create_zone_tab("enemy_battle_zone", tr("Enemy Battle Zone"))

        # Connect signals for auto-save to memory
        self.spin_my_mana.valueChanged.connect(self.save_config_to_memory)
        self.spin_enemy_shields.valueChanged.connect(self.save_config_to_memory)
        self.check_enemy_trigger.stateChanged.connect(self.save_config_to_memory)
        self.check_loop_proof.stateChanged.connect(self.save_config_to_memory)

        self.btn_save = QPushButton(tr("Save All to File"))
        self.btn_save.clicked.connect(self.save_to_file)
        right_layout.addWidget(self.btn_save)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        self.enable_inputs(False)

    def create_zone_tab(self, key, title):
        tab = QWidget()
        vbox = QVBoxLayout()
        lbl = QLabel(tr("Enter Card IDs or Names (one per line):"))
        vbox.addWidget(lbl)
        edit = QTextEdit()
        edit.setPlaceholderText("e.g.\nBronze-Arm Tribe\n1005\n...")
        # Connect textChanged to save? Might be too heavy. Use focus out or explicit save.
        # Let's use focusOut via event filter or just rely on 'Save to Memory' when switching or saving file.
        # Ideally, we update memory when switching list items.
        vbox.addWidget(edit)
        tab.setLayout(vbox)
        self.tabs.addTab(tab, title)
        self.zone_edits[key] = edit

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
        # Save previous selection to memory
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

            for key, edit in self.zone_edits.items():
                val = config.get(key, [])
                if isinstance(val, list):
                    # Convert list to string lines
                    text = "\n".join(str(x) for x in val)
                    edit.setText(text)
                else:
                    edit.clear()

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

        # Parse Zone Texts
        for key, edit in self.zone_edits.items():
            text = edit.toPlainText()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            parsed_list = []
            for line in lines:
                # Try to parse as int (ID), else keep as string (Name)
                try:
                    parsed_list.append(int(line))
                except ValueError:
                    parsed_list.append(line)
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
        for edit in self.zone_edits.values():
            edit.clear()
        self.is_updating_ui = False

    def enable_inputs(self, enable):
        self.name_edit.setEnabled(enable)
        self.desc_edit.setEnabled(enable)
        self.tabs.setEnabled(enable)
