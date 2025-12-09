from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QSplitter, QWidget)
from PyQt6.QtCore import Qt
import json
import os
from dm_toolkit.gui.localization import tr

class ScenarioEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scenario Editor"))
        self.resize(800, 600)
        self.scenarios = []
        self.current_index = -1
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

        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(lambda: self.update_memory_from_ui(self.current_index))
        right_layout.addWidget(QLabel(tr("Name (ID)")))
        right_layout.addWidget(self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.editingFinished.connect(lambda: self.update_memory_from_ui(self.current_index))
        right_layout.addWidget(QLabel(tr("Description")))
        right_layout.addWidget(self.desc_edit)

        self.config_edit = QTextEdit()
        self.config_edit.setPlaceholderText('{\n  "my_mana": 0,\n  "my_hand_cards": []\n}')
        # Trigger update on focus lost? Or text changed?
        # Text changed is too frequent. Let's use focusOutEvent subclassing or just button save.
        # But we want to sync when switching rows. on_selection_changed handles that.
        right_layout.addWidget(QLabel(tr("Configuration (JSON)")))
        right_layout.addWidget(self.config_edit)

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
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        self.enable_inputs(False)

    def load_data(self):
        # Load from data/scenarios.json
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
        # Save previous selection to memory before switching?
        # self.current_index is the OLD index.
        if self.current_index >= 0 and self.current_index < len(self.scenarios):
             self.update_memory_from_ui(self.current_index)

        self.current_index = row
        if row >= 0:
            item = self.scenarios[row]
            self.name_edit.setText(item.get("name", ""))
            self.desc_edit.setText(item.get("description", ""))
            config = item.get("config", {})
            self.config_edit.setText(json.dumps(config, indent=2))
            self.enable_inputs(True)
        else:
            self.clear_inputs()
            self.enable_inputs(False)

    def update_memory_from_ui(self, index):
        if index < 0 or index >= len(self.scenarios): return

        item = self.scenarios[index]
        item["name"] = self.name_edit.text()
        item["description"] = self.desc_edit.text()
        try:
            config = json.loads(self.config_edit.toPlainText())
            item["config"] = config
        except json.JSONDecodeError:
            # Maybe show a small indicator or status bar?
            print(f"Invalid JSON for scenario {item['name']}")
            pass

        # Update list item text
        self.list_widget.item(index).setText(item["name"])

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
            self.update_memory_from_ui(self.current_index)

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
        self.name_edit.clear()
        self.desc_edit.clear()
        self.config_edit.clear()

    def enable_inputs(self, enable):
        self.name_edit.setEnabled(enable)
        self.desc_edit.setEnabled(enable)
        self.config_edit.setEnabled(enable)
