import json
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QPushButton, QMessageBox, QFormLayout, QWidget, QCheckBox, QGridLayout, QScrollArea, QTabWidget, QGroupBox, QListWidget, QListWidgetItem
)
from gui.widgets.card_widget import CardWidget

class CardEditor(QDialog):
    def __init__(self, json_path, parent=None):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle("JSON Card Editor")
        self.resize(800, 600)
        self.cards_data = []
        self.current_card_index = -1
        self.load_data()
        self.init_ui()

    def load_data(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.cards_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")
                self.cards_data = []
        else:
            self.cards_data = []

    def save_data(self):
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.cards_data, f, indent=2)
            QMessageBox.information(self, "Success", "Cards saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON: {e}")

    def init_ui(self):
        # Create layouts without setting parent immediately
        
        # Left: Card List
        list_layout = QVBoxLayout()
        self.card_list = QListWidget()
        self.card_list.currentRowChanged.connect(self.load_selected_card)
        list_layout.addWidget(self.card_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("New Card")
        add_btn.clicked.connect(self.create_new_card)
        del_btn = QPushButton("Delete Card")
        del_btn.clicked.connect(self.delete_current_card)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        list_layout.addLayout(btn_layout)

        # Middle: Form (Tabs)
        self.tabs = QTabWidget()

        # Tab 1: Basic Info & Keywords
        self.basic_tab = QWidget()
        self.setup_basic_tab()
        self.tabs.addTab(self.basic_tab, "Basic Info")

        # Tab 2: Effects (JSON View/Simple Edit)
        self.effects_tab = QWidget()
        self.setup_effects_tab()
        self.tabs.addTab(self.effects_tab, "Effects")

        # Right: Preview
        preview_layout = QVBoxLayout()
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_container = QWidget()
        self.preview_container_layout = QVBoxLayout(self.preview_container)
        self.preview_card = None

        preview_layout.addWidget(self.preview_container)
        preview_layout.addStretch()

        # Bottom Buttons
        action_layout = QHBoxLayout()
        save_btn = QPushButton("Save All")
        save_btn.clicked.connect(self.save_data)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        action_layout.addWidget(save_btn)
        action_layout.addWidget(close_btn)

        # Assemble Layouts
        # Top container for 3 columns
        top_layout = QHBoxLayout()
        top_layout.addLayout(list_layout, 1)
        top_layout.addWidget(self.tabs, 2)
        top_layout.addLayout(preview_layout, 1)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.addLayout(top_layout)
        root_layout.addLayout(action_layout)

        self.refresh_list()

    def setup_basic_tab(self):
        layout = QVBoxLayout(self.basic_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.id_input = QSpinBox()
        self.id_input.setRange(1, 9999)
        form.addRow("ID:", self.id_input)

        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self.update_preview)
        self.name_input.textChanged.connect(self.update_current_card_data) # Auto-update data
        form.addRow("Name:", self.name_input)

        self.civ_input = QComboBox()
        self.civ_input.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        self.civ_input.currentTextChanged.connect(self.update_preview)
        self.civ_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow("Civilization:", self.civ_input)

        self.type_input = QComboBox()
        self.type_input.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
        self.type_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow("Type:", self.type_input)

        self.cost_input = QSpinBox()
        self.cost_input.setRange(0, 99)
        self.cost_input.valueChanged.connect(self.update_preview)
        self.cost_input.valueChanged.connect(self.update_current_card_data)
        form.addRow("Cost:", self.cost_input)

        self.power_input = QSpinBox()
        self.power_input.setRange(0, 99999)
        self.power_input.setSingleStep(500)
        self.power_input.valueChanged.connect(self.update_preview)
        self.power_input.valueChanged.connect(self.update_current_card_data)
        form.addRow("Power:", self.power_input)

        self.races_input = QLineEdit()
        self.races_input.setPlaceholderText("Comma separated (e.g. Human, Dragon)")
        self.races_input.textChanged.connect(self.update_current_card_data)
        form.addRow("Races:", self.races_input)

        # Keywords
        keywords_label = QLabel("Keywords (Maps to PASSIVE_CONST effects):")
        form.addRow(keywords_label)
        
        self.keywords_layout = QGridLayout()
        self.keyword_checkboxes = {}
        keywords_list = [
            "BLOCKER", "SPEED_ATTACKER", "SLAYER",
            "DOUBLE_BREAKER", "TRIPLE_BREAKER", "POWER_ATTACKER",
            "EVOLUTION", "MACH_FIGHTER", "G_STRIKE"
        ]
        
        for i, kw in enumerate(keywords_list):
            cb = QCheckBox(kw)
            cb.stateChanged.connect(self.update_current_card_data)
            self.keyword_checkboxes[kw] = cb
            self.keywords_layout.addWidget(cb, i // 2, i % 2)
            
        form.addRow(self.keywords_layout)

        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

    def setup_effects_tab(self):
        layout = QVBoxLayout(self.effects_tab)

        info_label = QLabel("For Phase 3 MVP, raw JSON editing for effects is recommended.\n"
                            "Keywords in the Basic tab are automatically synced to 'PASSIVE_CONST' effects.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # We can use a simple text area to show the effects JSON for now
        # Implementing a full nested effect editor is complex.
        # But we should at least allow adding standard triggers.

        # List of effects
        self.effects_list = QListWidget()
        layout.addWidget(self.effects_list)

        btn_layout = QHBoxLayout()
        add_eff_btn = QPushButton("Add Effect (Template)")
        add_eff_btn.clicked.connect(self.add_effect_template)
        rem_eff_btn = QPushButton("Remove Selected Effect")
        rem_eff_btn.clicked.connect(self.remove_effect)
        btn_layout.addWidget(add_eff_btn)
        btn_layout.addWidget(rem_eff_btn)
        layout.addLayout(btn_layout)
        
        # Note: editing details of an effect requires more UI.
        # For MVP, maybe just editing the JSON string of the selected effect?
        self.effect_json_edit = QLineEdit() # Single line for now, maybe QTextEdit later
        self.effect_json_edit.setPlaceholderText("Effect JSON representation")
        self.effect_json_edit.editingFinished.connect(self.update_effect_from_json)
        layout.addWidget(self.effect_json_edit)

    def refresh_list(self):
        self.card_list.clear()
        for card in self.cards_data:
            item = QListWidgetItem(f"{card.get('id')} - {card.get('name')}")
            item.setData(Qt.ItemDataRole.UserRole, card.get('id'))
            self.card_list.addItem(item)

    def create_new_card(self):
        new_id = 1
        if self.cards_data:
            new_id = max(c.get('id', 0) for c in self.cards_data) + 1
        
        new_card = {
            "id": new_id,
            "name": "New Card",
            "civilization": "FIRE",
            "type": "CREATURE",
            "cost": 1,
            "power": 1000,
            "races": [],
            "effects": []
        }
        self.cards_data.append(new_card)
        self.refresh_list()
        self.card_list.setCurrentRow(self.card_list.count() - 1)

    def delete_current_card(self):
        row = self.card_list.currentRow()
        if row >= 0:
            del self.cards_data[row]
            self.refresh_list()
            self.current_card_index = -1
            # Clear UI?

    def load_selected_card(self, row):
        if row < 0 or row >= len(self.cards_data):
            return
        
        self.current_card_index = row
        card = self.cards_data[row]
        
        # Block signals to prevent auto-update loop
        self.block_signals_recursive(True)

        self.id_input.setValue(card.get('id', 0))
        self.name_input.setText(card.get('name', ''))
        self.civ_input.setCurrentText(card.get('civilization', 'FIRE'))
        self.type_input.setCurrentText(card.get('type', 'CREATURE'))
        self.cost_input.setValue(card.get('cost', 0))
        self.power_input.setValue(card.get('power', 0))
        
        races = card.get('races', [])
        self.races_input.setText(", ".join(races))
        
        # Keywords check
        # Scan effects for PASSIVE_CONST and action.str_val
        effects = card.get('effects', [])
        active_keywords = set()
        for eff in effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                for act in eff.get('actions', []):
                    if 'str_val' in act:
                        active_keywords.add(act['str_val'])
        
        for kw, cb in self.keyword_checkboxes.items():
            cb.setChecked(kw in active_keywords)

        # Effects List
        self.effects_list.clear()
        for i, eff in enumerate(effects):
            trig = eff.get('trigger', 'NONE')
            self.effects_list.addItem(f"Effect {i}: {trig}")

        self.block_signals_recursive(False)
        self.update_preview()

    def block_signals_recursive(self, block):
        self.id_input.blockSignals(block)
        self.name_input.blockSignals(block)
        self.civ_input.blockSignals(block)
        self.type_input.blockSignals(block)
        self.cost_input.blockSignals(block)
        self.power_input.blockSignals(block)
        self.races_input.blockSignals(block)
        for cb in self.keyword_checkboxes.values():
            cb.blockSignals(block)

    def update_current_card_data(self):
        if self.current_card_index < 0:
            return

        card = self.cards_data[self.current_card_index]

        card['id'] = self.id_input.value()
        card['name'] = self.name_input.text()
        card['civilization'] = self.civ_input.currentText()
        card['type'] = self.type_input.currentText()
        card['cost'] = self.cost_input.value()
        card['power'] = self.power_input.value()

        races_str = self.races_input.text()
        card['races'] = [r.strip() for r in races_str.split(',')] if races_str.strip() else []

        # Handle Keywords -> Effects sync
        # 1. Remove existing PASSIVE_CONST effects that match our known keywords
        # This is a bit destructive if user added custom PASSIVE_CONST, but for MVP it is safe to manage them here.
        new_effects = []
        existing_effects = card.get('effects', [])

        # Keep non-keyword effects
        known_keywords = set(self.keyword_checkboxes.keys())

        for eff in existing_effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                # Check if this effect is purely one of our managed keywords
                # If it has multiple actions or unknown keywords, keep it?
                # Simplify: Rebuild all keyword effects.
                # If an effect has actions not in our known list, keep it.
                actions_to_keep = []
                for act in eff.get('actions', []):
                    if act.get('str_val') not in known_keywords:
                        actions_to_keep.append(act)

                if actions_to_keep:
                    eff['actions'] = actions_to_keep
                    new_effects.append(eff)
            else:
                new_effects.append(eff)

        # 2. Add selected keywords as new PASSIVE_CONST effects
        # We can group them or separate them. Let's group them into one PASSIVE_CONST effect if possible,
        # or one per keyword for simplicity. Grouping is cleaner.

        active_kws = []
        for kw, cb in self.keyword_checkboxes.items():
            if cb.isChecked():
                active_kws.append(kw)

        if active_kws:
            # Create actions
            actions = []
            for kw in active_kws:
                actions.append({
                    "type": "NONE", # Usually NONE for keywords, just str_val matters? Or specific type?
                    # In json_loader.cpp: `if (action.str_val == "BLOCKER")`... checks str_val.
                    # Type seems ignored or NONE.
                    "scope": "NONE",
                    "filter": {},
                    "value1": 0,
                    "value2": 0,
                    "str_val": kw
                })

            new_effects.append({
                "trigger": "PASSIVE_CONST",
                "condition": {"type": "NONE", "value": 0, "str_val": ""},
                "actions": actions
            })

        card['effects'] = new_effects

        # Refresh UI list item text
        item = self.card_list.item(self.current_card_index)
        if item:
            item.setText(f"{card['id']} - {card['name']}")

    def update_preview(self):
        # Clear old preview
        for i in reversed(range(self.preview_container_layout.count())):
            item = self.preview_container_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        name = self.name_input.text() or "New Card"
        cost = self.cost_input.value()
        power = self.power_input.value()
        civ = self.civ_input.currentText()
        
        self.preview_card = CardWidget(0, name, cost, power, civ)
        self.preview_container_layout.addWidget(self.preview_card)

    def add_effect_template(self):
        if self.current_card_index < 0: return

        # Ask user for template type
        from PyQt6.QtWidgets import QInputDialog
        types = ["Draw Card (ON_PLAY)", "Destroy Creature (ON_PLAY)", "Return to Hand (ON_PLAY)", "Mekraid (ON_PLAY)", "Search Top Deck (ON_PLAY)"]
        item, ok = QInputDialog.getItem(self, "Select Template", "Effect Template:", types, 0, False)

        if ok and item:
            new_eff = {
                "trigger": "ON_PLAY",
                "condition": {"type": "NONE"},
                "actions": []
            }

            if "Draw Card" in item:
                new_eff["actions"].append({
                    "type": "DRAW_CARD",
                    "scope": "PLAYER_SELF",
                    "value1": 1
                })
            elif "Destroy Creature" in item:
                new_eff["actions"].append({
                    "type": "DESTROY",
                    "scope": "TARGET_SELECT",
                    "value1": 1,
                    "filter": {"owner": "OPPONENT", "types": ["CREATURE"], "count": 1}
                })
            elif "Return to Hand" in item:
                new_eff["actions"].append({
                    "type": "RETURN_TO_HAND",
                    "scope": "TARGET_SELECT",
                    "value1": 1,
                    "filter": {"owner": "OPPONENT", "types": ["CREATURE"], "count": 1}
                })
            elif "Mekraid" in item:
                new_eff["actions"].append({
                    "type": "MEKRAID",
                    "scope": "PLAYER_SELF",
                    "value1": 3,
                    "filter": {"races": ["Magic"], "max_cost": 5}
                })
            elif "Search Top Deck" in item:
                new_eff["actions"].append({
                    "type": "SEARCH_DECK_BOTTOM",
                    "scope": "PLAYER_SELF",
                    "value1": 3,
                    "filter": {"types": ["SPELL"], "count": 1}
                })

            self.cards_data[self.current_card_index]['effects'].append(new_eff)

        # Refresh Effects List
        self.load_selected_card(self.current_card_index) # Reloads UI

    def remove_effect(self):
        if self.current_card_index < 0: return
        row = self.effects_list.currentRow()
        if row >= 0:
            del self.cards_data[self.current_card_index]['effects'][row]
            self.load_selected_card(self.current_card_index)

    def update_effect_from_json(self):
        if self.current_card_index < 0: return

        text = self.effect_json_edit.text()
        if not text.strip(): return

        try:
            effect_obj = json.loads(text)
            # Basic validation
            if not isinstance(effect_obj, dict):
                raise ValueError("Effect must be a JSON object")
            if "trigger" not in effect_obj:
                raise ValueError("Effect must have a 'trigger'")

            # Update selected effect
            row = self.effects_list.currentRow()
            if row >= 0:
                self.cards_data[self.current_card_index]['effects'][row] = effect_obj
                self.load_selected_card(self.current_card_index) # Refresh UI
                QMessageBox.information(self, "Success", "Effect updated from JSON.")
        except Exception as e:
            QMessageBox.warning(self, "Invalid JSON", f"Could not parse effect JSON: {e}")
