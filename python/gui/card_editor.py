import json
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QPushButton, QMessageBox, QFormLayout, QWidget, QCheckBox, QGridLayout, QScrollArea, QTabWidget, QGroupBox, QListWidget, QListWidgetItem,
    QStackedWidget, QTextEdit
)
from gui.widgets.card_widget import CardWidget
from gui.localization import tr

class CardEditor(QDialog):
    def __init__(self, json_path, parent=None):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle(tr("Card Editor"))
        self.resize(1000, 700)
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
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load JSON')}: {e}")
                self.cards_data = []
        else:
            self.cards_data = []

    def save_data(self):
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.cards_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, tr("Success"), tr("Cards saved successfully!"))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"{tr('Failed to save JSON')}: {e}")

    def init_ui(self):
        # Left: Card List
        list_layout = QVBoxLayout()
        self.card_list = QListWidget()
        self.card_list.currentRowChanged.connect(self.load_selected_card)
        list_layout.addWidget(self.card_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton(tr("New Card"))
        add_btn.clicked.connect(self.create_new_card)

        del_btn = QPushButton(tr("Delete Card"))
        del_btn.clicked.connect(self.delete_current_card)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        list_layout.addLayout(btn_layout)

        # Middle: Form (Tabs)
        self.tabs = QTabWidget()

        # Tab 1: Basic Info & Keywords
        self.basic_tab = QWidget()
        self.setup_basic_tab()
        self.tabs.addTab(self.basic_tab, tr("Card Details"))

        # Tab 2: Effects (Visual Builder)
        self.effects_tab = QWidget()
        self.setup_effects_tab()
        self.tabs.addTab(self.effects_tab, tr("Effects"))

        # Right: Preview
        preview_layout = QVBoxLayout()
        preview_label = QLabel(tr("Preview"))
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_container = QWidget()
        self.preview_container_layout = QVBoxLayout(self.preview_container)
        self.preview_card = None

        preview_layout.addWidget(self.preview_container)
        preview_layout.addStretch()

        # Bottom Buttons
        action_layout = QHBoxLayout()
        save_btn = QPushButton(tr("Save to JSON"))
        save_btn.clicked.connect(self.save_data)
        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.reject)
        action_layout.addWidget(save_btn)
        action_layout.addWidget(close_btn)

        # Assemble Layouts
        top_layout = QHBoxLayout()
        top_layout.addLayout(list_layout, 1)
        top_layout.addWidget(self.tabs, 3)
        top_layout.addLayout(preview_layout, 1)

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
        form.addRow(tr("ID") + ":", self.id_input)

        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self.update_preview)
        self.name_input.textChanged.connect(self.update_current_card_data)
        form.addRow(tr("Name") + ":", self.name_input)

        self.civ_input = QComboBox()
        self.civ_input.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        self.civ_input.currentTextChanged.connect(self.update_preview)
        self.civ_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow(tr("Civilization") + ":", self.civ_input)

        self.type_input = QComboBox()
        self.type_input.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
        self.type_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow(tr("Type") + ":", self.type_input)

        self.cost_input = QSpinBox()
        self.cost_input.setRange(0, 99)
        self.cost_input.valueChanged.connect(self.update_preview)
        self.cost_input.valueChanged.connect(self.update_current_card_data)
        form.addRow(tr("Cost") + ":", self.cost_input)

        self.power_input = QSpinBox()
        self.power_input.setRange(0, 99999)
        self.power_input.setSingleStep(500)
        self.power_input.valueChanged.connect(self.update_preview)
        self.power_input.valueChanged.connect(self.update_current_card_data)
        form.addRow(tr("Power") + ":", self.power_input)

        self.races_input = QLineEdit()
        self.races_input.setPlaceholderText(tr("Races"))
        self.races_input.textChanged.connect(self.update_current_card_data)
        form.addRow(tr("Races") + ":", self.races_input)

        # Keywords
        keywords_label = QLabel(tr("Keywords") + ":")
        form.addRow(keywords_label)
        
        self.keywords_layout = QGridLayout()
        self.keyword_checkboxes = {}
        keywords_list = [
            "BLOCKER", "SPEED_ATTACKER", "SLAYER",
            "DOUBLE_BREAKER", "TRIPLE_BREAKER", "POWER_ATTACKER",
            "EVOLUTION", "MACH_FIGHTER", "G_STRIKE",
            "JUST_DIVER"
        ]
        
        for i, kw in enumerate(keywords_list):
            cb = QCheckBox()

            label = kw.replace("_", " ").title() # "SPEED_ATTACKER" -> "Speed Attacker"
            if kw == "BLOCKER": label = "Blocker"
            # Use localization key if it matches English enum name or custom key
            # Assuming keys in localization.py handle specific keyword strings if added
            # For now use English label wrapped in tr() which returns input if no match
            # But "Blocker" -> "ブロッカー" is in localization.py

            cb.setText(tr(label))

            cb.stateChanged.connect(self.update_current_card_data)
            self.keyword_checkboxes[kw] = cb
            self.keywords_layout.addWidget(cb, i // 2, i % 2)
            
        form.addRow(self.keywords_layout)

        # Shield Trigger (Special Keyword)
        self.shield_trigger_cb = QCheckBox(tr("S_TRIGGER"))
        self.shield_trigger_cb.stateChanged.connect(self.update_current_card_data)
        form.addRow(self.shield_trigger_cb)

        # Hyper Energy (Special Action)
        self.hyper_energy_cb = QCheckBox(tr("Hyper Energy")) # tr() will return English if not found, but it's consistent
        self.hyper_energy_cb.stateChanged.connect(self.update_current_card_data)
        form.addRow(self.hyper_energy_cb)

        # Revolution Change
        self.rev_change_group = QGroupBox(tr("REVOLUTION_CHANGE"))
        self.rev_change_group.setCheckable(True)
        self.rev_change_group.setChecked(False)
        self.rev_change_group.toggled.connect(self.update_current_card_data)
        rev_layout = QGridLayout(self.rev_change_group)

        rev_layout.addWidget(QLabel(tr("Civilization") + ":"), 0, 0)
        self.rev_civ_input = QLineEdit()
        self.rev_civ_input.setPlaceholderText("FIRE, etc.")
        self.rev_civ_input.textChanged.connect(self.update_current_card_data)
        rev_layout.addWidget(self.rev_civ_input, 0, 1)

        rev_layout.addWidget(QLabel(tr("Races") + ":"), 1, 0)
        self.rev_race_input = QLineEdit()
        self.rev_race_input.setPlaceholderText("Dragon, etc.")
        self.rev_race_input.textChanged.connect(self.update_current_card_data)
        rev_layout.addWidget(self.rev_race_input, 1, 1)

        rev_layout.addWidget(QLabel(tr("Min Cost") + ":"), 2, 0)
        self.rev_cost_spin = QSpinBox()
        self.rev_cost_spin.setRange(0, 99)
        self.rev_cost_spin.valueChanged.connect(self.update_current_card_data)
        rev_layout.addWidget(self.rev_cost_spin, 2, 1)

        form.addRow(self.rev_change_group)

        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

    def setup_effects_tab(self):
        layout = QVBoxLayout(self.effects_tab)

        # Split: List of Effects (Top) and Effect Detail Editor (Bottom)

        # Top: List
        list_group = QGroupBox(tr("Effects"))
        list_layout = QVBoxLayout(list_group)
        self.effects_list = QListWidget()
        self.effects_list.currentRowChanged.connect(self.load_selected_effect)
        list_layout.addWidget(self.effects_list)

        btn_layout = QHBoxLayout()
        add_eff_btn = QPushButton(tr("Add Effect"))
        add_eff_btn.clicked.connect(self.create_new_effect)
        rem_eff_btn = QPushButton(tr("Remove Effect"))
        rem_eff_btn.clicked.connect(self.remove_effect)
        btn_layout.addWidget(add_eff_btn)
        btn_layout.addWidget(rem_eff_btn)
        list_layout.addLayout(btn_layout)

        layout.addWidget(list_group, 1)

        # Bottom: Editor
        editor_group = QGroupBox(tr("Effect"))
        editor_layout = QVBoxLayout(editor_group)

        # Trigger
        trig_layout = QHBoxLayout()
        trig_layout.addWidget(QLabel(tr("Trigger") + ":"))
        self.eff_trigger_combo = QComboBox()
        triggers = ["ON_PLAY", "ON_ATTACK", "ON_DESTROY", "PASSIVE_CONST", "TURN_START", "ON_OTHER_ENTER", "ON_ATTACK_FROM_HAND"]
        for t in triggers:
            self.eff_trigger_combo.addItem(tr(t), t)
        trig_layout.addWidget(self.eff_trigger_combo)
        editor_layout.addLayout(trig_layout)

        # Condition
        cond_layout = QHBoxLayout()
        cond_layout.addWidget(QLabel(tr("Condition") + ":"))

        self.eff_cond_type_combo = QComboBox()
        cond_types = ["NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH", "OPPONENT_PLAYED_WITHOUT_MANA"]
        for c in cond_types:
            self.eff_cond_type_combo.addItem(c, c)
        cond_layout.addWidget(self.eff_cond_type_combo)

        cond_layout.addWidget(QLabel(tr("Value 1") + ":"))
        self.eff_cond_val_spin = QSpinBox()
        self.eff_cond_val_spin.setRange(0, 99)
        cond_layout.addWidget(self.eff_cond_val_spin)

        cond_layout.addWidget(QLabel(tr("String Value") + ":"))
        self.eff_cond_str_edit = QLineEdit()
        self.eff_cond_str_edit.setPlaceholderText("FIRE, etc.")
        cond_layout.addWidget(self.eff_cond_str_edit)

        editor_layout.addLayout(cond_layout)

        # Action List inside Effect
        self.eff_action_list_widget = QListWidget()
        self.eff_action_list_widget.setFixedHeight(80)
        self.eff_action_list_widget.currentRowChanged.connect(self.load_selected_action)
        editor_layout.addWidget(QLabel(tr("Actions") + ":"))
        editor_layout.addWidget(self.eff_action_list_widget)

        act_btn_layout = QHBoxLayout()
        add_act_btn = QPushButton(tr("Add Action"))
        add_act_btn.clicked.connect(self.add_action_to_effect)
        rem_act_btn = QPushButton(tr("Remove Action"))
        rem_act_btn.clicked.connect(self.remove_action_from_effect)
        act_btn_layout.addWidget(add_act_btn)
        act_btn_layout.addWidget(rem_act_btn)
        editor_layout.addLayout(act_btn_layout)

        # Action Detail Editor
        self.action_detail_group = QGroupBox(tr("Action"))
        detail_form = QFormLayout(self.action_detail_group)

        self.act_type_combo = QComboBox()
        actions = [
            "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP",
            "COST_REFERENCE", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
            "REVOLUTION_CHANGE", "COUNT_CARDS", "GET_GAME_STAT", "APPLY_MODIFIER", "REVEAL_CARDS",
            "REGISTER_DELAYED_EFFECT", "RESET_INSTANCE"
        ]
        for a in actions:
            self.act_type_combo.addItem(tr(a), a)
        detail_form.addRow(tr("Action Type") + ":", self.act_type_combo)

        self.act_scope_combo = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"]
        for s in scopes:
            self.act_scope_combo.addItem(tr(s), s)
        detail_form.addRow(tr("Scope") + ":", self.act_scope_combo)

        # Filter Settings
        filter_box = QGroupBox(tr("Filter"))
        filter_layout = QGridLayout(filter_box)

        filter_layout.addWidget(QLabel(tr("Zones") + ":"), 0, 0)
        self.zone_checks = {}
        zones = ["BATTLE_ZONE", "MANA_ZONE", "HAND", "GRAVEYARD", "SHIELD_ZONE", "DECK"]
        zones_layout = QGridLayout()
        for i, z in enumerate(zones):
            cb = QCheckBox(tr(z))
            self.zone_checks[z] = cb
            zones_layout.addWidget(cb, i // 3, i % 3)
        filter_layout.addLayout(zones_layout, 0, 1)

        filter_layout.addWidget(QLabel(tr("Target Player") + ":"), 1, 0)
        self.filter_player_combo = QComboBox()
        self.filter_player_combo.addItems(["NONE", "SELF", "OPPONENT", "BOTH"])
        filter_layout.addWidget(self.filter_player_combo, 1, 1)

        filter_layout.addWidget(QLabel(tr("Card Types") + ":"), 2, 0)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["NONE", "CREATURE", "SPELL"])
        filter_layout.addWidget(self.filter_type_combo, 2, 1)

        filter_layout.addWidget(QLabel(tr("Count") + ":"), 3, 0)
        self.filter_count_spin = QSpinBox()
        self.filter_count_spin.setRange(0, 20)
        filter_layout.addWidget(self.filter_count_spin, 3, 1)

        # New Filter Fields (Civilization, Race, Cost, Power, Tapped, Blocker, Evolution)

        filter_layout.addWidget(QLabel(tr("Civilizations") + ":"), 4, 0)
        self.filter_civ_input = QLineEdit()
        self.filter_civ_input.setPlaceholderText("FIRE, DARKNESS")
        filter_layout.addWidget(self.filter_civ_input, 4, 1)

        filter_layout.addWidget(QLabel(tr("Races") + ":"), 5, 0)
        self.filter_race_input = QLineEdit()
        self.filter_race_input.setPlaceholderText("Dragon, Cyber Lord")
        filter_layout.addWidget(self.filter_race_input, 5, 1)

        # Cost Range
        cost_layout = QHBoxLayout()
        self.filter_min_cost_spin = QSpinBox()
        self.filter_min_cost_spin.setRange(-1, 99)
        self.filter_min_cost_spin.setValue(-1) # -1 means Ignore
        cost_layout.addWidget(QLabel(tr("Min") + ":"))
        cost_layout.addWidget(self.filter_min_cost_spin)
        self.filter_max_cost_spin = QSpinBox()
        self.filter_max_cost_spin.setRange(-1, 99)
        self.filter_max_cost_spin.setValue(-1)
        cost_layout.addWidget(QLabel(tr("Max") + ":"))
        cost_layout.addWidget(self.filter_max_cost_spin)
        filter_layout.addWidget(QLabel(tr("Cost") + ":"), 6, 0)
        filter_layout.addLayout(cost_layout, 6, 1)

        # Power Range
        power_layout = QHBoxLayout()
        self.filter_min_power_spin = QSpinBox()
        self.filter_min_power_spin.setRange(-1, 99999)
        self.filter_min_power_spin.setSingleStep(500)
        self.filter_min_power_spin.setValue(-1)
        power_layout.addWidget(QLabel(tr("Min") + ":"))
        power_layout.addWidget(self.filter_min_power_spin)
        self.filter_max_power_spin = QSpinBox()
        self.filter_max_power_spin.setRange(-1, 99999)
        self.filter_max_power_spin.setSingleStep(500)
        self.filter_max_power_spin.setValue(-1)
        power_layout.addWidget(QLabel(tr("Max") + ":"))
        power_layout.addWidget(self.filter_max_power_spin)
        filter_layout.addWidget(QLabel(tr("Power") + ":"), 7, 0)
        filter_layout.addLayout(power_layout, 7, 1)

        # Boolean Flags (Tapped, Blocker, Evolution)
        # Using Combo for Ignore/True/False
        flags_layout = QGridLayout()

        flags_layout.addWidget(QLabel(tr("Tapped") + ":"), 0, 0)
        self.filter_tapped_combo = QComboBox()
        self.filter_tapped_combo.addItems([tr("Ignore"), tr("True"), tr("False")])
        flags_layout.addWidget(self.filter_tapped_combo, 0, 1)

        flags_layout.addWidget(QLabel(tr("Blocker") + ":"), 1, 0)
        self.filter_blocker_combo = QComboBox()
        self.filter_blocker_combo.addItems([tr("Ignore"), tr("True"), tr("False")])
        flags_layout.addWidget(self.filter_blocker_combo, 1, 1)

        flags_layout.addWidget(QLabel(tr("Evolution") + ":"), 2, 0)
        self.filter_evolution_combo = QComboBox()
        self.filter_evolution_combo.addItems([tr("Ignore"), tr("True"), tr("False")])
        flags_layout.addWidget(self.filter_evolution_combo, 2, 1)

        filter_layout.addLayout(flags_layout, 8, 0, 1, 2)
        
        detail_form.addRow(filter_box)

        # Generic Values
        self.val1_label = QLabel(tr("Value 1") + ":")
        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(0, 99)
        detail_form.addRow(self.val1_label, self.val1_spin)

        self.val2_label = QLabel(tr("Value 2") + ":")
        self.val2_spin = QSpinBox()
        self.val2_spin.setRange(0, 99)
        detail_form.addRow(self.val2_label, self.val2_spin)

        self.str_val_edit = QLineEdit()
        detail_form.addRow(tr("String Value") + ":", self.str_val_edit)

        # Phase 5: Dynamic Variables
        self.input_key_edit = QLineEdit()
        detail_form.addRow(tr("Input Key") + ":", self.input_key_edit)
        self.output_key_edit = QLineEdit()
        detail_form.addRow(tr("Output Key") + ":", self.output_key_edit)

        # Connect Action Type change to dynamic labels
        self.act_type_combo.currentIndexChanged.connect(self.update_dynamic_labels)

        # Apply Button
        apply_btn = QPushButton(tr("Update Action"))
        apply_btn.clicked.connect(self.apply_action_changes)
        detail_form.addRow(apply_btn)

        editor_layout.addWidget(self.action_detail_group)
        layout.addWidget(editor_group, 2)

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
            # Clear UI

    def load_selected_card(self, row):
        if row < 0 or row >= len(self.cards_data):
            return
        
        self.current_card_index = row
        card = self.cards_data[row]
        
        self.block_signals_recursive(True)

        self.id_input.setValue(card.get('id', 0))
        self.name_input.setText(card.get('name', ''))
        self.civ_input.setCurrentText(card.get('civilization', 'FIRE'))
        self.type_input.setCurrentText(card.get('type', 'CREATURE'))
        self.cost_input.setValue(card.get('cost', 0))
        self.power_input.setValue(card.get('power', 0))
        
        races = card.get('races', [])
        self.races_input.setText(", ".join(races))
        
        # Keywords check (PASSIVE_CONST sync)
        effects = card.get('effects', [])
        active_keywords = set()
        for eff in effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                for act in eff.get('actions', []):
                    if 'str_val' in act:
                        active_keywords.add(act['str_val'])
        
        for kw, cb in self.keyword_checkboxes.items():
            cb.setChecked(kw in active_keywords)

        # Shield Trigger check (Root keyword)
        st_kw = card.get('keywords', {}).get('shield_trigger', False)
        self.shield_trigger_cb.setChecked(st_kw)

        # Hyper Energy check
        has_hyper = False
        for eff in effects:
            for act in eff.get('actions', []):
                if act.get('type') == 'COST_REFERENCE' and act.get('str_val') == 'HYPER_ENERGY':
                    has_hyper = True
                    break
        self.hyper_energy_cb.setChecked(has_hyper)

        # Revolution Change
        rev_cond = card.get('revolution_change_condition', None)
        if rev_cond:
            self.rev_change_group.setChecked(True)
            self.rev_civ_input.setText(",".join(rev_cond.get('civilizations', [])))
            self.rev_race_input.setText(",".join(rev_cond.get('races', [])))
            self.rev_cost_spin.setValue(rev_cond.get('min_cost', 0))
        else:
            self.rev_change_group.setChecked(False)
            self.rev_civ_input.setText("")
            self.rev_race_input.setText("")
            self.rev_cost_spin.setValue(5)

        # Load Effects List
        self.refresh_effects_list()

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
        self.shield_trigger_cb.blockSignals(block)
        self.hyper_energy_cb.blockSignals(block)
        self.rev_change_group.blockSignals(block)
        self.rev_civ_input.blockSignals(block)
        self.rev_race_input.blockSignals(block)
        self.rev_cost_spin.blockSignals(block)

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

        # Sync Shield Trigger
        if self.shield_trigger_cb.isChecked():
            if 'keywords' not in card: card['keywords'] = {}
            card['keywords']['shield_trigger'] = True
        else:
            if 'keywords' in card and 'shield_trigger' in card['keywords']:
                del card['keywords']['shield_trigger']

        # Sync Hyper Energy
        # 1. Remove existing Hyper Energy actions/effects
        cleaned_effects_for_hyper = []
        for eff in card.get('effects', []):
            new_actions = []
            for act in eff.get('actions', []):
                if not (act.get('type') == 'COST_REFERENCE' and act.get('str_val') == 'HYPER_ENERGY'):
                    new_actions.append(act)

            if new_actions or eff.get('trigger') != 'NONE':
                eff['actions'] = new_actions
                cleaned_effects_for_hyper.append(eff)

        card['effects'] = cleaned_effects_for_hyper

        # 2. Add back if checked
        if self.hyper_energy_cb.isChecked():
            card['effects'].append({
                "trigger": "NONE",
                "condition": {"type": "NONE"},
                "actions": [{
                    "type": "COST_REFERENCE",
                    "str_val": "HYPER_ENERGY",
                    "value1": 0,
                    "scope": "PLAYER_SELF",
                    "filter": {}
                }]
            })

        # Sync Revolution Change
        if self.rev_change_group.isChecked():
            rev_cond = {}
            civs = [c.strip() for c in self.rev_civ_input.text().split(',') if c.strip()]
            if civs: rev_cond['civilizations'] = civs
            races = [r.strip() for r in self.rev_race_input.text().split(',') if r.strip()]
            if races: rev_cond['races'] = races
            min_cost = self.rev_cost_spin.value()
            if min_cost > 0: rev_cond['min_cost'] = min_cost
            card['revolution_change_condition'] = rev_cond
        else:
            if 'revolution_change_condition' in card:
                del card['revolution_change_condition']

        # Sync Keyword Checkboxes to PASSIVE_CONST
        new_effects = []
        existing_effects = card.get('effects', [])

        known_keywords = set(self.keyword_checkboxes.keys())

        for eff in existing_effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                actions_to_keep = []
                for act in eff.get('actions', []):
                    if act.get('str_val') not in known_keywords:
                        actions_to_keep.append(act)
                if actions_to_keep:
                    eff['actions'] = actions_to_keep
                    new_effects.append(eff)
            else:
                new_effects.append(eff)

        active_kws = []
        for kw, cb in self.keyword_checkboxes.items():
            if cb.isChecked():
                active_kws.append(kw)

        if active_kws:
            actions = []
            for kw in active_kws:
                actions.append({
                    "type": "NONE",
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

        item = self.card_list.item(self.current_card_index)
        if item:
            item.setText(f"{card['id']} - {card['name']}")

        if self.tabs.currentWidget() == self.effects_tab:
             self.refresh_effects_list()

    def update_preview(self):
        for i in reversed(range(self.preview_container_layout.count())):
            item = self.preview_container_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        name = self.name_input.text() or "New Card"
        cost = self.cost_input.value()
        power = self.power_input.value()
        civ = self.civ_input.currentText()
        
        self.preview_card = CardWidget(0, name, cost, power, civ)
        self.preview_container_layout.addWidget(self.preview_card)

    # --- Effects Logic ---

    def refresh_effects_list(self):
        if self.current_card_index < 0: return
        self.effects_list.clear()
        effects = self.cards_data[self.current_card_index].get('effects', [])
        for i, eff in enumerate(effects):
            trig = eff.get('trigger', 'NONE')
            action_count = len(eff.get('actions', []))
            self.effects_list.addItem(f"#{i}: {tr(trig)} ({action_count} actions)")

    def create_new_effect(self):
        if self.current_card_index < 0: return
        new_eff = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "actions": []
        }
        self.cards_data[self.current_card_index]['effects'].append(new_eff)
        self.refresh_effects_list()
        self.effects_list.setCurrentRow(self.effects_list.count() - 1)

    def remove_effect(self):
        if self.current_card_index < 0: return
        row = self.effects_list.currentRow()
        if row >= 0:
            del self.cards_data[self.current_card_index]['effects'][row]
            self.refresh_effects_list()

    def load_selected_effect(self, row):
        if row < 0:
            self.eff_trigger_combo.setEnabled(False)
            self.eff_action_list_widget.clear()
            return

        self.eff_trigger_combo.setEnabled(True)
        effect = self.cards_data[self.current_card_index]['effects'][row]

        trig_val = effect.get('trigger', 'ON_PLAY')
        idx = self.eff_trigger_combo.findData(trig_val)
        if idx >= 0: self.eff_trigger_combo.setCurrentIndex(idx)
        else: self.eff_trigger_combo.setCurrentIndex(0)

        self.eff_trigger_combo.currentIndexChanged.disconnect() if self.eff_trigger_combo.receivers(self.eff_trigger_combo.currentIndexChanged) else None
        self.eff_trigger_combo.currentIndexChanged.connect(lambda: self.update_effect_trigger(row))

        # Load Condition
        cond = effect.get('condition', {"type": "NONE", "value": 0, "str_val": ""})

        c_idx = self.eff_cond_type_combo.findData(cond.get('type', 'NONE'))
        if c_idx >= 0: self.eff_cond_type_combo.setCurrentIndex(c_idx)
        else: self.eff_cond_type_combo.setCurrentIndex(0)

        self.eff_cond_val_spin.setValue(cond.get('value', 0))
        self.eff_cond_str_edit.setText(cond.get('str_val', ''))

        # Connect condition signals
        self.eff_cond_type_combo.currentIndexChanged.disconnect() if self.eff_cond_type_combo.receivers(self.eff_cond_type_combo.currentIndexChanged) else None
        self.eff_cond_type_combo.currentIndexChanged.connect(lambda: self.update_effect_condition(row))

        self.eff_cond_val_spin.valueChanged.disconnect() if self.eff_cond_val_spin.receivers(self.eff_cond_val_spin.valueChanged) else None
        self.eff_cond_val_spin.valueChanged.connect(lambda: self.update_effect_condition(row))

        self.eff_cond_str_edit.textChanged.disconnect() if self.eff_cond_str_edit.receivers(self.eff_cond_str_edit.textChanged) else None
        self.eff_cond_str_edit.textChanged.connect(lambda: self.update_effect_condition(row))

        self.eff_action_list_widget.clear()
        for i, act in enumerate(effect.get('actions', [])):
            act_type = act.get('type', 'NONE')
            self.eff_action_list_widget.addItem(f"{i}: {tr(act_type)}")

    def update_effect_trigger(self, row):
        new_trig = self.eff_trigger_combo.currentData()
        self.cards_data[self.current_card_index]['effects'][row]['trigger'] = new_trig
        item = self.effects_list.item(row)
        if item:
            action_count = len(self.cards_data[self.current_card_index]['effects'][row]['actions'])
            item.setText(f"#{row}: {tr(new_trig)} ({action_count} actions)")

    def update_effect_condition(self, row):
        if row < 0 or row >= len(self.cards_data[self.current_card_index]['effects']): return

        cond_type = self.eff_cond_type_combo.currentData()
        val = self.eff_cond_val_spin.value()
        str_val = self.eff_cond_str_edit.text()

        self.cards_data[self.current_card_index]['effects'][row]['condition'] = {
            "type": cond_type,
            "value": val,
            "str_val": str_val
        }

    def add_action_to_effect(self):
        row = self.effects_list.currentRow()
        if row < 0: return

        new_action = {
            "type": "DESTROY",
            "scope": "TARGET_SELECT",
            "value1": 1,
            "filter": {"zones": ["BATTLE_ZONE"], "count": 1}
        }
        self.cards_data[self.current_card_index]['effects'][row]['actions'].append(new_action)
        self.load_selected_effect(row) # Refresh action list
        self.eff_action_list_widget.setCurrentRow(self.eff_action_list_widget.count() - 1)

    def remove_action_from_effect(self):
        eff_row = self.effects_list.currentRow()
        act_row = self.eff_action_list_widget.currentRow()
        if eff_row >= 0 and act_row >= 0:
            del self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row]
            self.load_selected_effect(eff_row)

    def update_dynamic_labels(self):
        act_type = self.act_type_combo.currentData()
        if act_type == "APPLY_MODIFIER":
            self.val1_label.setText(tr("Reduction Amount") + ":")
            self.val2_label.setText(tr("Duration (Turns)") + ":")
        else:
            self.val1_label.setText(tr("Value 1") + ":")
            self.val2_label.setText(tr("Value 2") + ":")

    def load_selected_action(self, act_row):
        eff_row = self.effects_list.currentRow()
        if eff_row < 0 or act_row < 0: return

        action = self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row]

        # Populate form
        idx = self.act_type_combo.findData(action.get('type', 'DESTROY'))
        if idx >= 0: self.act_type_combo.setCurrentIndex(idx)

        # Update dynamic labels based on the loaded action
        self.update_dynamic_labels()

        idx = self.act_scope_combo.findData(action.get('scope', 'TARGET_SELECT'))
        if idx >= 0: self.act_scope_combo.setCurrentIndex(idx)

        # Filter
        filt = action.get('filter', {})
        zones = filt.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        tp = filt.get('owner', 'NONE')
        if tp == 'NONE' or tp is None:
             tp = filt.get('target_player', 'NONE')

        self.filter_player_combo.setCurrentText(tp if isinstance(tp, str) else 'NONE')

        types = filt.get('types', [])
        ftype = 'NONE'
        if 'CREATURE' in types: ftype = 'CREATURE'
        if 'SPELL' in types: ftype = 'SPELL'
        self.filter_type_combo.setCurrentText(ftype)

        self.filter_count_spin.setValue(filt.get('count', 1))

        # New Fields Load
        civs = filt.get('civilizations', [])
        self.filter_civ_input.setText(", ".join(civs))

        races = filt.get('races', [])
        self.filter_race_input.setText(", ".join(races))

        self.filter_min_cost_spin.setValue(filt.get('min_cost', -1) if filt.get('min_cost') is not None else -1)
        self.filter_max_cost_spin.setValue(filt.get('max_cost', -1) if filt.get('max_cost') is not None else -1)
        self.filter_min_power_spin.setValue(filt.get('min_power', -1) if filt.get('min_power') is not None else -1)
        self.filter_max_power_spin.setValue(filt.get('max_power', -1) if filt.get('max_power') is not None else -1)

        # Helper to set combo from optional bool
        def set_bool_combo(combo, key):
            val = filt.get(key)
            if val is None: combo.setCurrentIndex(0) # Ignore
            elif val is True: combo.setCurrentIndex(1) # True
            else: combo.setCurrentIndex(2) # False

        set_bool_combo(self.filter_tapped_combo, 'is_tapped')
        set_bool_combo(self.filter_blocker_combo, 'is_blocker')
        set_bool_combo(self.filter_evolution_combo, 'is_evolution')

        self.val1_spin.setValue(action.get('value1', 0))
        self.val2_spin.setValue(action.get('value2', 0))
        self.str_val_edit.setText(action.get('str_val', ''))
        self.input_key_edit.setText(action.get('input_value_key', ''))
        self.output_key_edit.setText(action.get('output_value_key', ''))

    def apply_action_changes(self):
        eff_row = self.effects_list.currentRow()
        act_row = self.eff_action_list_widget.currentRow()
        if eff_row < 0 or act_row < 0: return

        # Build action object
        new_act = {
            "type": self.act_type_combo.currentData(),
            "scope": self.act_scope_combo.currentData(),
            "value1": self.val1_spin.value(),
            "value2": self.val2_spin.value(),
            "str_val": self.str_val_edit.text(),
            "input_value_key": self.input_key_edit.text(),
            "output_value_key": self.output_key_edit.text()
        }

        # Build filter
        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        target_player = self.filter_player_combo.currentText()
        card_type = self.filter_type_combo.currentText()

        filt = {}
        if zones: filt['zones'] = zones
        if target_player != "NONE": filt['owner'] = target_player
        if card_type != "NONE": filt['types'] = [card_type]
        count = self.filter_count_spin.value()
        if count > 0: filt['count'] = count

        # New Fields Save
        civs_str = self.filter_civ_input.text()
        if civs_str.strip():
            filt['civilizations'] = [c.strip() for c in civs_str.split(',') if c.strip()]

        races_str = self.filter_race_input.text()
        if races_str.strip():
            filt['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        if self.filter_min_cost_spin.value() != -1: filt['min_cost'] = self.filter_min_cost_spin.value()
        if self.filter_max_cost_spin.value() != -1: filt['max_cost'] = self.filter_max_cost_spin.value()
        if self.filter_min_power_spin.value() != -1: filt['min_power'] = self.filter_min_power_spin.value()
        if self.filter_max_power_spin.value() != -1: filt['max_power'] = self.filter_max_power_spin.value()

        def get_bool_from_combo(combo):
            idx = combo.currentIndex()
            if idx == 0: return None
            if idx == 1: return True
            return False

        if (val := get_bool_from_combo(self.filter_tapped_combo)) is not None: filt['is_tapped'] = val
        if (val := get_bool_from_combo(self.filter_blocker_combo)) is not None: filt['is_blocker'] = val
        if (val := get_bool_from_combo(self.filter_evolution_combo)) is not None: filt['is_evolution'] = val

        new_act['filter'] = filt

        self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row] = new_act

        # Refresh list label
        self.eff_action_list_widget.currentItem().setText(f"{act_row}: {tr(new_act['type'])}")
        QMessageBox.information(self, tr("Success"), tr("Action updated!"))
