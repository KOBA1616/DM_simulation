import sys
import os
import json
import logging
from PyQt6.QtCore import Qt, QSize, QModelIndex, QMimeData, QByteArray, QDataStream, QIODevice
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction, QDrag, QIcon, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QTreeView, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QGroupBox, QComboBox, QLineEdit, QSpinBox,
    QFormLayout, QLabel, QPushButton, QMenu, QMessageBox, QAbstractItemView,
    QScrollArea, QGridLayout, QCheckBox, QApplication
)
from gui.localization import tr

# Constants
ROLE_NODE_TYPE = Qt.ItemDataRole.UserRole + 1
ROLE_DATA_REF = Qt.ItemDataRole.UserRole + 2 # We will store the data dict here

NODE_ROOT = 0
NODE_CARD = 1
NODE_KEYWORDS = 2
NODE_EFFECT = 3
NODE_ACTION = 4

class CardTreeModel(QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Hierarchy"])

    def flags(self, index):
        default_flags = super().flags(index)
        if index.isValid():
            return default_flags | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        return default_flags | Qt.ItemFlag.ItemIsDropEnabled

    def canDropMimeData(self, data, action, row, column, parent_index):
        # Allow dropping generally, relying on visual cues and subsequent validation if needed.
        # For InternalMove in TreeView, usually it just works for reordering.
        return super().canDropMimeData(data, action, row, column, parent_index)

class CardEditor(QMainWindow):
    def __init__(self, json_path="data/cards.json", parent=None):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle(tr("Card Editor"))
        self.resize(1300, 800)

        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left: Tree
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.tree_view = QTreeView()
        self.model = CardTreeModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_context_menu)
        self.tree_view.selectionModel().selectionChanged.connect(self.on_selection_changed)

        left_layout.addWidget(self.tree_view)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_card_btn = QPushButton(tr("New Card"))
        self.add_card_btn.clicked.connect(self.add_new_card)
        self.save_btn = QPushButton(tr("Save"))
        self.save_btn.clicked.connect(self.save_data)
        btn_layout.addWidget(self.add_card_btn)
        btn_layout.addWidget(self.save_btn)
        left_layout.addLayout(btn_layout)
        
        self.splitter.addWidget(left_widget)

        # Right: Property Inspector
        self.inspector_container = QWidget()
        self.inspector_layout = QVBoxLayout(self.inspector_container)
        self.stacked_widget = QStackedWidget()

        # 0: Empty
        self.page_empty = QLabel("Select an item to edit")
        self.page_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked_widget.addWidget(self.page_empty)

        # 1: Card Editor
        self.page_card = QWidget()
        self.setup_card_ui(self.page_card)
        self.stacked_widget.addWidget(self.page_card)

        # 2: Keywords Editor
        self.page_keywords = QWidget()
        self.setup_keywords_ui(self.page_keywords)
        self.stacked_widget.addWidget(self.page_keywords)

        # 3: Effect Editor
        self.page_effect = QWidget()
        self.setup_effect_ui(self.page_effect)
        self.stacked_widget.addWidget(self.page_effect)

        # 4: Action Editor
        self.page_action = QWidget()
        self.setup_action_ui(self.page_action)
        self.stacked_widget.addWidget(self.page_action)
        
        self.inspector_layout.addWidget(self.stacked_widget)
        self.splitter.addWidget(self.inspector_container)

        self.splitter.setSizes([400, 800])

    def setup_card_ui(self, parent):
        layout = QVBoxLayout(parent)
        form = QFormLayout()

        self.card_id_edit = QSpinBox()
        self.card_id_edit.setRange(1, 99999)
        self.card_id_edit.valueChanged.connect(self.on_card_data_changed)
        form.addRow(tr("ID"), self.card_id_edit)

        self.card_name_edit = QLineEdit()
        self.card_name_edit.textChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Name"), self.card_name_edit)

        self.card_civ_combo = QComboBox()
        self.card_civ_combo.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        self.card_civ_combo.currentTextChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Civilization"), self.card_civ_combo)

        self.card_type_combo = QComboBox()
        self.card_type_combo.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
        self.card_type_combo.currentTextChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Type"), self.card_type_combo)

        self.card_cost_spin = QSpinBox()
        self.card_cost_spin.setRange(0, 99)
        self.card_cost_spin.valueChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Cost"), self.card_cost_spin)

        self.card_power_spin = QSpinBox()
        self.card_power_spin.setRange(0, 99999)
        self.card_power_spin.setSingleStep(500)
        self.card_power_spin.valueChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Power"), self.card_power_spin)

        self.card_races_edit = QLineEdit()
        self.card_races_edit.setPlaceholderText("Race1, Race2")
        self.card_races_edit.textChanged.connect(self.on_card_data_changed)
        form.addRow(tr("Races"), self.card_races_edit)

        layout.addLayout(form)
        layout.addStretch()

    def setup_keywords_ui(self, parent):
        layout = QVBoxLayout(parent)
        self.keyword_checkboxes = {}
        keywords = [
            "BLOCKER", "SPEED_ATTACKER", "SLAYER", "DOUBLE_BREAKER",
            "TRIPLE_BREAKER", "POWER_ATTACKER", "EVOLUTION",
            "MACH_FIGHTER", "G_STRIKE", "JUST_DIVER"
        ]
        grid = QGridLayout()
        for i, kw in enumerate(keywords):
            cb = QCheckBox(tr(kw))
            cb.stateChanged.connect(self.on_keywords_changed)
            self.keyword_checkboxes[kw] = cb
            grid.addWidget(cb, i // 2, i % 2)

        layout.addLayout(grid)

        # Shield Trigger
        self.st_check = QCheckBox(tr("S_TRIGGER"))
        self.st_check.stateChanged.connect(self.on_keywords_changed)
        layout.addWidget(self.st_check)

        # Revolution Change
        self.rev_group = QGroupBox(tr("REVOLUTION_CHANGE"))
        self.rev_group.setCheckable(True)
        self.rev_group.toggled.connect(self.on_keywords_changed)
        rev_form = QFormLayout(self.rev_group)
        self.rev_civ_edit = QLineEdit()
        self.rev_civ_edit.textChanged.connect(self.on_keywords_changed)
        rev_form.addRow(tr("Civilizations"), self.rev_civ_edit)
        self.rev_race_edit = QLineEdit()
        self.rev_race_edit.textChanged.connect(self.on_keywords_changed)
        rev_form.addRow(tr("Races"), self.rev_race_edit)
        self.rev_cost_spin = QSpinBox()
        self.rev_cost_spin.setRange(0, 99)
        self.rev_cost_spin.valueChanged.connect(self.on_keywords_changed)
        rev_form.addRow(tr("Min Cost"), self.rev_cost_spin)
        layout.addWidget(self.rev_group)

        layout.addStretch()

    def setup_effect_ui(self, parent):
        layout = QVBoxLayout(parent)
        form = QFormLayout()

        self.eff_trigger_combo = QComboBox()
        triggers = ["ON_PLAY", "ON_ATTACK", "ON_DESTROY", "PASSIVE_CONST", "TURN_START", "ON_OTHER_ENTER", "ON_ATTACK_FROM_HAND"]
        for t in triggers:
            self.eff_trigger_combo.addItem(tr(t), t)
        self.eff_trigger_combo.currentIndexChanged.connect(self.on_effect_changed)
        form.addRow(tr("Trigger"), self.eff_trigger_combo)

        # Simple condition editor
        self.eff_cond_type = QComboBox()
        conds = ["NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH", "OPPONENT_PLAYED_WITHOUT_MANA"]
        for c in conds:
            self.eff_cond_type.addItem(c, c)
        self.eff_cond_type.currentIndexChanged.connect(self.on_effect_changed)
        form.addRow(tr("Condition Type"), self.eff_cond_type)

        self.eff_cond_val = QSpinBox()
        self.eff_cond_val.valueChanged.connect(self.on_effect_changed)
        form.addRow(tr("Value"), self.eff_cond_val)

        layout.addLayout(form)
        layout.addStretch()

    def setup_action_ui(self, parent):
        layout = QVBoxLayout(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)

        self.act_type_combo = QComboBox()
        actions = [
            "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP",
            "COST_REFERENCE", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
            "REVOLUTION_CHANGE", "COUNT_CARDS", "GET_GAME_STAT", "APPLY_MODIFIER", "REVEAL_CARDS",
            "REGISTER_DELAYED_EFFECT", "RESET_INSTANCE"
        ]
        actions.sort()
        for a in actions:
            self.act_type_combo.addItem(tr(a), a)
        self.act_type_combo.currentIndexChanged.connect(self.on_action_changed)
        form.addRow(tr("Action Type"), self.act_type_combo)

        # Variable Linking
        self.act_input_key = QComboBox() # Dynamic
        self.act_input_key.setEditable(True)
        self.act_input_key.editTextChanged.connect(self.on_action_changed) # Capture manual edits too
        form.addRow(tr("Input Key"), self.act_input_key)

        self.act_output_key = QLineEdit()
        self.act_output_key.textChanged.connect(self.on_action_changed)
        form.addRow(tr("Output Key"), self.act_output_key)

        # Other fields (Scope, Value1, Value2, String, Filter...)
        self.act_scope = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"]
        for s in scopes:
            self.act_scope.addItem(tr(s), s)
        self.act_scope.currentIndexChanged.connect(self.on_action_changed)
        form.addRow(tr("Scope"), self.act_scope)

        self.act_val1 = QSpinBox()
        self.act_val1.setRange(-1, 9999)
        self.act_val1.valueChanged.connect(self.on_action_changed)
        form.addRow(tr("Value 1"), self.act_val1)

        self.act_str = QLineEdit()
        self.act_str.textChanged.connect(self.on_action_changed)
        form.addRow(tr("String Value"), self.act_str)

        # Filter (Zones, Civ, Race, Cost, Power, Flags)
        self.act_filter_zones = QLineEdit()
        self.act_filter_zones.setPlaceholderText("BATTLE_ZONE, HAND etc")
        self.act_filter_zones.textChanged.connect(self.on_action_changed)
        form.addRow(tr("Filter Zones"), self.act_filter_zones)

        self.act_filter_civ = QLineEdit()
        self.act_filter_civ.textChanged.connect(self.on_action_changed)
        form.addRow(tr("Filter Civ"), self.act_filter_civ)

        self.act_filter_race = QLineEdit()
        self.act_filter_race.textChanged.connect(self.on_action_changed)
        form.addRow(tr("Filter Race"), self.act_filter_race)

        # Cost Range
        cost_layout = QHBoxLayout()
        self.act_filter_min_cost = QSpinBox()
        self.act_filter_min_cost.setRange(-1, 99)
        self.act_filter_min_cost.setValue(-1)
        self.act_filter_min_cost.valueChanged.connect(self.on_action_changed)
        cost_layout.addWidget(QLabel(tr("Min") + ":"))
        cost_layout.addWidget(self.act_filter_min_cost)

        self.act_filter_max_cost = QSpinBox()
        self.act_filter_max_cost.setRange(-1, 99)
        self.act_filter_max_cost.setValue(-1)
        self.act_filter_max_cost.valueChanged.connect(self.on_action_changed)
        cost_layout.addWidget(QLabel(tr("Max") + ":"))
        cost_layout.addWidget(self.act_filter_max_cost)
        form.addRow(tr("Filter Cost"), cost_layout)

        # Power Range
        power_layout = QHBoxLayout()
        self.act_filter_min_power = QSpinBox()
        self.act_filter_min_power.setRange(-1, 99999)
        self.act_filter_min_power.setSingleStep(500)
        self.act_filter_min_power.setValue(-1)
        self.act_filter_min_power.valueChanged.connect(self.on_action_changed)
        power_layout.addWidget(QLabel(tr("Min") + ":"))
        power_layout.addWidget(self.act_filter_min_power)

        self.act_filter_max_power = QSpinBox()
        self.act_filter_max_power.setRange(-1, 99999)
        self.act_filter_max_power.setSingleStep(500)
        self.act_filter_max_power.setValue(-1)
        self.act_filter_max_power.valueChanged.connect(self.on_action_changed)
        power_layout.addWidget(QLabel(tr("Max") + ":"))
        power_layout.addWidget(self.act_filter_max_power)
        form.addRow(tr("Filter Power"), power_layout)

        # Flags
        self.act_filter_tapped = QComboBox()
        self.act_filter_tapped.addItems([tr("Ignore"), tr("True"), tr("False")])
        self.act_filter_tapped.currentIndexChanged.connect(self.on_action_changed)
        form.addRow(tr("Tapped"), self.act_filter_tapped)

        self.act_filter_blocker = QComboBox()
        self.act_filter_blocker.addItems([tr("Ignore"), tr("True"), tr("False")])
        self.act_filter_blocker.currentIndexChanged.connect(self.on_action_changed)
        form.addRow(tr("Blocker"), self.act_filter_blocker)
        
        self.act_filter_evolution = QComboBox()
        self.act_filter_evolution.addItems([tr("Ignore"), tr("True"), tr("False")])
        self.act_filter_evolution.currentIndexChanged.connect(self.on_action_changed)
        form.addRow(tr("Evolution"), self.act_filter_evolution)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    # --- Data Loading ---

    def load_data(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Cards"])
        root = self.model.invisibleRootItem()

        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for card in data:
                        self.add_card_to_tree(root, card)
            except Exception as e:
                print(f"Error loading: {e}")

    def add_card_to_tree(self, root, card_data):
        name = f"{card_data.get('id', '?')} - {card_data.get('name', 'No Name')}"
        item = QStandardItem(name)
        item.setData(NODE_CARD, ROLE_NODE_TYPE)
        item.setData(card_data, ROLE_DATA_REF)
        root.appendRow(item)

        # Keywords Node
        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData(NODE_KEYWORDS, ROLE_NODE_TYPE)
        kw_item.setData(card_data, ROLE_DATA_REF)
        item.appendRow(kw_item)

        # Effects
        for eff in card_data.get('effects', []):
            if eff.get('trigger') == 'PASSIVE_CONST' and self.is_keyword_effect(eff):
                continue
            self.add_effect_to_tree(item, eff)

    def is_keyword_effect(self, eff):
        if eff.get('trigger') != 'PASSIVE_CONST': return False
        for act in eff.get('actions', []):
            if act.get('str_val') in ["BLOCKER", "SPEED_ATTACKER", "SLAYER", "DOUBLE_BREAKER", "TRIPLE_BREAKER", "POWER_ATTACKER", "EVOLUTION", "MACH_FIGHTER", "G_STRIKE", "JUST_DIVER"]:
                return True
        return False

    def add_effect_to_tree(self, card_item, eff_data):
        trig = eff_data.get('trigger', 'NONE')
        item = QStandardItem(tr(trig))
        item.setData(NODE_EFFECT, ROLE_NODE_TYPE)
        item.setData(eff_data, ROLE_DATA_REF)
        card_item.appendRow(item)
        
        for act in eff_data.get('actions', []):
            self.add_action_to_tree(item, act)

    def add_action_to_tree(self, eff_item, act_data):
        atype = act_data.get('type', 'NONE')
        item = QStandardItem(tr(atype))
        item.setData(NODE_ACTION, ROLE_NODE_TYPE)
        item.setData(act_data, ROLE_DATA_REF)
        eff_item.appendRow(item)

    # --- Selection & UI Update ---

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            self.stacked_widget.setCurrentWidget(self.page_empty)
            return
        
        index = indexes[0]
        item = self.model.itemFromIndex(index)
        node_type = item.data(ROLE_NODE_TYPE)
        data = item.data(ROLE_DATA_REF)
        
        self.block_signals(True)
        
        if node_type == NODE_CARD:
            self.stacked_widget.setCurrentWidget(self.page_card)
            self.populate_card_ui(data)
        elif node_type == NODE_KEYWORDS:
            self.stacked_widget.setCurrentWidget(self.page_keywords)
            self.populate_keywords_ui(data)
        elif node_type == NODE_EFFECT:
            self.stacked_widget.setCurrentWidget(self.page_effect)
            self.populate_effect_ui(data)
        elif node_type == NODE_ACTION:
            self.stacked_widget.setCurrentWidget(self.page_action)
            self.populate_action_ui(data)
            self.update_variable_suggestions(item)
        else:
            self.stacked_widget.setCurrentWidget(self.page_empty)

        self.block_signals(False)

    def block_signals(self, block):
        inputs = [
            self.card_id_edit, self.card_name_edit, self.card_civ_combo, self.card_type_combo,
            self.card_cost_spin, self.card_power_spin, self.card_races_edit,
            self.st_check, self.rev_group, self.rev_civ_edit, self.rev_race_edit, self.rev_cost_spin,
            self.eff_trigger_combo, self.eff_cond_type, self.eff_cond_val,
            self.act_type_combo, self.act_input_key, self.act_output_key, self.act_scope, self.act_val1,
            self.act_str, self.act_filter_zones, self.act_filter_civ, self.act_filter_race,
            self.act_filter_min_cost, self.act_filter_max_cost,
            self.act_filter_min_power, self.act_filter_max_power,
            self.act_filter_tapped, self.act_filter_blocker, self.act_filter_evolution
        ]
        for w in inputs: w.blockSignals(block)
        for w in self.keyword_checkboxes.values(): w.blockSignals(block)

    def populate_card_ui(self, data):
        self.card_id_edit.setValue(data.get('id', 0))
        self.card_name_edit.setText(data.get('name', ''))
        self.card_civ_combo.setCurrentText(data.get('civilization', 'FIRE'))
        self.card_type_combo.setCurrentText(data.get('type', 'CREATURE'))
        self.card_cost_spin.setValue(data.get('cost', 0))
        self.card_power_spin.setValue(data.get('power', 0))
        self.card_races_edit.setText(", ".join(data.get('races', [])))

    def populate_keywords_ui(self, data):
        active = set()
        for eff in data.get('effects', []):
            if eff.get('trigger') == 'PASSIVE_CONST':
                for act in eff.get('actions', []):
                    if 'str_val' in act:
                        active.add(act['str_val'])

        kws = data.get('keywords', {})
        if kws.get('shield_trigger'): self.st_check.setChecked(True)
        else: self.st_check.setChecked(False)
        
        for kw, cb in self.keyword_checkboxes.items():
            cb.setChecked(kw in active)

        rc = data.get('revolution_change_condition')
        if rc:
            self.rev_group.setChecked(True)
            self.rev_civ_edit.setText(", ".join(rc.get('civilizations', [])))
            self.rev_race_edit.setText(", ".join(rc.get('races', [])))
            self.rev_cost_spin.setValue(rc.get('min_cost', 0))
        else:
            self.rev_group.setChecked(False)

    def populate_effect_ui(self, data):
        idx = self.eff_trigger_combo.findData(data.get('trigger', 'NONE'))
        if idx >= 0: self.eff_trigger_combo.setCurrentIndex(idx)

        cond = data.get('condition', {})
        cidx = self.eff_cond_type.findData(cond.get('type', 'NONE'))
        if cidx >= 0: self.eff_cond_type.setCurrentIndex(cidx)
        self.eff_cond_val.setValue(cond.get('value', 0))

    def populate_action_ui(self, data):
        idx = self.act_type_combo.findData(data.get('type', 'NONE'))
        if idx >= 0: self.act_type_combo.setCurrentIndex(idx)

        self.act_input_key.setEditText(data.get('input_value_key', ''))
        self.act_output_key.setText(data.get('output_value_key', ''))

        sidx = self.act_scope.findData(data.get('scope', 'NONE'))
        if sidx >= 0: self.act_scope.setCurrentIndex(sidx)

        self.act_val1.setValue(data.get('value1', 0))
        self.act_str.setText(data.get('str_val', ''))

        filt = data.get('filter', {})
        self.act_filter_zones.setText(", ".join(filt.get('zones', [])))
        self.act_filter_civ.setText(", ".join(filt.get('civilizations', [])))
        self.act_filter_race.setText(", ".join(filt.get('races', [])))

        self.act_filter_min_cost.setValue(filt.get('min_cost', -1) if filt.get('min_cost') is not None else -1)
        self.act_filter_max_cost.setValue(filt.get('max_cost', -1) if filt.get('max_cost') is not None else -1)
        self.act_filter_min_power.setValue(filt.get('min_power', -1) if filt.get('min_power') is not None else -1)
        self.act_filter_max_power.setValue(filt.get('max_power', -1) if filt.get('max_power') is not None else -1)

        def set_bool_combo(combo, key):
            val = filt.get(key)
            if val is None: combo.setCurrentIndex(0)
            elif val is True: combo.setCurrentIndex(1)
            else: combo.setCurrentIndex(2)

        set_bool_combo(self.act_filter_tapped, 'is_tapped')
        set_bool_combo(self.act_filter_blocker, 'is_blocker')
        set_bool_combo(self.act_filter_evolution, 'is_evolution')

    # --- Data Saving (Sync UI to Data Ref) ---

    def on_card_data_changed(self):
        item = self.get_selected_item()
        if not item or item.data(ROLE_NODE_TYPE) != NODE_CARD: return
        data = item.data(ROLE_DATA_REF)

        data['id'] = self.card_id_edit.value()
        data['name'] = self.card_name_edit.text()
        data['civilization'] = self.card_civ_combo.currentText()
        data['type'] = self.card_type_combo.currentText()
        data['cost'] = self.card_cost_spin.value()
        data['power'] = self.card_power_spin.value()
        races = self.card_races_edit.text().split(',')
        data['races'] = [r.strip() for r in races if r.strip()]

        item.setText(f"{data['id']} - {data['name']}")

    def on_keywords_changed(self):
        item = self.get_selected_item()
        if not item or item.data(ROLE_NODE_TYPE) != NODE_KEYWORDS: return
        data = item.data(ROLE_DATA_REF)

        if self.st_check.isChecked():
            if 'keywords' not in data: data['keywords'] = {}
            data['keywords']['shield_trigger'] = True
        else:
            if 'keywords' in data and 'shield_trigger' in data['keywords']:
                del data['keywords']['shield_trigger']

        if self.rev_group.isChecked():
            rc = {}
            civs = [c.strip() for c in self.rev_civ_edit.text().split(',') if c.strip()]
            if civs: rc['civilizations'] = civs
            races = [r.strip() for r in self.rev_race_edit.text().split(',') if r.strip()]
            if races: rc['races'] = races
            if self.rev_cost_spin.value() > 0: rc['min_cost'] = self.rev_cost_spin.value()
            data['revolution_change_condition'] = rc
        else:
            if 'revolution_change_condition' in data:
                del data['revolution_change_condition']

        new_effects = []
        for eff in data.get('effects', []):
            if not self.is_keyword_effect(eff):
                new_effects.append(eff)

        active_kws = []
        for kw, cb in self.keyword_checkboxes.items():
            if cb.isChecked():
                active_kws.append(kw)

        if active_kws:
            actions = [{"type": "NONE", "scope": "NONE", "str_val": kw, "value1": 0} for kw in active_kws]
            new_effects.append({
                "trigger": "PASSIVE_CONST",
                "condition": {"type": "NONE"},
                "actions": actions
            })

        data['effects'] = new_effects

    def on_effect_changed(self):
        item = self.get_selected_item()
        if not item or item.data(ROLE_NODE_TYPE) != NODE_EFFECT: return
        data = item.data(ROLE_DATA_REF)

        data['trigger'] = self.eff_trigger_combo.currentData()

        cond_type = self.eff_cond_type.currentText()
        val = self.eff_cond_val.value()
        data['condition'] = {"type": cond_type, "value": val}

        item.setText(tr(data['trigger']))

    def on_action_changed(self):
        item = self.get_selected_item()
        if not item or item.data(ROLE_NODE_TYPE) != NODE_ACTION: return
        data = item.data(ROLE_DATA_REF)

        data['type'] = self.act_type_combo.currentData()
        data['input_value_key'] = self.act_input_key.currentText()
        data['output_value_key'] = self.act_output_key.text()
        data['scope'] = self.act_scope.currentData()
        data['value1'] = self.act_val1.value()
        data['str_val'] = self.act_str.text()

        filt = data.get('filter', {})
        zones = [z.strip() for z in self.act_filter_zones.text().split(',') if z.strip()]
        if zones: filt['zones'] = zones
        else: filt.pop('zones', None)

        civs = [c.strip() for c in self.act_filter_civ.text().split(',') if c.strip()]
        if civs: filt['civilizations'] = civs
        else: filt.pop('civilizations', None)

        races = [r.strip() for r in self.act_filter_race.text().split(',') if r.strip()]
        if races: filt['races'] = races
        else: filt.pop('races', None)

        if self.act_filter_min_cost.value() != -1: filt['min_cost'] = self.act_filter_min_cost.value()
        else: filt.pop('min_cost', None)

        if self.act_filter_max_cost.value() != -1: filt['max_cost'] = self.act_filter_max_cost.value()
        else: filt.pop('max_cost', None)

        if self.act_filter_min_power.value() != -1: filt['min_power'] = self.act_filter_min_power.value()
        else: filt.pop('min_power', None)

        if self.act_filter_max_power.value() != -1: filt['max_power'] = self.act_filter_max_power.value()
        else: filt.pop('max_power', None)

        def get_bool_from_combo(combo):
            idx = combo.currentIndex()
            if idx == 0: return None
            if idx == 1: return True
            return False

        if (val := get_bool_from_combo(self.act_filter_tapped)) is not None: filt['is_tapped'] = val
        else: filt.pop('is_tapped', None)
        
        if (val := get_bool_from_combo(self.act_filter_blocker)) is not None: filt['is_blocker'] = val
        else: filt.pop('is_blocker', None)
        
        if (val := get_bool_from_combo(self.act_filter_evolution)) is not None: filt['is_evolution'] = val
        else: filt.pop('is_evolution', None)

        data['filter'] = filt

        item.setText(tr(data['type']))

    def get_selected_item(self):
        idx = self.tree_view.currentIndex()
        if not idx.isValid(): return None
        return self.model.itemFromIndex(idx)

    # --- Variable Linking Logic ---

    def update_variable_suggestions(self, current_action_item):
        self.act_input_key.clear()
        self.act_input_key.addItem("")

        effect_item = current_action_item.parent()
        if not effect_item: return

        current_row = current_action_item.row()

        for i in range(current_row):
            sibling = effect_item.child(i)
            if not sibling: continue

            act_data = sibling.data(ROLE_DATA_REF)
            out_key = act_data.get('output_value_key')
            act_type = act_data.get('type', '?')

            if out_key:
                label = f"{out_key} (from #{i} {tr(act_type)})"
                self.act_input_key.addItem(label, out_key)

        current_val = current_action_item.data(ROLE_DATA_REF).get('input_value_key', '')
        self.act_input_key.setEditText(current_val)

    # --- Context Menu & Structure Edit ---

    def open_context_menu(self, position):
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            menu = QMenu()
            menu.addAction(tr("New Card"), self.add_new_card)
            menu.exec(self.tree_view.viewport().mapToGlobal(position))
            return

        item = self.model.itemFromIndex(index)
        node_type = item.data(ROLE_NODE_TYPE)

        menu = QMenu()

        if node_type == NODE_CARD:
            menu.addAction(tr("Add Effect"), lambda: self.add_new_effect(item))
            menu.addAction(tr("Delete Card"), lambda: self.delete_item(index))
        elif node_type == NODE_EFFECT:
            menu.addAction(tr("Add Action"), lambda: self.add_new_action(item))
            menu.addAction(tr("Delete Effect"), lambda: self.delete_item(index))
        elif node_type == NODE_ACTION:
            menu.addAction(tr("Delete Action"), lambda: self.delete_item(index))

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def add_new_card(self):
        root = self.model.invisibleRootItem()
        max_id = 0
        for i in range(root.rowCount()):
            item = root.child(i)
            if item.data(ROLE_NODE_TYPE) == NODE_CARD:
                cid = item.data(ROLE_DATA_REF).get('id', 0)
                if cid > max_id: max_id = cid

        new_data = {"id": max_id + 1, "name": "New Card", "effects": []}
        self.add_card_to_tree(root, new_data)

    def add_new_effect(self, card_item):
        card_data = card_item.data(ROLE_DATA_REF)
        new_eff = {"trigger": "ON_PLAY", "actions": []}
        card_data.setdefault('effects', []).append(new_eff)
        self.add_effect_to_tree(card_item, new_eff)

    def add_new_action(self, effect_item):
        eff_data = effect_item.data(ROLE_DATA_REF)
        new_act = {"type": "DESTROY"}
        eff_data.setdefault('actions', []).append(new_act)
        self.add_action_to_tree(effect_item, new_act)

    def delete_item(self, index):
        item = self.model.itemFromIndex(index)
        parent_item = item.parent() or self.model.invisibleRootItem()
        parent_data = parent_item.data(ROLE_DATA_REF)
        item_data = item.data(ROLE_DATA_REF)
        node_type = item.data(ROLE_NODE_TYPE)

        if node_type == NODE_EFFECT:
            if item_data in parent_data.get('effects', []):
                parent_data['effects'].remove(item_data)

        elif node_type == NODE_ACTION:
            if item_data in parent_data.get('actions', []):
                parent_data['actions'].remove(item_data)

        self.model.removeRow(index.row(), index.parent())

    def save_data(self):
        cards = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
            if card_item.data(ROLE_NODE_TYPE) != NODE_CARD: continue

            c_data = card_item.data(ROLE_DATA_REF)

            new_effects = []
            for eff in c_data.get('effects', []):
                 if self.is_keyword_effect(eff):
                     new_effects.append(eff)

            for j in range(card_item.rowCount()):
                eff_item = card_item.child(j)
                if eff_item.data(ROLE_NODE_TYPE) == NODE_EFFECT:
                    e_data = eff_item.data(ROLE_DATA_REF)
                    new_actions = []
                    for k in range(eff_item.rowCount()):
                        act_item = eff_item.child(k)
                        if act_item.data(ROLE_NODE_TYPE) == NODE_ACTION:
                            new_actions.append(act_item.data(ROLE_DATA_REF))

                    e_data['actions'] = new_actions
                    new_effects.append(e_data)

            c_data['effects'] = new_effects
            cards.append(c_data)

        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(cards, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, tr("Success"), tr("Cards saved successfully!"))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"{tr('Failed to save JSON')}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardEditor()
    window.show()
    sys.exit(app.exec())
