# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QLabel, QGroupBox, QVBoxLayout, QScrollArea,
    QPushButton, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector

class CardEditForm(BaseEditForm):
    # Signal to request structural changes in the Logic Tree
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.form_layout = QFormLayout(self.scroll_content)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        # ID (Hidden from UI as per requirement)
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        self.id_spin.setVisible(False)
        # self.form_layout.addRow(tr("ID"), self.id_spin)

        # Name
        self.name_edit = QLineEdit()
        self.form_layout.addRow(tr("Name"), self.name_edit)

        # Twinpact Checkbox
        self.twinpact_check = QCheckBox(tr("Is Twinpact?"))
        self.twinpact_check.setToolTip(tr("Enable to generate a Spell Side node in the logic tree."))
        self.twinpact_check.stateChanged.connect(self.toggle_twinpact)
        self.form_layout.addRow(tr("Twinpact"), self.twinpact_check)

        # Civilization
        self.civ_selector = CivilizationSelector()
        self.civ_selector.changed.connect(self.update_data)
        self.form_layout.addRow(tr("Civilization"), self.civ_selector)

        # Type
        self.type_combo = QComboBox()
        types = ["CREATURE", "SPELL", "EVOLUTION_CREATURE", "TAMASEED", "CROSS_GEAR", "CASTLE", "PSYCHIC_CREATURE", "GR_CREATURE", "NEO_CREATURE"]
        self.populate_combo(self.type_combo, types, data_func=lambda x: x)
        self.form_layout.addRow(tr("Type"), self.type_combo)

        # Cost
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.form_layout.addRow(tr("Cost"), self.cost_spin)

        # Power
        self.power_spin = QSpinBox()
        self.power_spin.setRange(0, 99999)
        self.power_spin.setSingleStep(500)
        self.lbl_power = QLabel(tr("Power"))
        self.form_layout.addRow(self.lbl_power, self.power_spin)

        # Races
        self.races_edit = QLineEdit()
        self.lbl_races = QLabel(tr("Races"))
        self.form_layout.addRow(self.lbl_races, self.races_edit)

        # Evolution Condition (Hidden by default, shown for Evolution types)
        self.evolution_condition_edit = QLineEdit()
        self.evolution_condition_edit.setPlaceholderText(tr("e.g. Fire Bird"))
        self.lbl_evolution_condition = QLabel(tr("Evolution Condition"))
        self.form_layout.addRow(self.lbl_evolution_condition, self.evolution_condition_edit)

        # AI Configuration Section
        ai_group = QGroupBox(tr("AI Configuration"))
        ai_layout = QFormLayout(ai_group)

        self.is_key_card_check = QCheckBox(tr("Is Key Card / Combo Piece"))
        self.is_key_card_check.stateChanged.connect(self.update_data)
        ai_layout.addRow(self.is_key_card_check)

        self.ai_importance_spin = QSpinBox()
        self.ai_importance_spin.setRange(0, 1000)
        self.ai_importance_spin.valueChanged.connect(self.update_data)
        ai_layout.addRow(tr("AI Importance Score"), self.ai_importance_spin)

        self.form_layout.addRow(ai_group)

        # Actions Section
        actions_group = QGroupBox(tr("Actions"))
        actions_layout = QVBoxLayout(actions_group)

        self.add_effect_btn = QPushButton(tr("Add Effect"))
        self.add_effect_btn.clicked.connect(self.on_add_effect_clicked)
        actions_layout.addWidget(self.add_effect_btn)

        self.form_layout.addRow(actions_group)

        # Define bindings
        self.bindings = {
            'id': self.id_spin,
            'name': self.name_edit,
            'type': self.type_combo,
            'cost': self.cost_spin,
            'power': self.power_spin,
            'evolution_condition': self.evolution_condition_edit,
            'is_key_card': self.is_key_card_check,
            'ai_importance_score': self.ai_importance_spin
            # 'civilizations': self.civ_selector, # Custom widget, handled manually or via duck typing set_data/get_data?
            # CivilizationSelector has set_selected_civs and get_selected_civs. Not standard.
            # 'races': handled manually (list vs string)
        }

        # Connect signals
        self.id_spin.valueChanged.connect(self.update_data)
        self.name_edit.textChanged.connect(self.update_data)
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.power_spin.valueChanged.connect(self.update_data)
        self.races_edit.textChanged.connect(self.update_data)
        self.evolution_condition_edit.textChanged.connect(self.update_data)

        # Initialize visibility
        self.update_type_visibility(self.type_combo.currentData())

    def on_add_effect_clicked(self):
        menu = QMenu(self)

        kw_act = menu.addAction(tr("Keyword Ability"))
        kw_act.triggered.connect(lambda: self.structure_update_requested.emit("ADD_CHILD_EFFECT", {"type": "KEYWORDS"}))

        trig_act = menu.addAction(tr("Triggered Ability"))
        trig_act.triggered.connect(lambda: self.structure_update_requested.emit("ADD_CHILD_EFFECT", {"type": "TRIGGERED"}))

        static_act = menu.addAction(tr("Static Ability"))
        static_act.triggered.connect(lambda: self.structure_update_requested.emit("ADD_CHILD_EFFECT", {"type": "STATIC"}))

        react_act = menu.addAction(tr("Reaction Ability"))
        react_act.triggered.connect(lambda: self.structure_update_requested.emit("ADD_CHILD_EFFECT", {"type": "REACTION"}))

        menu.exec(QCursor.pos())

    def toggle_twinpact(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit("ADD_SPELL_SIDE", {})
        else:
            self.structure_update_requested.emit("REMOVE_SPELL_SIDE", {})

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        # Apply standard bindings
        self._apply_bindings(data)

        # Custom population
        civs = data.get('civilizations')
        if not civs:
            civ_single = data.get('civilization')
            if civ_single: civs = [civ_single]
        self.civ_selector.set_selected_civs(civs)

        current_type = data.get('type', 'CREATURE')
        self.update_type_visibility(current_type)

        self.races_edit.setText(", ".join(data.get('races', [])))

        # Check for Spell Side child node to toggle Twinpact checkbox
        has_spell_side = False
        for i in range(item.rowCount()):
            child = item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                has_spell_side = True
                break

        self.twinpact_check.blockSignals(True)
        self.twinpact_check.setChecked(has_spell_side)
        self.twinpact_check.blockSignals(False)


    def update_type_visibility(self, type_str):
        # Hide Power if Spell
        is_spell = (type_str == "SPELL")
        self.power_spin.setVisible(not is_spell)
        self.lbl_power.setVisible(not is_spell)

        # Evolution Condition
        is_evolution = (type_str == "EVOLUTION_CREATURE" or type_str == "NEO_CREATURE")
        self.evolution_condition_edit.setVisible(is_evolution)
        self.evolution_condition_edit.setEnabled(is_evolution)
        self.lbl_evolution_condition.setVisible(is_evolution)

    def _save_data(self, data):
        # Apply bindings (collects into data)
        self._collect_bindings(data)

        # Custom saving
        data['civilizations'] = self.civ_selector.get_selected_civs()
        if 'civilization' in data:
            del data['civilization']

        type_str = self.type_combo.currentData()
        self.update_type_visibility(type_str)

        # Force power to 0 if Spell, regardless of hidden spinner value
        if type_str == "SPELL":
            data['power'] = 0

        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        # Save evolution condition if applicable
        if not (type_str == "EVOLUTION_CREATURE" or type_str == "NEO_CREATURE"):
             if 'evolution_condition' in data:
                 del data['evolution_condition']

        # Keywords are now handled by KeywordEditForm, but we must ensure we don't accidentally wipe them?
        # No, update_data reads from UI and writes to 'data' dict reference.
        # Since Keyword form is separate, we only touch non-keyword data here.
        # EXCEPT for auto-setting evolution keyword?

        current_keywords = data.get('keywords', {})
        if type_str == "EVOLUTION_CREATURE" or type_str == "NEO_CREATURE":
            current_keywords['evolution'] = True
        elif 'evolution' in current_keywords:
            del current_keywords['evolution']
        data['keywords'] = current_keywords

    def _get_display_text(self, data):
        return f"{data.get('id', 0)} - {data.get('name', '')}"

    def block_signals_all(self, block):
        self.civ_selector.blockSignals(block)
        self.races_edit.blockSignals(block)
        self.twinpact_check.blockSignals(block)
        super().block_signals_all(block)
