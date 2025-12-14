from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QLabel, QGridLayout, QGroupBox, QPushButton,
    QVBoxLayout, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget

class CardEditForm(BaseEditForm):
    # Signal to request structural changes in the Logic Tree
    # command: "ADD_SPELL_SIDE", "REMOVE_SPELL_SIDE", "ADD_REV_CHANGE", "REMOVE_REV_CHANGE"
    # payload: dict (optional)
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyword_checks = {} # Map key -> QCheckBox
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
        types = ["CREATURE", "SPELL", "EVOLUTION_CREATURE", "TAMASEED", "CROSS_GEAR", "CASTLE", "PSYCHIC_CREATURE", "GR_CREATURE"]
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

        # Keywords Section
        kw_group = QGroupBox(tr("Keywords"))
        kw_layout = QGridLayout(kw_group)

        # Removed legacy keywords: evolution, untap_in, meta_counter_play
        # Revolution Change is handled separately now
        keywords_list = [
            "speed_attacker", "blocker", "slayer",
            "double_breaker", "triple_breaker", "shield_trigger",
            "just_diver", "mach_fighter", "g_strike",
            "hyper_energy", "shield_burn", "power_attacker", "ex_life"
        ]

        kw_map = {
            "speed_attacker": "Speed Attacker",
            "blocker": "Blocker",
            "slayer": "Slayer",
            "double_breaker": "Double Breaker",
            "triple_breaker": "Triple Breaker",
            "shield_trigger": "Shield Trigger",
            "just_diver": "Just Diver",
            "mach_fighter": "Mach Fighter",
            "g_strike": "G Strike",
            "hyper_energy": "Hyper Energy",
            "shield_burn": "Shield Burn",
            "power_attacker": "Power Attacker",
            "ex_life": "Ex-Life"
        }

        row = 0
        col = 0
        for k in keywords_list:
            cb = QCheckBox(tr(kw_map.get(k, k)))
            kw_layout.addWidget(cb, row, col)
            self.keyword_checks[k] = cb
            cb.stateChanged.connect(self.update_data)

            col += 1
            if col > 2: # 3 columns
                col = 0
                row += 1

        self.form_layout.addRow(kw_group)

        # Special Abilities Generator
        special_group = QGroupBox(tr("Special Abilities"))
        special_layout = QVBoxLayout(special_group)

        # Revolution Change Checkbox
        self.rev_change_check = QCheckBox(tr("Revolution Change"))
        self.rev_change_check.setToolTip(tr("Enable Revolution Change to generate the necessary logic tree structure."))
        self.rev_change_check.stateChanged.connect(self.toggle_rev_change)
        special_layout.addWidget(self.rev_change_check)

        self.form_layout.addRow(special_group)

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

        # Connect signals
        self.id_spin.valueChanged.connect(self.update_data)
        self.name_edit.textChanged.connect(self.update_data)
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.power_spin.valueChanged.connect(self.update_data)
        self.races_edit.textChanged.connect(self.update_data)

    def toggle_twinpact(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit("ADD_SPELL_SIDE", {})
        else:
            self.structure_update_requested.emit("REMOVE_SPELL_SIDE", {})
        # Note: We do NOT call update_data here as this is a structural change handled by the Tree/DataManager

    def toggle_rev_change(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit("ADD_REV_CHANGE", {})
        else:
            self.structure_update_requested.emit("REMOVE_REV_CHANGE", {})

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.id_spin.setValue(data.get('id', 0))
        self.name_edit.setText(data.get('name', ''))

        civs = data.get('civilizations')
        if not civs:
            civ_single = data.get('civilization')
            if civ_single: civs = [civ_single]
        self.civ_selector.set_selected_civs(civs)

        current_type = data.get('type', 'CREATURE')
        self.set_combo_by_data(self.type_combo, current_type)
        self.update_type_visibility(current_type)

        self.cost_spin.setValue(data.get('cost', 0))
        self.power_spin.setValue(data.get('power', 0))
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

        kw_data = data.get('keywords', {})
        for k, cb in self.keyword_checks.items():
            is_checked = kw_data.get(k, False)
            cb.setChecked(is_checked)

        # Check Rev Change
        self.rev_change_check.blockSignals(True)
        self.rev_change_check.setChecked(kw_data.get('revolution_change', False))
        self.rev_change_check.blockSignals(False)

        self.is_key_card_check.setChecked(data.get('is_key_card', False))
        self.ai_importance_spin.setValue(data.get('ai_importance_score', 0))

    def update_type_visibility(self, type_str):
        # Hide Power if Spell
        is_spell = (type_str == "SPELL")
        self.power_spin.setVisible(not is_spell)
        self.lbl_power.setVisible(not is_spell)

    def _save_data(self, data):
        data['id'] = self.id_spin.value()
        data['name'] = self.name_edit.text()

        data['civilizations'] = self.civ_selector.get_selected_civs()
        if 'civilization' in data:
            del data['civilization']

        type_str = self.type_combo.currentData()
        data['type'] = type_str
        self.update_type_visibility(type_str)

        data['cost'] = self.cost_spin.value()
        data['power'] = self.power_spin.value()
        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        # Keyword handling
        # IMPORTANT: Preserve keywords not managed by this loop (like revolution_change)
        # We start with existing keywords to keep 'revolution_change' if it's there
        # but we must ensure we don't keep stale ones from the list.
        current_keywords = data.get('keywords', {}).copy()

        # Reset managed keywords
        for k in self.keyword_checks.keys():
            if k in current_keywords:
                del current_keywords[k]

        # Add checked ones
        for k, cb in self.keyword_checks.items():
            if cb.isChecked():
                current_keywords[k] = True

        # Ensure revolution_change keyword is set correctly if checkbox is checked
        if self.rev_change_check.isChecked():
            current_keywords['revolution_change'] = True
        elif 'revolution_change' in current_keywords:
            # If unchecked, we might need to remove it, but usually the structure removal handles it.
            # However, if we are in save_data, the UI state should be the source of truth.
            del current_keywords['revolution_change']

        # Auto-set evolution keyword based on type
        if type_str == "EVOLUTION_CREATURE":
            current_keywords['evolution'] = True

        data['keywords'] = current_keywords

        # data['reaction_abilities'] managed by Logic Tree now
        data['is_key_card'] = self.is_key_card_check.isChecked()
        data['ai_importance_score'] = self.ai_importance_spin.value()

    def _get_display_text(self, data):
        return f"{data.get('id', 0)} - {data.get('name', '')}"

    def block_signals_all(self, block):
        self.id_spin.blockSignals(block)
        self.name_edit.blockSignals(block)
        self.civ_selector.blockSignals(block)
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.power_spin.blockSignals(block)
        self.races_edit.blockSignals(block)
        self.twinpact_check.blockSignals(block)
        self.rev_change_check.blockSignals(block)
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.is_key_card_check.blockSignals(block)
        self.ai_importance_spin.blockSignals(block)
