from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QLabel, QGridLayout, QGroupBox, QPushButton,
    QVBoxLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector

# Simplified Spell Side Editor Widget (Sub-form)
class SpellSideWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QLabel(tr("--- Twinpact Spell Side ---"))
        self.header.setStyleSheet("font-weight: bold; color: blue;")
        layout.addRow(self.header)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("Spell Side Name"))
        layout.addRow(tr("Name"), self.name_edit)

        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        layout.addRow(tr("Cost"), self.cost_spin)

        # Note: Effects for spell side should be added in the tree view
        # We might need a way to indicate that an effect belongs to the spell side
        self.info_label = QLabel(tr("Effects for Spell side are managed in the tree."))
        layout.addRow(self.info_label)

class CardEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyword_checks = {} # Map key -> QCheckBox
        self.is_twinpact = False
        self.spell_side_data = {} # Keep spell side data
        self.setup_ui()

    def setup_ui(self):
        # Main layout is a vertical splitter? Or just VBox?
        # Requirement: "Enable registering/editing Twinpact cards intuitively... split the editor screen"

        main_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.form_layout = QFormLayout(self.scroll_content)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        # ID
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        self.form_layout.addRow(tr("ID"), self.id_spin)

        # Name
        self.name_edit = QLineEdit()
        self.form_layout.addRow(tr("Name"), self.name_edit)

        # Twinpact Checkbox
        self.twinpact_check = QCheckBox(tr("Is Twinpact?"))
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

        # Cost & Power
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.form_layout.addRow(tr("Cost"), self.cost_spin)

        self.power_spin = QSpinBox()
        self.power_spin.setRange(0, 99999)
        self.power_spin.setSingleStep(500)
        self.form_layout.addRow(tr("Power"), self.power_spin)

        self.races_edit = QLineEdit()
        self.form_layout.addRow(tr("Races"), self.races_edit)

        # Spell Side Widget (Hidden by default)
        self.spell_widget = SpellSideWidget()
        self.spell_widget.setVisible(False)
        self.spell_widget.name_edit.textChanged.connect(self.update_data)
        self.spell_widget.cost_spin.valueChanged.connect(self.update_data)
        self.form_layout.addRow(self.spell_widget)

        # Keywords Section
        kw_group = QGroupBox(tr("Keywords"))
        kw_layout = QGridLayout(kw_group)

        keywords_list = [
            "speed_attacker", "blocker", "slayer",
            "double_breaker", "triple_breaker", "shield_trigger",
            "evolution", "just_diver", "mach_fighter", "g_strike",
            "hyper_energy", "shield_burn", "revolution_change", "untap_in",
            "meta_counter_play", "power_attacker"
        ]

        kw_map = {
            "speed_attacker": "Speed Attacker",
            "blocker": "Blocker",
            "slayer": "Slayer",
            "double_breaker": "Double Breaker",
            "triple_breaker": "Triple Breaker",
            "shield_trigger": "Shield Trigger",
            "evolution": "Evolution",
            "just_diver": "Just Diver",
            "mach_fighter": "Mach Fighter",
            "g_strike": "G Strike",
            "hyper_energy": "Hyper Energy",
            "shield_burn": "Shield Burn",
            "revolution_change": "Revolution Change",
            "untap_in": "Untap In",
            "meta_counter_play": "Meta Counter",
            "power_attacker": "Power Attacker"
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

        # AI Configuration Section
        ai_group = QGroupBox(tr("AI Configuration"))
        ai_layout = QFormLayout(ai_group)

        self.is_key_card_check = QCheckBox(tr("Is Key Card / Combo Piece"))
        self.is_key_card_check.setToolTip(tr("Mark this card as a high-value target for AI analysis."))
        self.is_key_card_check.stateChanged.connect(self.update_data)
        ai_layout.addRow(self.is_key_card_check)

        self.ai_importance_spin = QSpinBox()
        self.ai_importance_spin.setRange(0, 1000)
        self.ai_importance_spin.setToolTip(tr("Manual importance score for AI (0 = default)."))
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
        self.is_twinpact = (state == Qt.CheckState.Checked.value or state == True)
        self.spell_widget.setVisible(self.is_twinpact)
        # Force update logic for spell side initialization if needed
        if self.is_twinpact and not self.spell_side_data:
            self.spell_side_data = {
                'name': '',
                'cost': 0,
                'type': 'SPELL',
                'effects': []
            }
        self.update_data()

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.id_spin.setValue(data.get('id', 0))

        # Name handling for Twinpact
        name = data.get('name', '')
        self.name_edit.setText(name)

        # Load Civilization(s)
        civs = data.get('civilizations')
        if not civs:
            civ_single = data.get('civilization')
            if civ_single:
                civs = [civ_single]
        self.civ_selector.set_selected_civs(civs)

        self.set_combo_by_data(self.type_combo, data.get('type', 'CREATURE'))

        self.cost_spin.setValue(data.get('cost', 0))
        self.power_spin.setValue(data.get('power', 0))
        self.races_edit.setText(", ".join(data.get('races', [])))

        # Twinpact handling
        self.spell_side_data = data.get('spell_side')
        if self.spell_side_data:
            self.twinpact_check.setChecked(True)
            self.spell_widget.name_edit.setText(self.spell_side_data.get('name', ''))
            self.spell_widget.cost_spin.setValue(self.spell_side_data.get('cost', 0))
        else:
            self.twinpact_check.setChecked(False)
            self.spell_widget.name_edit.clear()
            self.spell_widget.cost_spin.setValue(0)

        # Load Keywords
        kw_data = data.get('keywords', {})
        for k, cb in self.keyword_checks.items():
            is_checked = kw_data.get(k, False)
            cb.setChecked(is_checked)

        # Load AI Data
        self.is_key_card_check.setChecked(data.get('is_key_card', False))
        self.ai_importance_spin.setValue(data.get('ai_importance_score', 0))

    def _save_data(self, data):
        data['id'] = self.id_spin.value()

        # Sync name logic: if Twinpact, combine names?
        # Requirement: "Card name split by / should be reflected"
        # However, for editing, we might want separate fields, but save combined?
        # Let's keep `name` as the primary card name.
        # Ideally, `name` = "Creature / Spell"

        creature_name = self.name_edit.text()
        data['name'] = creature_name

        data['civilizations'] = self.civ_selector.get_selected_civs()
        if 'civilization' in data:
            del data['civilization']

        data['type'] = self.type_combo.currentData()
        data['cost'] = self.cost_spin.value()
        data['power'] = self.power_spin.value()
        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        if self.twinpact_check.isChecked():
            spell_name = self.spell_widget.name_edit.text()
            # Construct combined name if needed, but Engine handles separate names in spell_side
            if spell_name:
                 # Update spell side data
                 if not self.spell_side_data:
                     self.spell_side_data = {'type': 'SPELL', 'effects': []}

                 self.spell_side_data['name'] = spell_name
                 self.spell_side_data['cost'] = self.spell_widget.cost_spin.value()
                 self.spell_side_data['type'] = 'SPELL'
                 # Copy civilizations from parent? Twinpacts usually share civs.
                 self.spell_side_data['civilizations'] = data['civilizations']

                 data['spell_side'] = self.spell_side_data

                 # Optional: Update main name to "Creature / Spell" for display
                 if "/" not in creature_name:
                     data['name'] = f"{creature_name} / {spell_name}"
            else:
                 # If empty spell name, maybe warn?
                 pass
        else:
            if 'spell_side' in data:
                del data['spell_side']
            self.spell_side_data = None

            # Clean up name if it contains /
            if "/" in creature_name:
                data['name'] = creature_name.split("/")[0].strip()

        # Update Keywords
        kw_data = {}
        for k, cb in self.keyword_checks.items():
            if cb.isChecked():
                kw_data[k] = True
        data['keywords'] = kw_data

        # Save AI Data
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
        self.spell_widget.name_edit.blockSignals(block)
        self.spell_widget.cost_spin.blockSignals(block)
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.is_key_card_check.blockSignals(block)
        self.ai_importance_spin.blockSignals(block)
