from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QCheckBox, QGroupBox,
    QSplitter, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.card_props import CardPropertiesWidget

class CardEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)

        # Top Section: ID and Twinpact Toggle
        top_form = QFormLayout()

        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        self.id_spin.valueChanged.connect(self.update_data)
        top_form.addRow(tr("ID"), self.id_spin)

        self.twinpact_check = QCheckBox(tr("Is Twinpact Card"))
        self.twinpact_check.setToolTip(tr("Enable to edit Creature and Spell sides simultaneously."))
        self.twinpact_check.stateChanged.connect(self.on_twinpact_toggled)
        top_form.addRow(self.twinpact_check)

        main_layout.addLayout(top_form)

        # Splitter for Creature/Spell sides
        self.splitter = QSplitter(Qt.Orientation.Vertical)

        # Creature Side (Main)
        self.creature_group = QGroupBox(tr("Creature Side (Main)"))
        c_layout = QVBoxLayout(self.creature_group)
        self.creature_props = CardPropertiesWidget(is_spell_side=False)
        self.creature_props.dataChanged.connect(self.update_data)
        c_layout.addWidget(self.creature_props)
        self.splitter.addWidget(self.creature_group)

        # Spell Side (Optional)
        self.spell_group = QGroupBox(tr("Spell Side"))
        s_layout = QVBoxLayout(self.spell_group)
        self.spell_props = CardPropertiesWidget(is_spell_side=True)
        self.spell_props.dataChanged.connect(self.update_data)
        s_layout.addWidget(self.spell_props)
        self.splitter.addWidget(self.spell_group)

        # Initially hide spell side
        self.spell_group.setVisible(False)

        main_layout.addWidget(self.splitter)

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

        main_layout.addWidget(ai_group)

    def on_twinpact_toggled(self, state):
        is_twinpact = (state == Qt.CheckState.Checked.value or state == True)
        self.spell_group.setVisible(is_twinpact)

        # Update labels for clarity
        if is_twinpact:
            self.creature_group.setTitle(tr("Creature Side"))
        else:
            self.creature_group.setTitle(tr("Card Properties"))

        self.update_data()

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.id_spin.setValue(data.get('id', 0))

        spell_side = data.get('spell_side')
        has_spell_side = spell_side is not None and isinstance(spell_side, dict)

        self.twinpact_check.setChecked(has_spell_side)
        self.spell_group.setVisible(has_spell_side)

        # Load Creature/Main Data
        self.creature_props.load_data(data)

        # Load Spell Data if exists, or default
        if has_spell_side:
            self.spell_props.load_data(spell_side)
        else:
            # clear or default?
            self.spell_props.load_data({})

        self.is_key_card_check.setChecked(data.get('is_key_card', False))
        self.ai_importance_spin.setValue(data.get('ai_importance_score', 0))

    def _save_data(self, data):
        data['id'] = self.id_spin.value()

        # Get data from widgets
        c_data = self.creature_props.get_data()

        # Update main data with creature props
        # We merge c_data into data, overwriting keys
        for k, v in c_data.items():
            data[k] = v

        is_twinpact = self.twinpact_check.isChecked()

        if is_twinpact:
            s_data = self.spell_props.get_data()
            s_data['type'] = 'SPELL' # Enforce type for spell side usually
            data['spell_side'] = s_data

            # Auto-format Name: Creature / Spell
            c_name = c_data.get('name', '')
            s_name = s_data.get('name', '')
            if c_name and s_name:
                data['name'] = f"{c_name} / {s_name}"
            elif c_name:
                data['name'] = c_name # Fallback
        else:
            if 'spell_side' in data:
                del data['spell_side']

        # AI Data
        data['is_key_card'] = self.is_key_card_check.isChecked()
        data['ai_importance_score'] = self.ai_importance_spin.value()

    def _get_display_text(self, data):
        return f"{data.get('id', 0)} - {data.get('name', '')}"

    def block_signals_all(self, block):
        self.id_spin.blockSignals(block)
        self.twinpact_check.blockSignals(block)
        self.creature_props.blockSignals_all(block)
        self.spell_props.blockSignals_all(block)
        self.is_key_card_check.blockSignals(block)
        self.ai_importance_spin.blockSignals(block)
