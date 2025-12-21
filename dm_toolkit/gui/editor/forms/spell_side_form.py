# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QSpinBox, QComboBox, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector

class SpellSideForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        main_layout.addLayout(self.form_layout)

        self.info_label = QLabel(tr("Edit Spell Side Properties"))
        self.info_label.setStyleSheet("font-weight: bold; color: blue;")
        self.form_layout.addRow(self.info_label)

        # Name
        self.name_edit = QLineEdit()
        self.connect_signal(self.name_edit, self.name_edit.textChanged, self.update_data)
        self.form_layout.addRow(tr("Name"), self.name_edit)

        # Civilization
        self.civ_selector = CivilizationSelector()
        self.register_widget(self.civ_selector)
        self.civ_selector.changed.connect(self.update_data)
        self.form_layout.addRow(tr("Civilization"), self.civ_selector)

        # Cost
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.connect_signal(self.cost_spin, self.cost_spin.valueChanged, self.update_data)
        self.form_layout.addRow(tr("Cost"), self.cost_spin)

        main_layout.addStretch()

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)
        if not data: data = {}

        self.name_edit.setText(data.get('name', ''))

        civs = data.get('civilizations')
        if not civs:
            civ_single = data.get('civilization')
            if civ_single:
                civs = [civ_single]
        self.civ_selector.set_selected_civs(civs)

        self.cost_spin.setValue(data.get('cost', 0))

    def _save_data(self, data):
        data['name'] = self.name_edit.text()
        data['civilizations'] = self.civ_selector.get_selected_civs()
        if 'civilization' in data:
            del data['civilization']
        data['cost'] = self.cost_spin.value()
        data['type'] = 'SPELL' # Always SPELL for spell side

    def _get_display_text(self, data):
        return f"{tr('Spell Side')}: {data.get('name', '')}"
