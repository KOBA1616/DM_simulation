# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QSpinBox, QComboBox, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
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
        self.add_field(tr("Name"), self.name_edit, 'name')

        # Civilization
        self.civ_selector = CivilizationSelector()
        self.civ_selector.changed.connect(self.update_data)
        self.register_widget(self.civ_selector)
        self.add_field(tr("Civilization"), self.civ_selector)

        # Cost
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.add_field(tr("Cost"), self.cost_spin, 'cost')

        # Connect signals
        self.name_edit.textChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)

        main_layout.addStretch()

    def _load_ui_from_data(self, data, item):
        # Apply standard bindings
        self._apply_bindings(data)

        civs = data.get('civilizations')
        if not civs:
            civ_single = data.get('civilization')
            if civ_single:
                civs = [civ_single]
        self.civ_selector.set_selected_civs(civs)

    def _save_ui_to_data(self, data):
        self._collect_bindings(data)
        data['civilizations'] = self.civ_selector.get_selected_civs()
        if 'civilization' in data:
            del data['civilization']
        data['type'] = 'SPELL' # Always SPELL for spell side

    def _get_display_text(self, data):
        return f"{tr('Spell Side')}: {data.get('name', '')}"
