# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QLabel, QGroupBox,
    QVBoxLayout
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget

class ModifierEditForm(BaseEditForm):
    """
    Form to edit a Static Ability (ModifierDef).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Top section: Basic Modifier Properties
        self.basic_group = QGroupBox(tr("Modifier Settings"))
        form_layout = QFormLayout(self.basic_group)

        # Type
        self.type_combo = QComboBox()
        # "NONE", "COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"
        self.type_combo.addItems(["NONE", "COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"])
        self.type_combo.currentTextChanged.connect(self.update_data)
        self.type_combo.currentTextChanged.connect(self.update_visibility)
        form_layout.addRow(tr("Type"), self.type_combo)

        # Value (for Power/Cost)
        self.value_spin = QSpinBox()
        self.value_spin.setRange(-99999, 99999)
        self.value_spin.valueChanged.connect(self.update_data)
        self.label_value = QLabel(tr("Value"))
        form_layout.addRow(self.label_value, self.value_spin)

        # String Value (for Keyword)
        self.str_val_edit = QLineEdit()
        self.str_val_edit.textChanged.connect(self.update_data)
        self.label_str_val = QLabel(tr("Keyword / String"))
        form_layout.addRow(self.label_str_val, self.str_val_edit)

        # Keyword Helper (ComboBox) - optional, for easier selection
        self.keyword_combo = QComboBox()
        self.keyword_combo.addItems([
            "speed_attacker", "blocker", "slayer", "double_breaker", "triple_breaker",
            "shield_trigger", "mach_fighter", "just_diver", "g_strike"
        ])
        self.keyword_combo.currentTextChanged.connect(self.on_keyword_combo_changed)
        self.label_keyword = QLabel(tr("Select Keyword"))
        form_layout.addRow(self.label_keyword, self.keyword_combo)

        layout.addWidget(self.basic_group)

        # Condition Section
        self.condition_widget = ConditionEditorWidget()
        self.condition_widget.dataChanged.connect(self.update_data)
        layout.addWidget(self.condition_widget)

        # Filter Section (Target Filter or Source Filter)
        self.filter_widget = FilterEditorWidget()
        self.filter_widget.basic_group.setTitle(tr("Target Filter")) # FilterWidget doesn't have setTitle, but groups do
        self.filter_widget.filterChanged.connect(self.update_data)
        layout.addWidget(self.filter_widget)

        # Define bindings
        # Note: type_combo uses text, not user data, so binding won't work automatically with current BaseEditForm
        # which uses currentData() for ComboBox.
        # But wait, QComboBox without user data uses text as data if addItem(text) is used.
        # But we need to verify if set_combo_by_data works. It uses findData.
        # If we added items with text only, data is None or same as text?
        # Actually in Qt6, currentData returns None if not set?
        # ModifierEditForm uses addItems which sets text.
        # So we should probably fix ModifierEditForm to use data or custom populate.

        # Let's fix type_combo population to use data same as text
        self.type_combo.clear()
        types = ["NONE", "COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"]
        for t in types: self.type_combo.addItem(t, t)

        self.bindings = {
            'type': self.type_combo,
            'value': self.value_spin,
            'str_val': self.str_val_edit,
            'condition': self.condition_widget,
            'filter': self.filter_widget
        }

        # Initial visibility
        self.update_visibility()

    def update_visibility(self):
        mtype = self.type_combo.currentText()

        # Defaults
        self.label_value.setVisible(False)
        self.value_spin.setVisible(False)
        self.label_str_val.setVisible(False)
        self.str_val_edit.setVisible(False)
        self.label_keyword.setVisible(False)
        self.keyword_combo.setVisible(False)

        if mtype == "COST_MODIFIER":
            self.label_value.setVisible(True)
            self.value_spin.setVisible(True)
            self.label_value.setText(tr("Reduction Amount"))
            self.filter_widget.setTitle(tr("Cards to Reduce"))

        elif mtype == "POWER_MODIFIER":
            self.label_value.setVisible(True)
            self.value_spin.setVisible(True)
            self.label_value.setText(tr("Power Amount"))
            self.filter_widget.setTitle(tr("Creatures to Buff"))

        elif mtype == "GRANT_KEYWORD":
            self.label_str_val.setVisible(True)
            self.str_val_edit.setVisible(True)
            self.label_keyword.setVisible(True)
            self.keyword_combo.setVisible(True)
            self.filter_widget.setTitle(tr("Target Creatures"))

        elif mtype == "SET_KEYWORD":
            self.label_str_val.setVisible(True)
            self.str_val_edit.setVisible(True)
            self.label_keyword.setVisible(True)
            self.keyword_combo.setVisible(True)
            self.filter_widget.setTitle(tr("Target Creatures"))

    def on_keyword_combo_changed(self, text):
        self.str_val_edit.setText(text)

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        # Fallbacks for empty structures
        if not data.get('condition'):
             self.condition_widget.set_data({"type": "NONE"})
        if not data.get('filter'):
             self.filter_widget.set_data({})

        self._apply_bindings(data)

        self.update_visibility()

    def _save_data(self, data):
        self._collect_bindings(data)

        # Clean up empty filter/condition?
        # Maybe not strictly necessary but cleaner JSON
        pass

    def _get_display_text(self, data):
        mtype = data.get('type', 'NONE')
        return f"{tr('Static')}: {tr(mtype)}"

    def block_signals_all(self, block):
        self.keyword_combo.blockSignals(block)
        super().block_signals_all(block)

    def set_combo_text(self, combo, text):
        # Legacy support if needed, but BaseEditForm.set_combo_by_data should work now
        # that we populated with data.
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)
