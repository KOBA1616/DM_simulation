# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QLabel, QHBoxLayout, QFormLayout
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.schema_def import FieldType, FieldSchema
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.localization import tr

class ScopeSelectorWidget(QWidget):
    """Wrapper for player scope checkboxes to look like a single widget."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.scope_self = QCheckBox(tr("Self"))
        self.scope_opp = QCheckBox(tr("Opponent"))
        self.layout.addWidget(self.scope_self)
        self.layout.addWidget(self.scope_opp)

        # Default
        self.scope_self.setChecked(True)

    def set_data(self, val):
        self.scope_self.setChecked(False)
        self.scope_opp.setChecked(False)
        if val == "PLAYER_SELF":
            self.scope_self.setChecked(True)
        elif val == "PLAYER_OPPONENT":
            self.scope_opp.setChecked(True)
        elif val == "PLAYER_BOTH":
            self.scope_self.setChecked(True)
            self.scope_opp.setChecked(True)

    def get_data(self):
        s = self.scope_self.isChecked()
        o = self.scope_opp.isChecked()
        if s and o: return "PLAYER_BOTH"
        if o: return "PLAYER_OPPONENT"
        return "PLAYER_SELF"

    def setReadOnly(self, val):
        self.scope_self.setEnabled(not val)
        self.scope_opp.setEnabled(not val)

class WidgetFactory:
    """
    Factory class to create PyQt widgets based on FieldSchema.
    """

    @staticmethod
    def create_widget(schema: FieldSchema, parent=None):
        widget = None

        if schema.field_type == FieldType.INT:
            widget = QSpinBox(parent)
            if schema.min_value is not None:
                widget.setMinimum(schema.min_value)
            else:
                widget.setMinimum(0) # Default
            if schema.max_value is not None:
                widget.setMaximum(schema.max_value)
            else:
                widget.setMaximum(9999)

        elif schema.field_type == FieldType.FLOAT:
            widget = QDoubleSpinBox(parent)

        elif schema.field_type == FieldType.STRING:
            widget = QLineEdit(parent)

        elif schema.field_type == FieldType.BOOL:
            widget = QCheckBox(parent)
            # Label is usually handled by the form layout, but checkbox often has its own text
            # For form consistency, we might leave text empty and use layout label

        elif schema.field_type == FieldType.SELECT:
            widget = QComboBox(parent)
            for opt in schema.options:
                if isinstance(opt, tuple):
                    widget.addItem(str(opt[0]), opt[1])
                else:
                    widget.addItem(str(opt), opt)

        elif schema.field_type == FieldType.ZONE:
            # We assume single zone selector for now
            widget = QComboBox(parent)
            from dm_toolkit.consts import ZONE_NAMES
            for z in ZONE_NAMES:
                widget.addItem(tr(z), z)

        elif schema.field_type == FieldType.FILTER:
            widget = FilterEditorWidget(parent)

        elif schema.field_type == FieldType.LINK:
            widget = VariableLinkWidget(parent)
            if schema.produces_output and hasattr(widget, 'set_output_hint'):
                widget.set_output_hint(True)

        elif schema.field_type == FieldType.PLAYER:
            widget = ScopeSelectorWidget(parent)

        # Set tooltip
        if widget and schema.tooltip:
            widget.setToolTip(schema.tooltip)

        return widget

class FormBuilder:
    """
    Helper to build a form layout from a CommandSchema.
    """
    def __init__(self, parent_form):
        self.form = parent_form

    def build(self, schema, layout: QFormLayout):
        """
        Populates the layout with widgets defined in schema.
        Registers widgets to the parent form.
        """
        for field in schema.fields:
            widget = WidgetFactory.create_widget(field, self.form)
            if not widget:
                continue

            # Register with base form
            # Note: We need to ensure unique keys if multiple fields share generic names?
            # The schema keys should be unique per command.
            self.form.register_widget(widget, field.key)

            # Add to layout
            layout.addRow(field.label, widget)

            # Connect signals if possible to update_data
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.form.update_data)
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.form.update_data)
            elif hasattr(widget, 'currentIndexChanged'):
                widget.currentIndexChanged.connect(self.form.update_data)
            elif hasattr(widget, 'stateChanged'):
                widget.stateChanged.connect(self.form.update_data)
            elif hasattr(widget, 'filterChanged'):
                widget.filterChanged.connect(self.form.update_data)
            elif hasattr(widget, 'linkChanged'):
                widget.linkChanged.connect(self.form.update_data)
            # Custom ScopeSelectorWidget uses checkboxes inside
            elif isinstance(widget, ScopeSelectorWidget):
                 widget.scope_self.stateChanged.connect(self.form.update_data)
                 widget.scope_opp.stateChanged.connect(self.form.update_data)
