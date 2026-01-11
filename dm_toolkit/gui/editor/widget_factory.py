# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QLabel, QHBoxLayout, QFormLayout
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.schema_def import FieldType, FieldSchema
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.forms.unified_widgets import make_zone_combos

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
            # Use utility to populate zones?
            # For simplicity in this factory, we just create the generic combo
            # Real implementation might need access to make_zone_combos logic
            from dm_toolkit.consts import ZONE_NAMES
            for z in ZONE_NAMES:
                widget.addItem(z, z)

        elif schema.field_type == FieldType.FILTER:
            widget = FilterEditorWidget(parent)

        elif schema.field_type == FieldType.LINK:
            widget = VariableLinkWidget(parent)
            if schema.produces_output and hasattr(widget, 'set_output_hint'):
                widget.set_output_hint(True)

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
        # Clear existing layout rows?
        # Usually we assume a fresh build or cleared layout.

        for field in schema.fields:
            widget = WidgetFactory.create_widget(field, self.form)
            if not widget:
                continue

            # Register with base form
            self.form.register_widget(widget, field.key)

            # Add to layout
            layout.addRow(field.label, widget)

            # Store reference in form if needed?
            # BaseForm stores it in self.bindings[key]
