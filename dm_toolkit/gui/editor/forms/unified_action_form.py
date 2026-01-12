# -*- coding: utf-8 -*-
import json
import os
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor.widget_factory import WidgetFactory
from dm_toolkit.gui.editor.configs.action_config import COMMAND_GROUPS
from dm_toolkit.gui.editor.schema_def import SchemaLoader, get_schema, FieldSchema, FieldType

class UnifiedActionForm(BaseEditForm):
    """Schema-driven Unified Action Form using WidgetFactory and External Config."""
    
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets_map = {} # key -> widget
        self.current_model = None # CommandModel instance
        
        # Initialize Schema Registry from schema_config.py
        from dm_toolkit.gui.editor.schema_config import register_all_schemas
        register_all_schemas()

        try:
            self.setup_base_ui()
        except Exception:
            pass

    def setup_base_ui(self):
        """Sets up the fixed top-level UI (Group/Type selectors)."""
        self.main_layout = QFormLayout(self)

        # Group Combo
        self.action_group_combo = QComboBox()
        self.populate_combo(self.action_group_combo, list(COMMAND_GROUPS.keys()), display_func=tr)
        self.main_layout.addRow(tr("Command Group"), self.action_group_combo)
        self.action_group_combo.currentIndexChanged.connect(self.on_group_changed)

        # Type Combo
        self.type_combo = QComboBox()
        self.main_layout.addRow(tr("Command Type"), self.type_combo)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)

        # Dynamic Content Container
        self.dynamic_container = QWidget()
        self.dynamic_layout = QFormLayout(self.dynamic_container)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addRow(self.dynamic_container)

        # Trigger initial population
        self.on_group_changed()

    def on_group_changed(self):
        grp = self.action_group_combo.currentData()
        types = COMMAND_GROUPS.get(grp, [])
        if not types:
            types = []

        prev = self.type_combo.currentData()
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)

        if prev and prev in types:
            self.set_combo_by_data(self.type_combo, prev)
        else:
            if types:
                self.type_combo.setCurrentIndex(0)

    def on_type_changed(self):
        t = self.type_combo.currentData()
        if t is None:
            t = "DRAW"  # Default fallback
        self.rebuild_dynamic_ui(t)
        self.update_data()

    def rebuild_dynamic_ui(self, cmd_type):
        """Rebuilds the dynamic part of the form based on schema."""
        # Clear existing widgets
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.widgets_map = {}

        schema = get_schema(cmd_type)
        if not schema:
            return

        # Group optional and up_to fields for horizontal layout
        optional_field = None
        up_to_field = None
        other_fields = []
        
        for field_schema in schema.fields:
            if field_schema.key == 'optional':
                optional_field = field_schema
            elif field_schema.key == 'up_to':
                up_to_field = field_schema
            else:
                other_fields.append(field_schema)
        
        # Add regular fields first
        for field_schema in other_fields:
            self._create_widget_for_field(field_schema)
        
        # Add optional and up_to in a horizontal layout if both exist
        if optional_field or up_to_field:
            from PyQt6.QtWidgets import QHBoxLayout, QWidget
            options_widget = QWidget()
            options_layout = QHBoxLayout(options_widget)
            options_layout.setContentsMargins(0, 0, 0, 0)
            
            if optional_field:
                opt_widget = WidgetFactory.create_widget(self, optional_field, self.update_data)
                if opt_widget:
                    self.widgets_map['optional'] = opt_widget
                    options_layout.addWidget(opt_widget)
            
            if up_to_field:
                upto_widget = WidgetFactory.create_widget(self, up_to_field, self.update_data)
                if upto_widget:
                    self.widgets_map['up_to'] = upto_widget
                    options_layout.addWidget(upto_widget)
            
            options_layout.addStretch()
            self.dynamic_layout.addRow(tr("Options"), options_widget)

    def _create_widget_for_field(self, field_schema: FieldSchema):
        key = field_schema.key

        widget = WidgetFactory.create_widget(self, field_schema, self.update_data)

        if widget:
            # Set produces_output hint for VariableLinkWidget
            if field_schema.field_type == FieldType.LINK and hasattr(widget, 'set_output_hint'):
                widget.set_output_hint(field_schema.produces_output)
            
            self.widgets_map[key] = widget
            self.dynamic_layout.addRow(tr(field_schema.label), widget)

    def _load_ui_from_data(self, data, item):
        """Loads data using interface-based widgets."""
        if not data: data = {}
        model = CommandModel(**data)
        self.current_model = model
        self.current_item = item

        cmd_type = model.type
        # Mapping back group
        grp = 'OTHER'
        for g, types in COMMAND_GROUPS.items():
            if cmd_type in types:
                grp = g
                break
        
        self.set_combo_by_data(self.action_group_combo, grp)
        self.set_combo_by_data(self.type_combo, cmd_type)
        
        # Set current_item for VariableLinkWidget
        for key, widget in self.widgets_map.items():
            if key == 'links' or key == 'input_link' or key == 'output_link':
                if hasattr(widget, 'set_current_item'):
                    widget.set_current_item(item)
        
        # Populate widgets via interface
        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'set_value'):
                # Special handling for flattened models vs structured widgets
                if key == 'links' or key == 'input_link' or key == 'output_link':
                    widget.set_value(data) # VariableLink expects full dict
                elif key == 'target_filter':
                    widget.set_value(model.target_filter.model_dump() if model.target_filter else {})
                elif key == 'options':
                    widget.set_value(model.options)
                else:
                    val = getattr(model, key, None)
                    if val is not None:
                        widget.set_value(val)

    def _save_ui_to_data(self, data):
        """Saves data using interface-based widgets."""
        cmd_type = self.type_combo.currentData()
        if cmd_type is None:
            cmd_type = "DRAW"  # Default fallback

        # Start with fresh dict
        new_data = {'type': cmd_type}

        # Use Pydantic model for validation/structure
        model = CommandModel(type=cmd_type)

        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'get_value'):
                val = widget.get_value()

                if key == 'links' or key == 'input_link' or key == 'output_link':
                    # VariableLink returns a dict of updates
                    new_data.update(val)
                elif key == 'target_filter':
                     # FilterEditor returns dict
                     if val: model.target_filter = val
                else:
                     if hasattr(model, key):
                         setattr(model, key, val)

        # Merge model back to dict
        dump = model.model_dump(exclude_none=True)
        new_data.update(dump)

        # Check outputs config
        schema = get_schema(cmd_type)
        if schema:
            # Check if any field produces output
            produces_output = any(f.produces_output for f in schema.fields)
            if produces_output and 'output_value_key' not in new_data:
                 row = 0
                 if getattr(self, 'current_item', None):
                     row = self.current_item.row()
                 new_data['output_value_key'] = f"var_{cmd_type}_{row}"

        data.clear()
        data.update(new_data)

    def _get_display_text(self, data):
        t = data.get('type', 'UNKNOWN')
        return f"{tr('Command')}: {tr(t)}"
