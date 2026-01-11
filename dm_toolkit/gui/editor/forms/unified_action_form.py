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
from dm_toolkit.gui.editor.configs.action_config import COMMAND_GROUPS, COMMAND_UI_CONFIG

class UnifiedActionForm(BaseEditForm):
    """Schema-driven Unified Action Form using WidgetFactory and External Config."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets_map = {} # key -> widget
        self.current_model = None # CommandModel instance

        # Ensure minimal attributes for headless env
        self.current_item = getattr(self, 'current_item', None)
        self.structure_update_requested = getattr(self, 'structure_update_requested', None)
        
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
        self.rebuild_dynamic_ui(t)
        self.update_data()

    def rebuild_dynamic_ui(self, cmd_type):
        """Rebuilds the dynamic part of the form based on schema/config."""
        # Clear existing widgets
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.widgets_map = {}

        config = COMMAND_UI_CONFIG.get(cmd_type, {})
        # Support both new 'fields' list and legacy 'visible' list (compatibility)
        fields = config.get('fields', [])

        if not fields and 'visible' in config:
            # Fallback mapper for legacy config structure if present
            fields = self._map_legacy_visible_to_fields(config['visible'], config)

        for field_def in fields:
            self._create_widget_from_def(field_def)

    def _map_legacy_visible_to_fields(self, visible_keys, config):
        """Compatibility layer for old 'visible' list config."""
        fields = []
        for key in visible_keys:
            f = {'key': key, 'label': key}
            # Infer widget type (same logic as before)
            if key == 'target_group':
                f['widget'] = 'player_scope'
                f['label'] = 'Target Player'
            elif key == 'target_filter':
                f['widget'] = 'filter_editor'
                f['label'] = 'Filter'
            elif key == 'amount':
                f['widget'] = 'spinbox'
                f['label'] = config.get('label_amount', 'Amount')
            elif key == 'str_param':
                f['widget'] = 'text'
                f['label'] = config.get('label_str_param', 'String Param')
            elif key == 'from_zone':
                f['widget'] = 'zone_combo'
                f['label'] = config.get('label_from_zone', 'From Zone')
            elif key == 'to_zone':
                f['widget'] = 'zone_combo'
                f['label'] = config.get('label_to_zone', 'To Zone')
            elif key == 'optional':
                f['widget'] = 'checkbox'
                f['label'] = 'Optional'
            elif key in ['input_link', 'output_link']:
                f['widget'] = 'variable_link'
                f['label'] = 'Variables'
            elif key == 'mutation_kind':
                f['widget'] = 'text'
                f['label'] = config.get('label_mutation_kind', 'Mutation')
            elif key == 'ref_mode':
                 f['widget'] = 'ref_mode_combo'
                 f['label'] = 'Reference Mode'
            fields.append(f)
        return fields

    def _create_widget_from_def(self, field_def):
        key = field_def['key']
        widget = WidgetFactory.create_widget(self, field_def, self.update_data)

        if widget:
            self.widgets_map[key] = widget
            self.dynamic_layout.addRow(tr(field_def.get('label', key)), widget)

    def _load_ui_from_data(self, data, item):
        """Loads data using interface-based widgets."""
        if not data: data = {}
        model = CommandModel(**data)
        self.current_model = model

        cmd_type = model.type
        # Mapping back group
        grp = 'OTHER'
        for g, types in COMMAND_GROUPS.items():
            if cmd_type in types:
                grp = g
                break
        
        self.set_combo_by_data(self.action_group_combo, grp)
        self.set_combo_by_data(self.type_combo, cmd_type)
        
        # Populate widgets via interface
        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'set_value'):
                # Special handling for complex types
                if key == 'input_link' or key == 'output_link':
                    widget.set_value(data)
                elif key == 'target_filter':
                    widget.set_value(model.target_filter.model_dump() if model.target_filter else {})
                elif key == 'optional':
                    widget.set_value(model.optional)
                else:
                    val = getattr(model, key, None)
                    if val is not None:
                        widget.set_value(val)

    def _save_ui_to_data(self, data):
        """Saves data using interface-based widgets."""
        cmd_type = self.type_combo.currentData()

        # Start with fresh dict
        new_data = {'type': cmd_type}

        # Use Pydantic model for validation/structure
        model = CommandModel(type=cmd_type)

        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'get_value'):
                val = widget.get_value()

                if key == 'input_link' or key == 'output_link':
                    new_data.update(val)
                elif key == 'target_filter':
                     if val: model.target_filter = val
                else:
                     if hasattr(model, key):
                         setattr(model, key, val)

        # Merge model back to dict
        dump = model.model_dump(exclude_none=True)
        new_data.update(dump)

        # Check outputs config
        config = COMMAND_UI_CONFIG.get(cmd_type, {})
        # Handle both list and dict styles
        produces_output = config.get('produces_output', False)

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
