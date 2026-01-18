# -*- coding: utf-8 -*-
# unified_action_form.py
# スキーマ駆動の動的フォーム生成。復数だが、単一の責務（アクション設定）に集中している。
# 依存: AIが編集する際は schema_def.py や ACTION_UI_CONFIG (command_config.py) とセットで理解する必要があります。
# 重要: このファイルは削除しないでください。schema_def.py および command_config.py と密接に連携して動作します。

import json
import os
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from pydantic import ValidationError
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm, get_attr, to_dict
from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor.widget_factory import WidgetFactory
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
from dm_toolkit.gui.editor.schema_def import SchemaLoader, get_schema, FieldSchema, FieldType
from dm_toolkit.gui.editor.consts import STRUCT_CMD_GENERATE_OPTIONS

COMMAND_GROUPS = EditorConfigLoader.get_command_groups()
UI_CONFIG = EditorConfigLoader.get_command_ui_config()

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
        
        # Block signals to prevent triggering on_type_changed during population
        self.type_combo.blockSignals(True)
        try:
            self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)

            if prev and prev in types:
                self.set_combo_by_data(self.type_combo, prev)
            else:
                if types:
                    self.type_combo.setCurrentIndex(0)
        finally:
            self.type_combo.blockSignals(False)
        
        # Manually trigger UI rebuild with the new selection
        self.on_type_changed()

    def on_type_changed(self):
        t = self.type_combo.currentData()
        if t is None:
            # Fallback: use first item in combo, or NONE if empty
            if self.type_combo.count() > 0:
                t = self.type_combo.itemData(0)
            if t is None:
                t = "NONE"  # Final fallback to valid command type
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

        # Load UI config overrides
        ui_cfg = UI_CONFIG.get(cmd_type, {})
        visible_fields = ui_cfg.get("visible", [])
        labels = ui_cfg.get("labels", {})
        tooltips = ui_cfg.get("tooltip", {})
        if isinstance(tooltips, str): tooltips = {"_root": tooltips} # Handle root tooltip

        # Prepare fields list, respecting 'visible' order if available
        # If 'visible' is empty, fall back to schema order
        schema_fields_map = {f.key: f for f in schema.fields}
        ordered_fields = []

        # Mapping for pseudo-fields in visible list to actual schema keys
        # e.g. input_link/output_link -> links
        processed_schema_keys = set()

        if visible_fields:
            for vk in visible_fields:
                if vk in schema_fields_map:
                    ordered_fields.append(schema_fields_map[vk])
                    processed_schema_keys.add(vk)
                elif vk in ['input_link', 'output_link'] and 'links' in schema_fields_map:
                    # Map pseudo-keys to 'links' field
                    if 'links' not in processed_schema_keys:
                        ordered_fields.append(schema_fields_map['links'])
                        processed_schema_keys.add('links')

            # Add any remaining schema fields that weren't in visible list (fallback)
            for f in schema.fields:
                if f.key not in processed_schema_keys:
                    ordered_fields.append(f)
        else:
            ordered_fields = schema.fields

        # Group optional and up_to fields for horizontal layout
        optional_field = None
        up_to_field = None
        other_fields = []
        
        for field_schema in ordered_fields:
            # Apply Label Override
            if field_schema.key in labels:
                # We clone the schema temporarily or just pass label to creation?
                # FieldSchema is likely shared, so better not modify it.
                # We will handle label override at addRow time.
                pass

            if field_schema.key == 'optional':
                optional_field = field_schema
            elif field_schema.key == 'up_to':
                up_to_field = field_schema
            else:
                other_fields.append(field_schema)
        
        # Add regular fields first
        for field_schema in other_fields:
            # Determine label: Override > Schema Label
            label_text = labels.get(field_schema.key, field_schema.label)
            # Determine tooltip: Override > Schema Tooltip
            tooltip_text = tooltips.get(field_schema.key, field_schema.tooltip)
            self._create_widget_for_field(field_schema, label_override=label_text, tooltip_override=tooltip_text)
        
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
                    # Apply override label for checkbox if supported?
                    # BoolCheckWidget handles its own label.
                    if hasattr(opt_widget, 'setText'):
                        opt_label = labels.get('optional', optional_field.label)
                        opt_widget.setText(tr(opt_label))
                    if hasattr(opt_widget, 'setToolTip'):
                        opt_tooltip = tooltips.get('optional', optional_field.tooltip)
                        if opt_tooltip: opt_widget.setToolTip(tr(opt_tooltip))
                    options_layout.addWidget(opt_widget)
            
            if up_to_field:
                upto_widget = WidgetFactory.create_widget(self, up_to_field, self.update_data)
                if upto_widget:
                    self.widgets_map['up_to'] = upto_widget
                    if hasattr(upto_widget, 'setText'):
                        upto_label = labels.get('up_to', up_to_field.label)
                        upto_widget.setText(tr(upto_label))
                    if hasattr(upto_widget, 'setToolTip'):
                        upto_tooltip = tooltips.get('up_to', up_to_field.tooltip)
                        if upto_tooltip: upto_widget.setToolTip(tr(upto_tooltip))
                    options_layout.addWidget(upto_widget)
            
            options_layout.addStretch()
            self.dynamic_layout.addRow(tr("Options"), options_widget)

    def _create_widget_for_field(self, field_schema: FieldSchema, label_override: str = None, tooltip_override: str = None):
        key = field_schema.key

        # Wrap update_data to also check for auto-generation triggers
        def update_and_trigger():
            self.update_data()
            self._check_auto_generation(key)

        widget = WidgetFactory.create_widget(self, field_schema, update_and_trigger)

        if widget:
            # Do NOT set default value here - only set values when loading actual data
            # This ensures empty/unset fields show "---" until user explicitly sets them
            
            # Set produces_output hint for VariableLinkWidget
            if field_schema.field_type == FieldType.LINK and hasattr(widget, 'set_output_hint'):
                widget.set_output_hint(field_schema.produces_output)
            
            # Apply Tooltip
            if tooltip_override and hasattr(widget, 'setToolTip'):
                widget.setToolTip(tr(tooltip_override))

            self.widgets_map[key] = widget

            label = tr(label_override if label_override else field_schema.label)
            self.dynamic_layout.addRow(label, widget)

    def _check_auto_generation(self, changed_key):
        """Checks if a field change should trigger structure updates (e.g. generating options)."""
        cmd_type = self.type_combo.currentData()

        # If type is CHOICE and amount changes, auto-generate options
        if cmd_type == "CHOICE" and changed_key == "amount":
            self.request_generate_options()
        elif cmd_type == "SELECT_OPTION" and changed_key == "option_count":
            self.request_generate_options()

    def request_generate_options(self):
        """
        Gathers option count from the form and requests structure update.
        Called by OptionsControlWidget or auto-generation logic.
        """
        # Prioritize 'option_count' widget, then fallback to 'amount'
        target_widget = self.widgets_map.get('option_count')
        if not target_widget:
            target_widget = self.widgets_map.get('amount')

        if not target_widget:
             # Fallback: maybe it's named something else or we use default 1
             count = 1
        else:
             # Depending on widget type, get value
             if hasattr(target_widget, 'get_value'):
                 count = target_widget.get_value()
             elif hasattr(target_widget, 'value'):
                 count = target_widget.value()
             else:
                 count = 1

        # Emit signal to CardEditor/LogicTreeWidget
        self.structure_update_requested.emit(STRUCT_CMD_GENERATE_OPTIONS, {"count": count})

    def _load_ui_from_data(self, data, item):
        """Loads data using interface-based widgets."""
        if not data: data = {}
        # Handle case where data is already a CommandModel instance
        if isinstance(data, CommandModel):
            model = data
            # Get dict representation for widgets that need it
            data_dict = model.model_dump()
        else:
            model = CommandModel(**data)
            data_dict = data
        self.current_model = model
        self.current_item = item

        cmd_type = model.type
        # Legacy compatibility: ADD_KEYWORD used str_param in older data
        if cmd_type == "ADD_KEYWORD":
            legacy_kw = model.params.get('str_param')
            if legacy_kw and not model.params.get('str_val'):
                model.params['str_val'] = legacy_kw
        # Mapping back group
        grp = 'OTHER'
        for g, types in COMMAND_GROUPS.items():
            if cmd_type in types:
                grp = g
                break
        
        # Block signals during combo updates to prevent rebuild_dynamic_ui from being called prematurely
        self.action_group_combo.blockSignals(True)
        self.type_combo.blockSignals(True)
        
        try:
            # Set group first (this will repopulate type_combo)
            self.set_combo_by_data(self.action_group_combo, grp)
            # Trigger group change to repopulate type_combo with correct items
            # (must be done with signals still blocked)
            types = COMMAND_GROUPS.get(grp, [])
            self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)
            # Now set the specific type
            self.set_combo_by_data(self.type_combo, cmd_type)
            
            # Now rebuild UI based on the new type
            self.rebuild_dynamic_ui(cmd_type)
        finally:
            self.action_group_combo.blockSignals(False)
            self.type_combo.blockSignals(False)
        
        # Set current_item for VariableLinkWidget
        for key, widget in self.widgets_map.items():
            if key in ['links', 'input_link', 'output_link', 'input_var', 'output_var']:
                if hasattr(widget, 'set_current_item'):
                    widget.set_current_item(item)
        
        # Populate widgets via interface
        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'set_value'):
                # Special handling for flattened models vs structured widgets
                if key in ['links', 'input_link', 'output_link', 'input_var', 'output_var']:
                    # VariableLink expects full dict to parse keys (it handles legacy keys internally usually)
                    # We pass the data_dict which is the dictionary representation
                    widget.set_value(data_dict)
                elif key == 'target_filter':
                    # target_filter is now stored in params
                    tf = model.params.get('target_filter')
                    widget.set_value(tf if tf else {})
                elif key == 'options':
                    widget.set_value(model.options)
                else:
                    # check model attrs first, then params
                    val = getattr(model, key, None)
                    if val is None:
                        val = model.params.get(key)

                    # Only set value if data actually exists (not None and not empty string)
                    # This ensures saved data is displayed, but unsaved fields remain empty ("---")
                    # For boolean/numeric fields, False/0 are valid values, so we check type
                    if val is not None and (isinstance(val, (bool, int, float)) or val != ''):
                        widget.set_value(val)
                    # else: Leave widget at initial state (empty item "---" for combos)

        # Clear validation styles on load
        self._clear_validation_styles()

    def _save_ui_to_data(self, data):
        """Saves data using interface-based widgets and provides validation feedback."""
        # Ensure data is a dict
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        
        cmd_type = self.type_combo.currentData()
        if cmd_type is None:
            cmd_type = "DRAW"  # Default fallback

        # Clear previous validation styles before saving
        self._clear_validation_styles()

        # Start with fresh dict
        new_data = {'type': cmd_type}

        try:
            # Use Pydantic model for validation/structure
            # Initialize with type first
            model = CommandModel(type=cmd_type)

            for key, widget in self.widgets_map.items():
                if hasattr(widget, 'get_value'):
                    val = widget.get_value()

                    if key in ['links', 'input_link', 'output_link', 'input_var', 'output_var']:
                        new_data.update(val)
                        if 'input_var' in val: model.input_var = val['input_var']
                        if 'output_var' in val: model.output_var = val['output_var']
                        if 'input_value_key' in val: model.input_var = val['input_value_key']
                        if 'output_value_key' in val: model.output_var = val['output_value_key']

                    elif key == 'target_filter':
                        if val: model.params['target_filter'] = val
                    else:
                        # Only save non-None values to avoid storing empty selections
                        if val is not None and val != '':
                            if hasattr(model, key):
                                setattr(model, key, val)
                            else:
                                model.params[key] = val

            # Required field checks for grant/keyword actions
            def _get_widget_value(field_key):
                w = self.widgets_map.get(field_key)
                if w and hasattr(w, 'get_value'):
                    return w.get_value()
                return None

            required_missing = []
            if cmd_type == "ADD_KEYWORD":
                kw = _get_widget_value('str_val')
                if not kw:
                    required_missing.append('str_val')
            if cmd_type == "MUTATE":
                mk = _get_widget_value('mutation_kind')
                if not mk:
                    required_missing.append('mutation_kind')

            if required_missing:
                for field_key in required_missing:
                    widget = self.widgets_map.get(field_key)
                    if widget and hasattr(widget, 'setStyleSheet'):
                        widget.setStyleSheet("border: 1px solid red;")
                        if hasattr(widget, 'setToolTip'):
                            widget.setToolTip("必須項目です")
                return

            # Validate manually if needed or catch validation during assignment above if we used setters
            # Since we assigned attrs, Pydantic (if v2 with validate_assignment=True) would raise.
            # If v1 or default, we might need manual validate or model_dump check.

            # Merge model back to dict
            dump = model.model_dump(exclude_none=True)
            new_data.update(dump)

            # Check outputs config
            schema = get_schema(cmd_type)
            if schema:
                produces_output = any(f.produces_output for f in schema.fields)
                if produces_output and 'output_value_key' not in new_data:
                    row = 0
                    if getattr(self, 'current_item', None):
                        row = self.current_item.row()
                    new_data['output_value_key'] = f"var_{cmd_type}_{row}"

            data.clear()
            data.update(new_data)

            # Clear styles on success
            self._clear_validation_styles()

        except ValidationError as e:
            print(f"Validation Error: {e}")
            self._apply_validation_styles(e)
            # We still allow saving partially valid data to avoid data loss,
            # but UI shows error. Or we could abort save.
            # Aborting save usually reverts the change in the tree item which might be confusing.
            # For now, we save what we can but highlight errors.
            data.clear()
            data.update(new_data) # Best effort save

    def _clear_validation_styles(self):
        """Resets the style of all widgets."""
        for widget in self.widgets_map.values():
            if hasattr(widget, 'setStyleSheet'):
                widget.setStyleSheet("")

    def _apply_validation_styles(self, error: ValidationError):
        """Parses ValidationError and highlights invalid widgets."""
        for err in error.errors():
            # err['loc'] is tuple like ('field_name',)
            loc = err.get('loc', [])
            if not loc: continue

            field_name = str(loc[0])
            if field_name in self.widgets_map:
                widget = self.widgets_map[field_name]
                if hasattr(widget, 'setStyleSheet'):
                    # Red border for error
                    widget.setStyleSheet("border: 1px solid red;")
                    widget.setToolTip(err.get('msg', 'Invalid Value'))

    def _get_display_text(self, data):
        t = data.get('type', 'UNKNOWN')
        return f"{tr('Command')}: {tr(t)}"
