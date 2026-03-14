# -*- coding: utf-8 -*-
# unified_action_form.py
# \x83X\x83L\x81[\x83}\x8b쓮\x82̓\xae\x93I\x83t\x83H\x81[\x83\x80\x90\xb6\x90\xac\x81B\x95\xa1\x8eG\x82\xbe\x82\xaa\x81A\x92P\x88\xea\x82̐Ӗ\xb1\x81i\x83A\x83N\x83V\x83\x87\x83\x93\x90ݒ\xe8\x81j\x82ɏW\x92\x86\x82\xb5\x82Ă\xa2\x82\xe9\x81B
# \x88ێ\x9d: AI\x82\xaa\x95ҏW\x82\xb7\x82\xe9\x8dۂ\xcd schema_def.py \x82\xe2 ACTION_UI_CONFIG (command_config.py) \x82ƃZ\x83b\x83g\x82ŗ\x9d\x89\xf0\x82\xb7\x82\xe9\x95K\x97v\x82\xaa\x82\xa0\x82\xe8\x82܂\xb7\x81B
# \x8fd\x97v: \x82\xb1\x82̃t\x83@\x83C\x83\x8b\x82͍폜\x82\xb5\x82Ȃ\xa2\x82ł\xad\x82\xbe\x82\xb3\x82\xa2\x81Bschema_def.py \x82\xa8\x82\xe6\x82\xd1 command_config.py \x82Ɩ\xa7\x90ڂɘA\x8cg\x82\xb5\x82ē\xae\x8d삵\x82܂\xb7\x81B

import json
import os
from typing import Any
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
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
from dm_toolkit.gui.editor.schema_def import SchemaLoader, get_schema, FieldSchema, FieldType
from dm_toolkit.gui.editor.consts import STRUCT_CMD_GENERATE_OPTIONS
from dm_toolkit.gui.editor.forms.diff_tree_widget import DiffTreeWidget
from dm_toolkit.gui.editor.consistency import format_integrity_warnings, validate_command_list

COMMAND_GROUPS = EditorConfigLoader.get_command_groups()

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
        safe_connect(self.action_group_combo, 'currentIndexChanged', self.on_group_changed)

        # Type Combo
        self.type_combo = QComboBox()
        self.main_layout.addRow(tr("Command Type"), self.type_combo)
        safe_connect(self.type_combo, 'currentIndexChanged', self.on_type_changed)

        # Dynamic Content Container
        self.dynamic_container = QWidget()
        self.dynamic_layout = QFormLayout(self.dynamic_container)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addRow(self.dynamic_container)

        # CIR summary + action (minimal integration)
        cir_row = QHBoxLayout()
        self.cir_label = QLabel("")
        self.cir_label.setVisible(False)
        cir_row.addWidget(self.cir_label)
        self.apply_cir_btn = QPushButton(tr("Apply CIR"))
        self.apply_cir_btn.setEnabled(False)
        safe_connect(self.apply_cir_btn, 'clicked', self.on_apply_cir)
        cir_row.addWidget(self.apply_cir_btn)
        cir_row.addStretch()
        self.main_layout.addRow(cir_row)

        # Diff tree widget (hidden until CIR present)
        self.diff_tree_widget = DiffTreeWidget()
        self.diff_tree_widget.setVisible(False)
        self.main_layout.addRow(tr("Diff"), self.diff_tree_widget)

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
            
            self.widgets_map[key] = widget
            self.dynamic_layout.addRow(tr(field_schema.label), widget)

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
        # Clear any previous diff highlights before rebuilding/loading
        try:
            self.clear_diff_highlight()
        except Exception:
            pass

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

        # Minimal CIR UI integration: show count and enable apply button
        try:
            cir = None
            if item is not None and hasattr(item, 'data'):
                cir = item.data('ROLE_CIR')
            if cir:
                self.cir_label.setText(tr("CIR entries: {n}").format(n=len(cir)))
                self.cir_label.setVisible(True)
                self.apply_cir_btn.setEnabled(True)
                # tooltip for debugging
                # show tooltip with CIR and shallow diff summary
                try:
                    # Use structural diff for richer nested/key path summaries
                    # Use formatted structural diff for tooltip (multiline)
                    summary = self.format_structural_diff(cir[0].get('payload', cir[0]))
                    tip = str(cir)
                    if summary:
                        tip += '\nDiff:\n' + summary
                    self.cir_label.setToolTip(tip)
                except Exception:
                    self.cir_label.setToolTip(str(cir))
                # Highlight differences between current UI/model and first CIR payload
                try:
                    first = cir[0]
                    if isinstance(first, dict):
                        self.highlight_diff(first.get('payload', first))
                        try:
                            # attach structured diff tree to the diff widget
                            if hasattr(self, 'diff_tree_widget'):
                                tree = self.compute_structural_diff_tree(first.get('payload', first))
                                self.diff_tree_widget.set_diff_tree(tree)
                                self.diff_tree_widget.setVisible(bool(tree))
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                self.cir_label.setVisible(False)
                self.apply_cir_btn.setEnabled(False)
        except Exception:
            self.cir_label.setVisible(False)
            self.apply_cir_btn.setEnabled(False)

    def on_apply_cir(self):
        """Handler for Apply CIR button - emits an action for higher layers to consume.

        Currently this performs a best-effort emit with the CIR payload; actual
        model population logic can be implemented iteratively in follow-up tasks.
        """
        try:
            cir = None
            if self.current_item is not None and hasattr(self.current_item, 'data'):
                cir = self.current_item.data('ROLE_CIR')
            if cir:
                # Emit for CardEditor or other listeners to act upon
                self.structure_update_requested.emit('APPLY_CIR', {'cir': cir})
        except Exception:
            pass

    def highlight_diff(self, cir_payload: dict[str, Any]) -> None:
        """Mark widgets whose current value differs from CIR payload.

        This is a best-effort shallow comparison keyed by param name.
        Widgets are expected to be accessible via `self.widgets_map[name]`
        and expose `get_value()` and `setStyleSheet()`.
        """
        if not cir_payload:
            return
        for key, widget in getattr(self, 'widgets_map', {}).items():
            try:
                widget_val = None
                # prefer widget.get_value() if available
                if hasattr(widget, 'get_value'):
                    widget_val = widget.get_value()
                else:
                    widget_val = getattr(widget, 'value', None)

                cir_val = cir_payload.get(key)
                if cir_val is None:
                    # no comparison value; clear highlight
                    if hasattr(widget, 'setStyleSheet'):
                        widget.setStyleSheet('')
                    continue

                if widget_val != cir_val:
                    if hasattr(widget, 'setStyleSheet'):
                        widget.setStyleSheet('background: yellow;')
                else:
                    if hasattr(widget, 'setStyleSheet'):
                        widget.setStyleSheet('')
            except Exception:
                # best-effort: ignore widget failures
                continue

    def clear_diff_highlight(self) -> None:
        """Clear any diff highlights on managed widgets."""
        for widget in getattr(self, 'widgets_map', {}).values():
            try:
                if hasattr(widget, 'setStyleSheet'):
                    widget.setStyleSheet('')
            except Exception:
                continue
        try:
            if hasattr(self, 'diff_tree_widget'):
                self.diff_tree_widget.set_diff_tree({})
                self.diff_tree_widget.setVisible(False)
        except Exception:
            pass

    def compute_diff_summary(self, cir_payload: dict[str, Any]) -> list[str]:
        """Return list of keys where cir_payload and current widget/model differ.

        - Keys present in payload but missing in widgets are included (as 'extra').
        - Keys present in widgets but missing in payload are not considered changed.
        - Shallow comparison only.
        """
        diffs: list[str] = []
        if not cir_payload:
            return diffs

        # compare keys present in payload
        for k, v in cir_payload.items():
            w = getattr(self, 'widgets_map', {}).get(k)
            try:
                if w is None:
                    diffs.append(k)
                    continue
                if hasattr(w, 'get_value'):
                    wval = w.get_value()
                else:
                    wval = getattr(w, 'value', None)

                if wval != v:
                    diffs.append(k)
            except Exception:
                diffs.append(k)

        return diffs

    def compute_structural_diff(self, cir_payload: dict[str, Any]) -> list[str]:
        """Return flattened paths of keys where cir_payload and current widget/model differ.

        Examples of returned paths:
        - 'target_filter.cost' for nested dicts
        - 'options[1].label' for lists of dicts
        - top-level keys that have no corresponding widget will be returned as-is
        """
        diffs: list[str] = []
        if not cir_payload:
            return diffs

        def _get_widget_value_for_key(key: str):
            w = getattr(self, 'widgets_map', {}).get(key)
            if w is None:
                return None, False
            try:
                if hasattr(w, 'get_value'):
                    return w.get_value(), True
                return getattr(w, 'value', None), True
            except Exception:
                return None, True

        def _compare(prefix: str, payload_val, widget_val):
            # payload_val exists; widget_val may be None meaning missing widget or missing value
            if isinstance(payload_val, dict):
                if not isinstance(widget_val, dict):
                    # widget missing or not a dict: report full prefix
                    diffs.append(prefix)
                    return
                for k, v in payload_val.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    _compare(new_prefix, v, widget_val.get(k))
            elif isinstance(payload_val, list):
                if not isinstance(widget_val, list):
                    diffs.append(prefix)
                    return
                # compare element-wise; if lengths differ, mark extra indices as diffs
                min_len = min(len(payload_val), len(widget_val))
                for i in range(min_len):
                    pv = payload_val[i]
                    wv = widget_val[i]
                    new_prefix = f"{prefix}[{i}]"
                    if isinstance(pv, dict):
                        if not isinstance(wv, dict):
                            diffs.append(new_prefix)
                        else:
                            for kk, vv in pv.items():
                                _compare(f"{new_prefix}.{kk}", vv, wv.get(kk))
                    else:
                        if pv != wv:
                            diffs.append(new_prefix)
                # extra payload elements are considered diffs
                if len(payload_val) > len(widget_val):
                    for i in range(min_len, len(payload_val)):
                        diffs.append(f"{prefix}[{i}]")
            else:
                # primitive comparison
                if widget_val != payload_val:
                    diffs.append(prefix)

        # top-level: iterate payload keys and fetch widget values
        for key, val in cir_payload.items():
            widget_val, has_widget = _get_widget_value_for_key(key)
            if not has_widget:
                # no widget present for this key: include it
                diffs.append(key)
                continue
            _compare(key, val, widget_val)

        return diffs

    def compute_structural_diff_tree(self, cir_payload: dict[str, Any]) -> dict:
        """Return nested dict marking paths that differ between payload and widgets.

        Leaves are True to indicate a difference. Example:
        { 'target_filter': { 'cost': True }, 'options': { 1: { 'label': True } }, 'extra': True }
        """
        tree: dict = {}
        if not cir_payload:
            return tree

        def _set_path(t: dict, parts: list, value=True):
            if not parts:
                return
            key = parts[0]
            if len(parts) == 1:
                t[key] = value
                return
            if key not in t or not isinstance(t[key], dict):
                t[key] = {}
            _set_path(t[key], parts[1:], value)

        def _get_widget_value_for_key(key: str):
            w = getattr(self, 'widgets_map', {}).get(key)
            if w is None:
                return None, False
            try:
                if hasattr(w, 'get_value'):
                    return w.get_value(), True
                return getattr(w, 'value', None), True
            except Exception:
                return None, True

        def _recurse(prefix_parts: list, payload_val, widget_val):
            if isinstance(payload_val, dict):
                if not isinstance(widget_val, dict):
                    _set_path(tree, prefix_parts)
                    return
                for k, v in payload_val.items():
                    _recurse(prefix_parts + [k], v, widget_val.get(k))
            elif isinstance(payload_val, list):
                if not isinstance(widget_val, list):
                    _set_path(tree, prefix_parts)
                    return
                min_len = min(len(payload_val), len(widget_val))
                for i in range(min_len):
                    pv = payload_val[i]
                    wv = widget_val[i]
                    if isinstance(pv, dict):
                        for kk, vv in pv.items():
                            _recurse(prefix_parts + [i, kk], vv, (wv.get(kk) if isinstance(wv, dict) else None))
                    else:
                        if pv != wv:
                            _set_path(tree, prefix_parts + [i])
                if len(payload_val) > len(widget_val):
                    for i in range(min_len, len(payload_val)):
                        _set_path(tree, prefix_parts + [i])
            else:
                if widget_val != payload_val:
                    _set_path(tree, prefix_parts)

        for key, val in cir_payload.items():
            widget_val, has_widget = _get_widget_value_for_key(key)
            if not has_widget:
                tree[key] = True
                continue
            _recurse([key], val, widget_val)

        return tree

    def format_structural_diff(self, cir_payload: dict[str, Any]) -> str:
        """Return a human-readable multiline summary for structural diffs.

        Each changed path is on its own line. For lists/dicts the path format
        follows compute_structural_diff (e.g. 'options[1].label').
        """
        paths = self.compute_structural_diff(cir_payload)
        if not paths:
            return ''
        # Sort for deterministic output
        paths = sorted(paths)
        return '\n'.join(paths)

    def apply_cir(self, cir_list: list) -> bool:
        """Apply a canonical IR payload to the current form.

        This is a best-effort mapper: it takes the first CIR entry, maps its
        `type` to the form's type selector and copies any payload keys into
        corresponding widgets (via `set_value`) and into `current_model.params`.
        Returns True if any widget/model was updated.
        """
        if not cir_list:
            return False
        cir = cir_list[0]
        updated = False
        try:
            # Set type if present
            ctype = cir.get('type') or cir.get('kind')
            if ctype:
                try:
                    self.set_combo_by_data(self.action_group_combo, 'OTHER')
                    self.set_combo_by_data(self.type_combo, ctype)
                except Exception:
                    # best-effort: ignore if combo helpers unavailable
                    pass

            payload = cir.get('payload') or cir.get('params') or {}
            if not isinstance(payload, dict):
                payload = {}

            # Ensure current_model exists
            if self.current_model is None:
                try:
                    self.current_model = CommandModel(type=ctype if ctype else 'UNKNOWN')
                except Exception:
                    pass

            for k, v in payload.items():
                # Update model.params
                try:
                    if not hasattr(self.current_model, 'params'):
                        self.current_model.params = {}
                    self.current_model.params[k] = v
                    updated = True
                except Exception:
                    pass

                # Update widget if present
                w = self.widgets_map.get(k)
                if w and hasattr(w, 'set_value'):
                    try:
                        w.set_value(v)
                        updated = True
                    except Exception:
                        pass
            # After applying values, clear any diff highlights (best-effort)
            try:
                self.clear_diff_highlight()
            except Exception:
                pass
        except Exception:
            return False
        return updated

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
                        # Normalize legacy link keys to canonical names for saving.
                        in_key = None
                        out_key = None
                        if isinstance(val, dict):
                            in_key = val.get('input_value_key') or val.get('input_var') or val.get('input_key')
                            out_key = val.get('output_value_key') or val.get('output_var') or val.get('output_key')

                        if in_key:
                            new_data['input_value_key'] = in_key
                            model.input_var = in_key
                        if out_key:
                            new_data['output_value_key'] = out_key
                            model.output_var = out_key

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
                            widget.setToolTip(tr("必須項目です"))
                return

            # Validate manually if needed or catch validation during assignment above if we used setters
            # Since we assigned attrs, Pydantic (if v2 with validate_assignment=True) would raise.
            # If v1 or default, we might need manual validate or model_dump check.

            # Auto-map "COST" usage to filter cost_ref
            # input_value_usage is in new_data (from links widget)
            usage = new_data.get('input_value_usage')
            key_var = new_data.get('input_value_key') or new_data.get('input_var')

            if usage == 'COST' and key_var:
                if 'target_filter' in model.params:
                    tf = model.params['target_filter']
                    if isinstance(tf, dict):
                        tf['cost_ref'] = key_var
                        model.params['target_filter'] = tf

                # When using a dynamic cost reference (e.g. from selected number), target ALL matching cards by default
                # 255 is the internal constant for AMOUNT_ALL
                if hasattr(model, 'amount'):
                    model.amount = 255
                else:
                    model.params['amount'] = 255

            # Merge model back to dict
            dump = model.model_dump(exclude_none=True)
            new_data.update(dump)

            # Ensure legacy keys are not persisted; prefer canonical key names
            for legacy in ('input_var', 'output_var'):
                if legacy in new_data:
                    del new_data[legacy]

            # Check outputs config
            # 再発防止: output_value_key は全コマンドに uid ベースで自動生成する。
            # row ベースはユニーク性が保証できないため废止。
            # produces_output=False のコマンドもキーを持つが UI には表示しない。
            schema = get_schema(cmd_type)
            if 'output_value_key' not in new_data or not new_data.get('output_value_key'):
                uid_val = new_data.get('uid', '')
                if uid_val:
                    new_data['output_value_key'] = f"var_{uid_val[:8]}"
                elif schema and any(f.produces_output for f in schema.fields):
                    # uidなしかった場合のフォールバック
                    row = getattr(self.current_item, 'row', lambda: 0)()
                    new_data['output_value_key'] = f"var_{cmd_type}_{row}"

            # コマンド整合性チェック（警告が致命的な場合は保存を中止）
            try:
                warns = validate_command_list([new_data], _path=cmd_type)
                if warns:
                    # Emit warnings for UI/console listeners
                    self.structure_update_requested.emit("INTEGRITY_WARNINGS", {"warnings": warns})
                    formatted_warns = format_integrity_warnings(warns)
                    # Apply simple validation styles to indicate problem fields
                    for widget in self.widgets_map.values():
                        if hasattr(widget, 'setStyleSheet'):
                            widget.setStyleSheet("border: 1px solid red;")
                            if hasattr(widget, 'setToolTip'):
                                # 再発防止: Tooltip も共通フォーマッタを使い、UI 表示の揺れを防ぐ。
                                widget.setToolTip(formatted_warns)
                    # Abort save to avoid persisting invalid command
                    return
            except Exception:
                # Don't let validation exceptions block saving in odd test/runtime setups
                pass

            # No blocking warnings: commit data and clear styles
            data.clear()
            data.update(new_data)
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
                    widget.setToolTip(tr(err.get('msg', 'Invalid Value')))

    def _get_display_text(self, data):
        t = data.get('type', 'UNKNOWN')
        return f"{tr('Command')}: {tr(t)}"
