# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.consts import TRIGGER_TYPES, SPELL_TRIGGER_TYPES, LAYER_TYPES
from dm_toolkit.gui.editor.forms.parts.keyword_selector import KeywordSelectorWidget
from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler

class EffectEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults for headless/static import
        self.form_layout = getattr(self, 'form_layout', None)
        self.mode_combo = getattr(self, 'mode_combo', None)
        self.trigger_combo = getattr(self, 'trigger_combo', None)
        self.trigger_scope_combo = getattr(self, 'trigger_scope_combo', None)
        self.trigger_filter = getattr(self, 'trigger_filter', None)
        self.layer_group = getattr(self, 'layer_group', None)
        self.layer_type_combo = getattr(self, 'layer_type_combo', None)
        self.target_filter = getattr(self, 'target_filter', None)
        self.condition_widget = getattr(self, 'condition_widget', None)
        try:
            self.setup_ui()
        except Exception:
            pass

    def setup_ui(self):
        self.form_layout = QFormLayout(self)

        # Ability Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("TRIGGERED"), "TRIGGERED")
        self.mode_combo.addItem(tr("STATIC"), "STATIC")
        self.register_widget(self.mode_combo)
        self.add_field(tr("Ability Mode"), self.mode_combo)

        # Trigger Definition
        self.trigger_combo = QComboBox()
        # Initial population, will be updated by Logic Mask
        self.populate_combo(self.trigger_combo, TRIGGER_TYPES, display_func=tr, data_func=lambda x: x)
        self.lbl_trigger = self.add_field(tr("Trigger"), self.trigger_combo, 'trigger')

        # Trigger Scope
        self.trigger_scope_combo = QComboBox()
        scopes = ["NONE", "SELF", "PLAYER_SELF", "PLAYER_OPPONENT", "ALL_PLAYERS"]
        self.populate_combo(self.trigger_scope_combo, scopes, display_func=tr, data_func=lambda x: x)
        self.register_widget(self.trigger_scope_combo, 'trigger_scope')
        self.lbl_scope = self.add_field(tr("Trigger Scope"), self.trigger_scope_combo)

        # Trigger Filter
        self.trigger_filter_group = QGroupBox(tr("Trigger Filter"))
        tf_layout = QGridLayout(self.trigger_filter_group)
        self.trigger_filter = UnifiedFilterHandler.create_filter_widget("TRIGGER", self)
        self.trigger_filter.filterChanged.connect(self.update_data)
        self.register_widget(self.trigger_filter, 'trigger_filter')
        tf_layout.addWidget(self.trigger_filter, 0, 0)
        self.add_field(None, self.trigger_filter_group)

        # Layer Definition (Static)
        self.layer_group = QGroupBox(tr("Layer Definition"))
        l_layout = QGridLayout(self.layer_group)

        self.layer_type_combo = QComboBox()
        self.populate_combo(self.layer_type_combo, LAYER_TYPES, display_func=tr, data_func=lambda x: x)
        self.register_widget(self.layer_type_combo, 'type')

        self.layer_val_spin = QSpinBox()
        self.layer_val_spin.setRange(-9999, 9999)
        self.register_widget(self.layer_val_spin, 'value')

        self.layer_str_edit = QLineEdit()
        self.register_widget(self.layer_str_edit, 'str_val')
        
        # Keyword Helper - Unified Widget
        self.layer_keyword_combo = KeywordSelectorWidget(allow_settable=True)
        self.layer_keyword_combo.keywordSelected.connect(self.on_layer_keyword_changed)

        l_layout.addWidget(QLabel(tr("Layer Type")), 0, 0)
        l_layout.addWidget(self.layer_type_combo, 0, 1)
        l_layout.addWidget(QLabel(tr("Value")), 1, 0)
        l_layout.addWidget(self.layer_val_spin, 1, 1)
        l_layout.addWidget(QLabel(tr("String/Keyword")), 2, 0)
        l_layout.addWidget(self.layer_str_edit, 2, 1)
        l_layout.addWidget(QLabel(tr("Select Keyword")), 3, 0)
        l_layout.addWidget(self.layer_keyword_combo, 3, 1)

        # Target Filter - Unified Handler
        self.target_filter = UnifiedFilterHandler.create_filter_widget("STATIC", self)
        self.target_filter.filterChanged.connect(self.update_data)
        self.register_widget(self.target_filter, 'filter')
        l_layout.addWidget(QLabel(tr("Target Filter")), 4, 0)
        l_layout.addWidget(self.target_filter, 4, 1)

        self.add_field(None, self.layer_group)

        # Condition (Shared)
        self.condition_widget = ConditionEditorWidget()
        self.condition_widget.dataChanged.connect(self.update_data)
        self.add_field(None, self.condition_widget, 'condition')

        # Actions Section
        self.add_action_btn = QPushButton(tr("Add Command"))
        self.add_action_btn.clicked.connect(self.on_add_action_clicked)
        self.add_field(None, self.add_action_btn)

        # Connect signals
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo.currentIndexChanged.connect(self.update_data)

        self.trigger_combo.currentIndexChanged.connect(self.update_data)
        self.trigger_scope_combo.currentIndexChanged.connect(self.update_data)

        self.layer_type_combo.currentIndexChanged.connect(self.update_data)
        self.layer_type_combo.currentIndexChanged.connect(self.update_layer_keyword_visibility)
        self.layer_val_spin.valueChanged.connect(self.update_data)
        self.layer_str_edit.textChanged.connect(self.update_data)

        # Initial UI State
        self.on_mode_changed()

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        if mode is None:
            mode = "TRIGGERED"  # Default fallback
        is_triggered = (mode == "TRIGGERED")

        self.trigger_combo.setVisible(is_triggered)
        self.lbl_trigger.setVisible(is_triggered)

        self.trigger_scope_combo.setVisible(is_triggered)
        self.lbl_scope.setVisible(is_triggered)

        self.trigger_filter_group.setVisible(is_triggered)

        self.layer_group.setVisible(not is_triggered)

        if is_triggered:
            self.condition_widget.setTitle(tr("Trigger Condition"))
        else:
            self.condition_widget.setTitle(tr("Apply Condition"))
        
        # Update keyword combo visibility
        self.update_layer_keyword_visibility()
    
    def on_layer_keyword_changed(self, keyword: str):
        """Update str_val when keyword is selected from unified widget."""
        if keyword:
            self.layer_str_edit.setText(keyword)
        self.update_data()
    
    def update_layer_keyword_visibility(self):
        """Show keyword combo only for GRANT_KEYWORD or SET_KEYWORD types"""
        layer_type = self.layer_type_combo.currentData() if hasattr(self, 'layer_type_combo') else None
        show_keyword = layer_type in ['GRANT_KEYWORD', 'SET_KEYWORD']
        if hasattr(self, 'layer_keyword_combo'):
            self.layer_keyword_combo.setVisible(show_keyword)

    def on_add_action_clicked(self):
        self.structure_update_requested.emit("ADD_CHILD_ACTION", {})

    def _load_ui_from_data(self, data, item):
        """
        Populate UI from data (Hook).
        """
        item_type = "EFFECT"
        if item:
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)

            # Logic Mask: Filter triggers based on Card Type
            card_type = "CREATURE"
            parent = item.parent() # Group
            if parent:
                grandparent = parent.parent() # Card or Spell Side
                if grandparent:
                    role = grandparent.data(Qt.ItemDataRole.UserRole + 1)
                    if role == "SPELL_SIDE":
                        card_type = "SPELL"
                    elif role == "CARD":
                         cdata = grandparent.data(Qt.ItemDataRole.UserRole + 2)
                         card_type = cdata.get('type', 'CREATURE')

            self.update_trigger_options(card_type)

        # Determine Mode
        mode = "TRIGGERED"
        if item_type == "MODIFIER":
            mode = "STATIC"
        elif 'layer_type' in data or 'type' in data and item_type != "EFFECT":
            # Legacy check or inferred from data
            mode = "STATIC"

        self.set_combo_by_data(self.mode_combo, mode)
        # Trigger visibility update immediately
        self.on_mode_changed()

        if mode == "TRIGGERED":
             # Try to normalize data for binding if legacy keys exist
             if 'trigger_condition' in data and 'condition' not in data:
                 data['condition'] = data['trigger_condition']

             # Load Trigger Filter explicitly
             if 'trigger_filter' in data and self.trigger_filter:
                 self.trigger_filter.set_filter_data(data['trigger_filter'])
             else:
                 self.trigger_filter.set_filter_data({})
        else:
            # STATIC (ModifierDef) - Normalize for bindings
            if 'layer_type' in data: data['type'] = data['layer_type']
            if 'layer_value' in data: data['value'] = data['layer_value']
            if 'layer_str' in data: data['str_val'] = data['layer_str']
            if 'static_condition' in data and 'condition' not in data:
                 data['condition'] = data['static_condition']

        # Use Bindings
        self._apply_bindings(data)
        
        # Set keyword combo based on str_val for STATIC mode
        if mode == "STATIC":
            str_val = data.get('str_val', '')
            if str_val and hasattr(self, 'layer_keyword_combo'):
                self.set_combo_by_data(self.layer_keyword_combo, str_val)

        # Ensure fallback for condition if missing
        if not data.get('condition'):
             self.condition_widget.set_data({})
        
        # Update keyword combo visibility
        self.update_layer_keyword_visibility()

    def update_trigger_options(self, card_type):
        is_spell = (card_type == "SPELL")

        allowed = SPELL_TRIGGER_TYPES if is_spell else TRIGGER_TYPES

        current_data = self.trigger_combo.currentData()
        self.trigger_combo.blockSignals(True)
        self.trigger_combo.clear()

        # Ensure current data is preserved if it was valid before or legacy
        if current_data and current_data not in allowed:
            allowed.append(current_data)

        self.populate_combo(self.trigger_combo, allowed, display_func=tr, data_func=lambda x: x)

        # Restore selection
        idx = self.trigger_combo.findData(current_data)
        if idx >= 0:
            self.trigger_combo.setCurrentIndex(idx)
        else:
            self.trigger_combo.setCurrentIndex(0)

        self.trigger_combo.blockSignals(False)

    def _save_ui_to_data(self, data):
        """
        Save UI to data (Hook).
        """
        mode = self.mode_combo.currentData()

        # Apply bindings (collects into data)
        self._collect_bindings(data)

        # Post-processing based on Mode
        if self.current_item:
             current_type_code = self.current_item.data(Qt.ItemDataRole.UserRole + 1)
             target_type_code = "EFFECT" if mode == "TRIGGERED" else "MODIFIER"

             # Update type if changed
             if current_type_code != target_type_code:
                 self.current_item.setData(target_type_code, Qt.ItemDataRole.UserRole + 1)
                 # Emit MOVE_EFFECT to trigger UI/Label updates in the tree
                 self.structure_update_requested.emit("MOVE_EFFECT", {"item": self.current_item, "target_type": mode})

        if mode == "TRIGGERED":
            # Explicitly save trigger filter from widget (bindings might not catch it if it's complex/custom getter)
            data['trigger_filter'] = self.trigger_filter.get_filter_data()

            # Clean Static/Legacy keys
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        else: # STATIC
            # Handle str_val optionality
            if not self.layer_str_edit.text():
                data.pop('str_val', None)

            # Clean Trigger/Legacy keys
            for k in ['trigger', 'trigger_scope', 'trigger_filter', 'trigger_condition', 'layer_type', 'layer_value', 'layer_str', 'static_condition']:
                data.pop(k, None)

    def _get_display_text(self, data):
        if 'trigger' in data:
             scope = data.get('trigger_scope', 'NONE')
             scope_str = "" if scope == "NONE" else f" ({tr(scope)})"
             return f"{tr('Effect')}: {tr(data.get('trigger', ''))}{scope_str}"
        elif 'type' in data or 'layer_type' in data:
             t = data.get('type', data.get('layer_type', ''))
             return f"{tr('Static')}: {tr(t)}"
        else:
             return tr("Unknown Effect")
