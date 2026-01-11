# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout,
    QSpinBox, QLabel, QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.gui.editor.forms.effect_config import EFFECT_UI_CONFIG, TRIGGER_TYPES, LAYER_TYPES

class EffectEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults for headless/static import
        self.form_layout = getattr(self, 'form_layout', None)
        self.mode_combo = getattr(self, 'mode_combo', None)
        self.trigger_combo = getattr(self, 'trigger_combo', None)
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
        # Populated from Config
        self.populate_combo(self.trigger_combo, TRIGGER_TYPES, display_func=tr, data_func=lambda x: x)
        self.lbl_trigger = self.add_field(tr("Trigger"), self.trigger_combo, 'trigger')

        # Layer Definition (Static)
        self.layer_group = QGroupBox(tr("Layer Definition"))
        l_layout = QGridLayout(self.layer_group)

        self.layer_type_combo = QComboBox()
        self.populate_combo(self.layer_type_combo, LAYER_TYPES, display_func=tr, data_func=lambda x: x)
        self.register_widget(self.layer_type_combo, 'type')

        self.layer_val_spin = QSpinBox()
        self.layer_val_spin.setRange(-9999, 9999)
        self.register_widget(self.layer_val_spin, 'value')
        self.lbl_layer_val = QLabel(tr("Value"))

        self.layer_str_edit = QLineEdit()
        self.register_widget(self.layer_str_edit, 'str_val')
        self.lbl_layer_str = QLabel(tr("String/Keyword"))
        
        # Keyword Helper ComboBox for GRANT_KEYWORD
        self.layer_keyword_combo = QComboBox()
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        from dm_toolkit.consts import GRANTABLE_KEYWORDS
        keyword_items = [(kw, CardTextGenerator.KEYWORD_TRANSLATION.get(kw, kw)) for kw in GRANTABLE_KEYWORDS]
        for kw_val, kw_display in keyword_items:
            self.layer_keyword_combo.addItem(kw_display, kw_val)
        self.layer_keyword_combo.currentIndexChanged.connect(self.on_layer_keyword_changed)
        self.lbl_layer_keyword = QLabel(tr("Select Keyword"))

        # Layout for Layer Group
        # Row 0: Type
        l_layout.addWidget(QLabel(tr("Layer Type")), 0, 0)
        l_layout.addWidget(self.layer_type_combo, 0, 1)

        # Row 1: Value
        l_layout.addWidget(self.lbl_layer_val, 1, 0)
        l_layout.addWidget(self.layer_val_spin, 1, 1)

        # Row 2: String
        l_layout.addWidget(self.lbl_layer_str, 2, 0)
        l_layout.addWidget(self.layer_str_edit, 2, 1)

        # Row 3: Keyword Combo
        l_layout.addWidget(self.lbl_layer_keyword, 3, 0)
        l_layout.addWidget(self.layer_keyword_combo, 3, 1)

        # Target Filter (Static)
        self.target_filter = FilterEditorWidget()
        self.target_filter.filterChanged.connect(self.update_data)
        self.target_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        self.register_widget(self.target_filter, 'filter')

        # Row 4: Filter
        self.lbl_filter = QLabel(tr("Target Filter"))
        l_layout.addWidget(self.lbl_filter, 4, 0)
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

        self.trigger_combo.currentIndexChanged.connect(self.on_trigger_changed)
        self.trigger_combo.currentIndexChanged.connect(self.update_data)

        self.layer_type_combo.currentIndexChanged.connect(self.on_layer_type_changed)
        self.layer_type_combo.currentIndexChanged.connect(self.update_data)

        self.layer_val_spin.valueChanged.connect(self.update_data)
        self.layer_str_edit.textChanged.connect(self.update_data)

        # Initial UI State
        self.on_mode_changed()

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_triggered = (mode == "TRIGGERED")

        # Top-level visibility
        self.trigger_combo.setVisible(is_triggered)
        self.lbl_trigger.setVisible(is_triggered)
        self.layer_group.setVisible(not is_triggered)

        if is_triggered:
            self.condition_widget.setTitle(tr("Trigger Condition"))
            self.on_trigger_changed() # Update fields based on selected trigger
        else:
            self.condition_widget.setTitle(tr("Apply Condition"))
            self.on_layer_type_changed() # Update fields based on selected layer

    def on_trigger_changed(self):
        key = self.trigger_combo.currentData()
        self.update_field_visibility(key)

    def on_layer_type_changed(self):
        key = self.layer_type_combo.currentData()
        self.update_field_visibility(key)

    def update_field_visibility(self, type_key):
        """
        Updates visibility of fields based on EFFECT_UI_CONFIG.
        """
        cfg = EFFECT_UI_CONFIG.get(type_key, {})
        visible_fields = set(cfg.get('visible', []))
        labels = cfg.get('labels', {})

        # Value
        show_val = 'value' in visible_fields
        self.layer_val_spin.setVisible(show_val)
        self.lbl_layer_val.setVisible(show_val)
        if show_val:
            self.lbl_layer_val.setText(tr(labels.get('value', 'Value')))

        # String
        show_str = 'str_val' in visible_fields
        self.layer_str_edit.setVisible(show_str)
        self.lbl_layer_str.setVisible(show_str)
        if show_str:
            self.lbl_layer_str.setText(tr(labels.get('str_val', 'String/Keyword')))

        # Keyword Combo
        show_kw = 'keyword' in visible_fields
        self.layer_keyword_combo.setVisible(show_kw)
        self.lbl_layer_keyword.setVisible(show_kw)

        # Filter
        show_filt = 'filter' in visible_fields
        self.target_filter.setVisible(show_filt)
        self.lbl_filter.setVisible(show_filt)

        # Condition
        # Condition widget is almost always visible, but configurable
        self.condition_widget.setVisible('condition' in visible_fields)

    def on_layer_keyword_changed(self):
        kw_val = self.layer_keyword_combo.currentData()
        if kw_val:
            self.layer_str_edit.setText(kw_val)
        self.update_data()
    
    def on_add_action_clicked(self):
        self.structure_update_requested.emit("ADD_CHILD_ACTION", {})

    def _load_ui_from_data(self, data, item):
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
            mode = "STATIC"

        self.set_combo_by_data(self.mode_combo, mode)
        # Visibility will be updated by on_mode_changed -> on_X_changed

        if mode == "TRIGGERED":
             if 'trigger_condition' in data and 'condition' not in data:
                 data['condition'] = data['trigger_condition']
        else:
            if 'layer_type' in data: data['type'] = data['layer_type']
            if 'layer_value' in data: data['value'] = data['layer_value']
            if 'layer_str' in data: data['str_val'] = data['layer_str']
            if 'static_condition' in data and 'condition' not in data:
                 data['condition'] = data['static_condition']

        self._apply_bindings(data)
        
        # Set keyword combo if applicable
        if mode == "STATIC":
            str_val = data.get('str_val', '')
            if str_val and hasattr(self, 'layer_keyword_combo'):
                self.set_combo_by_data(self.layer_keyword_combo, str_val)

        if not data.get('condition'):
             self.condition_widget.set_data({})

    def update_trigger_options(self, card_type):
        is_spell = (card_type == "SPELL")

        # This Logic Mask could be moved to Registry later
        allowed = []
        if is_spell:
            allowed = ["ON_PLAY", "ON_CAST_SPELL", "TURN_START", "ON_OPPONENT_DRAW", "PASSIVE_CONST", "ON_OTHER_ENTER"]
        else:
            allowed = TRIGGER_TYPES # Use imported list

        current_data = self.trigger_combo.currentData()
        self.trigger_combo.blockSignals(True)
        self.trigger_combo.clear()

        # Preserve current if valid or legacy
        if current_data and current_data not in allowed:
            allowed.append(current_data)

        self.populate_combo(self.trigger_combo, allowed, display_func=tr, data_func=lambda x: x)

        idx = self.trigger_combo.findData(current_data)
        if idx >= 0:
            self.trigger_combo.setCurrentIndex(idx)
        else:
            self.trigger_combo.setCurrentIndex(0)

        self.trigger_combo.blockSignals(False)

    def _save_ui_to_data(self, data):
        mode = self.mode_combo.currentData()
        self._collect_bindings(data)

        if self.current_item:
             current_type_code = self.current_item.data(Qt.ItemDataRole.UserRole + 1)
             target_type_code = "EFFECT" if mode == "TRIGGERED" else "MODIFIER"

             if current_type_code != target_type_code:
                 self.current_item.setData(target_type_code, Qt.ItemDataRole.UserRole + 1)
                 self.structure_update_requested.emit("MOVE_EFFECT", {"item": self.current_item, "target_type": mode})

        # Cleanup keys based on Mode
        if mode == "TRIGGERED":
            # Clean Static/Legacy keys
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        else: # STATIC
            if not self.layer_str_edit.text():
                data.pop('str_val', None)
            for k in ['trigger', 'trigger_condition', 'layer_type', 'layer_value', 'layer_str', 'static_condition']:
                data.pop(k, None)

    def _get_display_text(self, data):
        if 'trigger' in data:
             return f"{tr('Effect')}: {tr(data.get('trigger', ''))}"
        elif 'type' in data or 'layer_type' in data:
             t = data.get('type', data.get('layer_type', ''))
             return f"{tr('Static')}: {tr(t)}"
        else:
             return tr("Unknown Effect")
