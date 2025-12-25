# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget

class EffectEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.form_layout = QFormLayout(self)

        # Ability Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("TRIGGERED"), "TRIGGERED")
        self.mode_combo.addItem(tr("STATIC"), "STATIC")
        self.add_field(tr("Ability Mode"), self.mode_combo)

        # Trigger Definition
        self.trigger_combo = QComboBox()
        # Initial population, will be updated by Logic Mask
        triggers = [
            "ON_PLAY", "ON_ATTACK", "ON_BLOCK", "ON_DESTROY", "TURN_START", "PASSIVE_CONST", "ON_OTHER_ENTER",
            "ON_ATTACK_FROM_HAND", "AT_BREAK_SHIELD", "ON_CAST_SPELL", "ON_OPPONENT_DRAW"
        ]
        self.populate_combo(self.trigger_combo, triggers, display_func=tr, data_func=lambda x: x)
        self.lbl_trigger = self.add_field(tr("Trigger"), self.trigger_combo)

        # Layer Definition (Static)
        self.layer_group = QGroupBox(tr("Layer Definition"))
        l_layout = QGridLayout(self.layer_group)

        self.layer_type_combo = QComboBox()
        layers = ["COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"]
        self.populate_combo(self.layer_type_combo, layers, display_func=tr, data_func=lambda x: x)

        self.layer_val_spin = QSpinBox()
        self.layer_val_spin.setRange(-9999, 9999)

        self.layer_str_edit = QLineEdit()

        l_layout.addWidget(QLabel(tr("Layer Type")), 0, 0)
        l_layout.addWidget(self.layer_type_combo, 0, 1)
        l_layout.addWidget(QLabel(tr("Value")), 1, 0)
        l_layout.addWidget(self.layer_val_spin, 1, 1)
        l_layout.addWidget(QLabel(tr("String/Keyword")), 2, 0)
        l_layout.addWidget(self.layer_str_edit, 2, 1)

        # Target Filter (Static)
        self.target_filter = FilterEditorWidget()
        self.target_filter.filterChanged.connect(self.update_data)
        self.target_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        l_layout.addWidget(QLabel(tr("Target Filter")), 3, 0)
        l_layout.addWidget(self.target_filter, 3, 1)

        self.add_field(None, self.layer_group)

        # Condition (Shared)
        self.condition_widget = ConditionEditorWidget()
        self.condition_widget.dataChanged.connect(self.update_data)
        self.add_field(None, self.condition_widget)

        # Actions Section
        self.add_action_btn = QPushButton(tr("Add Command"))
        self.add_action_btn.clicked.connect(self.on_add_action_clicked)
        self.add_field(None, self.add_action_btn)

        # Define bindings
        # Note: 'trigger', 'condition', 'type', 'value', 'str_val' are context dependent
        # We can bind the widgets, but need to be careful with saving when mode switches.
        # But _apply_bindings sets widgets, _collect_bindings gathers.
        # Since triggers are hidden in static mode, gathering them doesn't matter much if we clear them in _save_data override.
        # However, due to logic structure, we might want to keep manual saving for conditional parts.
        # Let's bind what is straightforward or 1:1.
        self.bindings = {
            'trigger': self.trigger_combo,
            # For static/modifier
            'type': self.layer_type_combo,
            'value': self.layer_val_spin,
            'str_val': self.layer_str_edit,
            'filter': self.target_filter,
            'condition': self.condition_widget
        }

        # Connect signals
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo.currentIndexChanged.connect(self.update_data)

        self.trigger_combo.currentIndexChanged.connect(self.update_data)

        self.layer_type_combo.currentIndexChanged.connect(self.update_data)
        self.layer_val_spin.valueChanged.connect(self.update_data)
        self.layer_str_edit.textChanged.connect(self.update_data)

        # Initial UI State
        self.on_mode_changed()

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_triggered = (mode == "TRIGGERED")

        self.trigger_combo.setVisible(is_triggered)
        self.lbl_trigger.setVisible(is_triggered)

        self.layer_group.setVisible(not is_triggered)

        if is_triggered:
            self.condition_widget.setTitle(tr("Trigger Condition"))
        else:
            self.condition_widget.setTitle(tr("Apply Condition"))

    def on_add_action_clicked(self):
        self.structure_update_requested.emit("ADD_CHILD_ACTION", {})

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        # Logic Mask: Filter triggers based on Card Type
        # Traverse up to find Card or Spell Side to determine type
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
        self.on_mode_changed()

        if mode == "TRIGGERED":
             # Try to normalize data for binding if legacy keys exist
             if 'trigger_condition' in data and 'condition' not in data:
                 data['condition'] = data['trigger_condition']
        else:
            # STATIC (ModifierDef) - Normalize for bindings
            # Map legacy/variant keys to standard keys matching bindings
            if 'layer_type' in data: data['type'] = data['layer_type']
            if 'layer_value' in data: data['value'] = data['layer_value']
            if 'layer_str' in data: data['str_val'] = data['layer_str']
            if 'static_condition' in data and 'condition' not in data:
                 data['condition'] = data['static_condition']

        # Use Bindings
        self._apply_bindings(data)

        # Ensure fallback for condition if missing
        if not data.get('condition'):
             self.condition_widget.set_data({})


    def update_trigger_options(self, card_type):
        is_spell = (card_type == "SPELL")

        all_triggers = [
            "ON_PLAY", "ON_ATTACK", "ON_BLOCK", "ON_DESTROY", "TURN_START",
            "PASSIVE_CONST", "ON_OTHER_ENTER", "ON_ATTACK_FROM_HAND",
            "AT_BREAK_SHIELD", "ON_CAST_SPELL", "ON_OPPONENT_DRAW"
        ]

        allowed = []
        if is_spell:
            # Limit triggers for Spells to relevant ones
            allowed = ["ON_PLAY", "ON_CAST_SPELL", "TURN_START", "ON_OPPONENT_DRAW", "PASSIVE_CONST", "ON_OTHER_ENTER"]
        else:
            allowed = all_triggers

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

    def _save_data(self, data):
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
            # Clean Static/Legacy keys
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        else: # STATIC
            # Handle str_val optionality
            if not self.layer_str_edit.text():
                data.pop('str_val', None)

            # Clean Trigger/Legacy keys
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

    def block_signals_all(self, block):
        self.mode_combo.blockSignals(block)
        super().block_signals_all(block)
