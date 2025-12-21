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
        layout = QFormLayout(self)

        # Ability Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("TRIGGERED"), "TRIGGERED")
        self.mode_combo.addItem(tr("STATIC"), "STATIC")
        layout.addRow(tr("Ability Mode"), self.mode_combo)

        # Trigger Definition
        self.trigger_combo = QComboBox()
        # Initial population, will be updated by Logic Mask
        triggers = [
            "ON_PLAY", "ON_ATTACK", "ON_BLOCK", "ON_DESTROY", "TURN_START", "PASSIVE_CONST", "ON_OTHER_ENTER",
            "ON_ATTACK_FROM_HAND", "AT_BREAK_SHIELD", "ON_CAST_SPELL", "ON_OPPONENT_DRAW"
        ]
        self.populate_combo(self.trigger_combo, triggers, display_func=tr, data_func=lambda x: x)
        self.lbl_trigger = QLabel(tr("Trigger"))
        layout.addRow(self.lbl_trigger, self.trigger_combo)

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
        self.register_widget(self.target_filter)
        self.target_filter.filterChanged.connect(self.update_data)
        self.target_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        l_layout.addWidget(QLabel(tr("Target Filter")), 3, 0)
        l_layout.addWidget(self.target_filter, 3, 1)

        layout.addRow(self.layer_group)

        # Condition (Shared)
        self.condition_widget = ConditionEditorWidget()
        self.register_widget(self.condition_widget)
        self.condition_widget.dataChanged.connect(self.update_data)
        layout.addRow(self.condition_widget)

        # Actions Section
        self.add_action_btn = QPushButton(tr("Add Action"))
        self.add_action_btn.clicked.connect(self.on_add_action_clicked)
        layout.addRow(self.add_action_btn)

        # Connect signals
        self.connect_signal(self.mode_combo, self.mode_combo.currentIndexChanged, self.on_mode_changed)
        self.connect_signal(self.mode_combo, self.mode_combo.currentIndexChanged, self.update_data)

        self.connect_signal(self.trigger_combo, self.trigger_combo.currentIndexChanged, self.update_data)

        self.connect_signal(self.layer_type_combo, self.layer_type_combo.currentIndexChanged, self.update_data)
        self.connect_signal(self.layer_val_spin, self.layer_val_spin.valueChanged, self.update_data)
        self.connect_signal(self.layer_str_edit, self.layer_str_edit.textChanged, self.update_data)

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
            self.set_combo_by_data(self.trigger_combo, data.get('trigger', 'ON_PLAY'))
            cond = data.get('condition', data.get('trigger_condition', {}))
        else:
            # STATIC (ModifierDef)
            m_type = data.get('type', data.get('layer_type', 'COST_MODIFIER'))
            m_val = data.get('value', data.get('layer_value', 0))
            m_str = data.get('str_val', data.get('layer_str', ''))
            m_filter = data.get('filter', {})

            self.set_combo_by_data(self.layer_type_combo, m_type)
            self.layer_val_spin.setValue(m_val)
            self.layer_str_edit.setText(m_str)
            self.target_filter.set_data(m_filter)

            cond = data.get('condition', data.get('static_condition', {}))

        self.condition_widget.set_data(cond)

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

        # Build Condition Dict
        cond = self.condition_widget.get_data()

        # Update Item Type if possible
        if self.current_item:
             if mode == "TRIGGERED":
                 self.current_item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
             else:
                 self.current_item.setData("MODIFIER", Qt.ItemDataRole.UserRole + 1)

             # Check for Group Mismatch and Request Move
             parent = self.current_item.parent()
             if parent:
                 parent_type = parent.data(Qt.ItemDataRole.UserRole + 1)
                 # We use the item pointer itself to identify it, but passing QStandardItem via dict is safe in-process
                 # However, usually we pass indices. But index is ephemeral.
                 # Let's pass the item object.
                 if mode == "TRIGGERED" and parent_type == "GROUP_STATIC":
                     self.structure_update_requested.emit("MOVE_EFFECT", {"item": self.current_item, "target_type": "TRIGGERED"})
                 elif mode == "STATIC" and parent_type == "GROUP_TRIGGER":
                     self.structure_update_requested.emit("MOVE_EFFECT", {"item": self.current_item, "target_type": "STATIC"})

        if mode == "TRIGGERED":
            data['trigger'] = self.trigger_combo.currentData()
            data['condition'] = cond

            # Clean Static/Legacy keys
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        else: # STATIC
            data['type'] = self.layer_type_combo.currentData()
            data['value'] = self.layer_val_spin.value()
            if self.layer_str_edit.text():
                data['str_val'] = self.layer_str_edit.text()
            else:
                data.pop('str_val', None)

            data['condition'] = cond
            data['filter'] = self.target_filter.get_data()

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
