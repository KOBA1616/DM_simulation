from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QLabel, QLineEdit
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget

# Configuration for Condition UI logic
CONDITION_UI_CONFIG = {
    "NONE": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "MANA_ARMED": {
        "show_val": True,
        "show_str": True, # Usually specifies civ
        "label_val": "Count",
        "label_str": "Civilization"
    },
    "SHIELD_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count",
        "label_str": "Comparison (Optional)"
    },
    "CIVILIZATION_MATCH": {
        "show_val": False,
        "show_str": True,
        "label_val": "Value",
        "label_str": "Civilization"
    },
    "OPPONENT_PLAYED_WITHOUT_MANA": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "OPPONENT_DRAW_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count (>=)",
        "label_str": "String"
    },
    "DURING_YOUR_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "DURING_OPPONENT_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "FIRST_ATTACK": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "EVENT_FILTER_MATCH": {
        "show_val": False,
        "show_str": False,
        "show_filter": True
    }
}

class EffectEditForm(BaseEditForm):
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
        triggers = [
            "ON_PLAY", "ON_ATTACK", "ON_DESTROY", "TURN_START", "PASSIVE_CONST", "ON_OTHER_ENTER",
            "ON_ATTACK_FROM_HAND", "AT_BREAK_SHIELD", "ON_CAST_SPELL", "ON_OPPONENT_DRAW"
        ]
        # Use localized strings for display
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
        self.target_filter.filterChanged.connect(self.update_data)
        self.target_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        l_layout.addWidget(QLabel(tr("Target Filter")), 3, 0)
        l_layout.addWidget(self.target_filter, 3, 1)

        layout.addRow(self.layer_group)

        # Condition (Shared)
        self.condition_group = QGroupBox(tr("Condition"))
        c_layout = QGridLayout(self.condition_group)
        self.cond_type_combo = QComboBox()
        cond_types = [
            "NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH",
            "OPPONENT_PLAYED_WITHOUT_MANA", "OPPONENT_DRAW_COUNT",
            "DURING_YOUR_TURN", "DURING_OPPONENT_TURN",
            "FIRST_ATTACK", "EVENT_FILTER_MATCH"
        ]
        # Use localized strings for display
        self.populate_combo(self.cond_type_combo, cond_types, display_func=tr, data_func=lambda x: x)

        c_layout.addWidget(QLabel(tr("Type")), 0, 0)
        c_layout.addWidget(self.cond_type_combo, 0, 1)

        # Value Row
        self.lbl_val = QLabel(tr("Value"))
        self.cond_val_spin = QSpinBox()
        c_layout.addWidget(self.lbl_val, 1, 0)
        c_layout.addWidget(self.cond_val_spin, 1, 1)

        # String Row
        self.lbl_str = QLabel(tr("String Value"))
        self.cond_str_edit = QLineEdit()
        c_layout.addWidget(self.lbl_str, 2, 0)
        c_layout.addWidget(self.cond_str_edit, 2, 1)

        # Filter Widget
        self.cond_filter = FilterEditorWidget()
        self.cond_filter.filterChanged.connect(self.update_data)
        self.cond_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        self.cond_filter.setVisible(False)
        c_layout.addWidget(self.cond_filter, 3, 0, 1, 2)

        layout.addRow(self.condition_group)

        # Connect signals
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo.currentIndexChanged.connect(self.update_data)

        self.trigger_combo.currentIndexChanged.connect(self.update_data)

        self.layer_type_combo.currentIndexChanged.connect(self.update_data)
        self.layer_val_spin.valueChanged.connect(self.update_data)
        self.layer_str_edit.textChanged.connect(self.update_data)

        self.cond_type_combo.currentIndexChanged.connect(self.on_cond_type_changed)
        self.cond_val_spin.valueChanged.connect(self.update_data)
        self.cond_str_edit.textChanged.connect(self.update_data)

        # Initial UI State
        self.update_ui_visibility("NONE")
        self.on_mode_changed()

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_triggered = (mode == "TRIGGERED")

        self.trigger_combo.setVisible(is_triggered)
        self.lbl_trigger.setVisible(is_triggered)

        self.layer_group.setVisible(not is_triggered)

        if is_triggered:
            self.condition_group.setTitle(tr("Trigger Condition"))
        else:
            self.condition_group.setTitle(tr("Apply Condition"))

    def on_cond_type_changed(self):
        ctype = self.cond_type_combo.currentData()
        self.update_ui_visibility(ctype)
        self.update_data()

    def update_ui_visibility(self, condition_type):
        config = CONDITION_UI_CONFIG.get(condition_type, CONDITION_UI_CONFIG["NONE"])

        show_val = config.get("show_val", True)
        label_val = config.get("label_val", "Value")

        show_str = config.get("show_str", True)
        label_str = config.get("label_str", "String Value")

        show_filter = config.get("show_filter", False)

        self.lbl_val.setText(tr(label_val))
        self.lbl_val.setVisible(show_val)
        self.cond_val_spin.setVisible(show_val)

        self.lbl_str.setText(tr(label_str))
        self.lbl_str.setVisible(show_str)
        self.cond_str_edit.setVisible(show_str)

        self.cond_filter.setVisible(show_filter)

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

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
            # Use 'type' preferentially, fallback to 'layer_type'
            m_type = data.get('type', data.get('layer_type', 'COST_MODIFIER'))
            m_val = data.get('value', data.get('layer_value', 0))
            m_str = data.get('str_val', data.get('layer_str', ''))
            m_filter = data.get('filter', {})

            self.set_combo_by_data(self.layer_type_combo, m_type)
            self.layer_val_spin.setValue(m_val)
            self.layer_str_edit.setText(m_str)
            self.target_filter.set_data(m_filter)

            cond = data.get('condition', data.get('static_condition', {}))

        ctype = cond.get('type', 'NONE')
        self.set_combo_by_data(self.cond_type_combo, ctype)

        self.cond_val_spin.setValue(cond.get('value', 0))
        self.cond_str_edit.setText(cond.get('str_val', ''))

        self.cond_filter.set_data(cond.get('filter', {}))

        self.update_ui_visibility(ctype)

    def _save_data(self, data):
        mode = self.mode_combo.currentData()

        # Build Condition Dict
        cond = {}
        cond['type'] = self.cond_type_combo.currentData()
        cond['value'] = self.cond_val_spin.value()
        str_val = self.cond_str_edit.text()
        if str_val: cond['str_val'] = str_val
        if self.cond_filter.isVisible():
             cond['filter'] = self.cond_filter.get_data()

        # Update Item Type if possible
        if self.current_item:
             if mode == "TRIGGERED":
                 self.current_item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
             else:
                 self.current_item.setData("MODIFIER", Qt.ItemDataRole.UserRole + 1)

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
        # Use existence of 'type' or 'trigger' keys,
        # or fallback to item type check (but we don't have item here, only data)
        if 'trigger' in data:
             return f"{tr('Effect')}: {tr(data.get('trigger', ''))}"
        elif 'type' in data or 'layer_type' in data:
             t = data.get('type', data.get('layer_type', ''))
             return f"{tr('Static')}: {tr(t)}"
        else:
             return tr("Unknown Effect")

    def block_signals_all(self, block):
        self.mode_combo.blockSignals(block)
        self.trigger_combo.blockSignals(block)
        self.layer_type_combo.blockSignals(block)
        self.layer_val_spin.blockSignals(block)
        self.layer_str_edit.blockSignals(block)

        self.cond_type_combo.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
        self.cond_filter.blockSignals(block)
