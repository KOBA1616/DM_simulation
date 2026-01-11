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

# Load schema
def load_schema():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, '..', '..', '..', 'data', 'configs', 'command_schema.json'),
        os.path.join(os.getcwd(), 'data', 'configs', 'command_schema.json')
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

COMMAND_SCHEMA = load_schema()

COMMAND_GROUPS = {
    'DRAW': ['DRAW_CARD'],
    'CARD_MOVE': ['TRANSITION', 'REPLACE_CARD_MOVE', 'RETURN_TO_HAND', 'DISCARD', 'DESTROY', 'MANA_CHARGE'],
    'DECK_OPS': ['SEARCH_DECK', 'LOOK_AND_ADD', 'REVEAL_CARDS', 'SHUFFLE_DECK', 'SEND_TO_DECK_BOTTOM', 'SEARCH_DECK_BOTTOM'],
    'PLAY': ['PLAY_FROM_ZONE', 'CAST_SPELL', 'SUMMON_TOKEN', 'PUT_CREATURE'],
    'BUFFER': ['LOOK_TO_BUFFER', 'SELECT_FROM_BUFFER', 'PLAY_FROM_BUFFER', 'MOVE_BUFFER_TO_ZONE'],
    'CHEAT_PUT': ['MEKRAID', 'FRIEND_BURST', 'REVOLUTION_CHANGE'],
    'GRANT': ['MUTATE', 'POWER_MOD', 'ADD_KEYWORD', 'TAP', 'UNTAP', 'REGISTER_DELAYED_EFFECT', 'APPLY_MODIFIER'],
    'LOGIC': ['QUERY', 'FLOW', 'SELECT_NUMBER', 'CHOICE', 'SELECT_OPTION', 'IF', 'IF_ELSE', 'ELSE', 'STAT'],
    'BATTLE': ['BREAK_SHIELD', 'RESOLVE_BATTLE', 'SHIELD_BURN', 'SHIELD_TRIGGER', 'ADD_SHIELD', 'SEND_SHIELD_TO_GRAVE'],
    'RESTRICTION': [],
    'SPECIAL': ['GAME_RESULT', 'COST_REFERENCE']
}

class UnifiedActionForm(BaseEditForm):
    """Schema-driven Unified Action Form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets_map = {} # key -> widget
        self.fields_config = [] # Current list of field configs
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
            # Fallback to all known types if group empty or error
            types = sorted(list(COMMAND_SCHEMA.keys()))

        prev = self.type_combo.currentData()
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)

        if prev and prev in types:
            self.set_combo_by_data(self.type_combo, prev)
        else:
            self.type_combo.setCurrentIndex(0)

    def on_type_changed(self):
        t = self.type_combo.currentData()
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
        self.fields_config = COMMAND_SCHEMA.get(cmd_type, {}).get('fields', [])

        for field in self.fields_config:
            self._create_widget_for_field(field)

    def _create_widget_for_field(self, field):
        key = field['key']
        label = field.get('label')

        # Use WidgetFactory to create widget
        widget = WidgetFactory.create_widget(self, field, self.update_data)

        # Store for options generation callback
        if field.get('widget') == 'options_control':
             self.option_count_spin = widget.spin
             self.option_gen_layout = widget.option_layout

        if widget:
            self.widgets_map[key] = widget
            row_label = tr(label) if label else None
            # Special handling for checkboxes where label is internal
            if field.get('widget') == 'checkbox':
                row_label = None

            if row_label:
                self.dynamic_layout.addRow(row_label, widget)
            else:
                self.dynamic_layout.addRow(widget)

    def request_generate_options(self):
        """Re-implementation for option generation."""
        if not getattr(self, 'current_item', None):
            return
        try:
            count = int(self.option_count_spin.value())
        except Exception:
            count = 1

        new_options = []
        for _ in range(max(1, count)):
            new_options.append([{"type": "NONE"}])

        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2) or {}
        data['options'] = new_options
        data['type'] = 'CHOICE'

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        try:
            self.load_data(self.current_item)
        except Exception:
            self.dataChanged.emit()

    def _load_ui_from_data(self, data, item):
        """Loads data into the UI widgets."""
        if not data: data = {}
        model = CommandModel(data)
        self.current_model = model

        # 1. Determine Type and Group
        cmd_type = model.type

        # Special case: MUTATE + REVOLUTION_CHANGE
        if cmd_type == 'MUTATE' and model.get('mutation_kind') == 'REVOLUTION_CHANGE':
            cmd_type = 'REVOLUTION_CHANGE'

        grp = 'OTHER'
        for g, types in COMMAND_GROUPS.items():
            if cmd_type in types:
                grp = g
                break
        
        self.set_combo_by_data(self.action_group_combo, grp)
        self.set_combo_by_data(self.type_combo, cmd_type)
        
        # 2. Iterate schema fields and populate widgets
        for field in self.fields_config:
            key = field['key']
            widget = self.widgets_map.get(key)
            if not widget: continue

            w_type = field.get('widget')

            if w_type == 'text':
                widget.setText(str(model.get(key, '')))

            elif w_type == 'spinbox':
                val = model.get(key, 0)
                widget.setValue(int(val))

            elif w_type == 'checkbox':
                if key == 'play_for_free':
                     flags = model.play_flags
                     widget.setChecked('PLAY_FOR_FREE' in flags)
                elif key == 'allow_duplicates':
                     flags = model.flags
                     widget.setChecked('ALLOW_DUPLICATES' in flags)
                elif key == 'up_to':
                     flags = model.flags
                     widget.setChecked('UP_TO' in flags)
                else:
                    widget.setChecked(bool(model.get(key, False)))

            elif w_type == 'player_scope':
                tg = model.target_group
                widget.self_chk.blockSignals(True)
                widget.opp_chk.blockSignals(True)
                widget.self_chk.setChecked(tg in ['PLAYER_SELF', 'PLAYER_BOTH'])
                widget.opp_chk.setChecked(tg in ['PLAYER_OPPONENT', 'PLAYER_BOTH'])
                widget.self_chk.blockSignals(False)
                widget.opp_chk.blockSignals(False)

            elif w_type in ['zone_combo', 'scope_combo', 'query_mode_combo', 'keyword_combo', 'ref_mode_combo']:
                val = model.get(key, 'NONE')
                # For query_mode, logic mapped str_param to mode if type=QUERY
                if cmd_type == 'QUERY' and key == 'str_param':
                    if val == '': val = 'CARDS_MATCHING_FILTER'

                self.set_combo_by_data(widget, val)

            elif w_type == 'filter_editor':
                widget.set_data(model.get(key, {}))

            elif w_type == 'variable_link':
                widget.set_data(data) # It handles extraction internally

            elif w_type == 'options_control':
                 widget.spin.setValue(int(model.get('amount', 0)))


    def _save_ui_to_data(self, data):
        """Saves UI state back to dictionary data."""
        cmd_type = self.type_combo.currentData()

        new_data = {'format': 'command', 'type': cmd_type}

        # Handle REVOLUTION_CHANGE mapping back
        if cmd_type == 'REVOLUTION_CHANGE':
            new_data['type'] = 'MUTATE'
            new_data['mutation_kind'] = 'REVOLUTION_CHANGE'

        model = CommandModel(new_data)

        for field in self.fields_config:
            key = field['key']
            widget = self.widgets_map.get(key)
            if not widget: continue

            w_type = field.get('widget')

            if w_type == 'text':
                model.set(key, widget.text())

            elif w_type == 'spinbox':
                model.set(key, widget.value())

            elif w_type == 'checkbox':
                val = widget.isChecked()
                if key == 'play_for_free':
                    if val: model.play_flags = ['PLAY_FOR_FREE']
                elif key == 'allow_duplicates':
                    if val: model.flags = ['ALLOW_DUPLICATES']
                elif key == 'up_to':
                    if val: model.flags = ['UP_TO']
                else:
                    if val: model.set(key, True)

            elif w_type == 'player_scope':
                s = widget.self_chk.isChecked()
                o = widget.opp_chk.isChecked()
                if s and o: t = 'PLAYER_BOTH'
                elif o: t = 'PLAYER_OPPONENT'
                else: t = 'PLAYER_SELF'
                model.target_group = t

            elif w_type in ['zone_combo', 'scope_combo', 'query_mode_combo', 'keyword_combo', 'ref_mode_combo']:
                model.set(key, widget.currentData())

            elif w_type == 'filter_editor':
                model.set(key, widget.get_data())

            elif w_type == 'variable_link':
                widget.get_data(new_data) # Writes directly to dict

            elif w_type == 'options_control':
                model.amount = widget.spin.value()

        # Ensure output keys are generated if needed
        # (Legacy logic did this, good to keep)
        out_key = new_data.get('output_value_key')
        if not out_key:
             # Check if schema says it produces output
             field = next((f for f in self.fields_config if f.get('key') == 'output_link'), None)
             if field and field.get('produces_output'):
                  row = 0
                  if getattr(self, 'current_item', None):
                      row = self.current_item.row()
                  new_data['output_value_key'] = f"var_{cmd_type}_{row}"

        data.clear()
        data.update(new_data)

    def _get_display_text(self, data):
        t = data.get('type', 'UNKNOWN')
        if t == 'MUTATE' and data.get('mutation_kind') == 'REVOLUTION_CHANGE':
            return f"{tr('Command')}: {tr('REVOLUTION_CHANGE')}"
        return f"{tr('Command')}: {tr(t)}"
