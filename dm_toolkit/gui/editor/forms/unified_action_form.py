# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys, normalize_command_zone_keys
from dm_toolkit.consts import COMMAND_TYPES, GRANTABLE_KEYWORDS
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.gui.editor.action_converter import ActionConverter
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_player_scope_selector, make_value_spin, make_measure_mode_combo,
    make_ref_mode_combo, make_zone_combos, make_option_controls
)
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

# Grouping of Command types for UI organization
# 新分類: ドロー、カード移動、デッキ操作、プレイ、踏み倒し、付与、ロジック、バトル、制限、特殊
# 注: エンジン内部専用のタイプ（SUMMON_TOKEN, RESOLVE_EFFECT, RESOLVE_PLAY, NONE, ATTACK_PLAYER, ATTACK_CREATURE, BLOCK）はGUIから除外
COMMAND_GROUPS = {
    'DRAW': [
        'DRAW_CARD'
    ],
    'CARD_MOVE': [
        'TRANSITION', 'RETURN_TO_HAND', 'DISCARD', 'DESTROY', 'MANA_CHARGE', 'MOVE_BUFFER_TO_ZONE'
    ],
    'DECK_OPS': [
        'SEARCH_DECK', 'LOOK_AND_ADD', 'REVEAL_CARDS', 'SHUFFLE_DECK', 'LOOK_TO_BUFFER', 'SELECT_FROM_BUFFER'
    ],
    'PLAY': [
        'PLAY_FROM_ZONE', 'PLAY_FROM_BUFFER', 'CAST_SPELL'
    ],
    'CHEAT_PUT': [
        'MEKRAID', 'FRIEND_BURST'
    ],
    'GRANT': [
        'MUTATE', 'POWER_MOD', 'ADD_KEYWORD', 'TAP', 'UNTAP', 'REGISTER_DELAYED_EFFECT'
    ],
    'LOGIC': [
        'QUERY', 'FLOW', 'SELECT_NUMBER', 'CHOICE', 'SELECT_OPTION'
    ],
    'BATTLE': [
        'BREAK_SHIELD', 'RESOLVE_BATTLE', 'SHIELD_BURN', 'SHIELD_TRIGGER'
    ],
    'RESTRICTION': [
    ]
}


class UnifiedActionForm(BaseEditForm):
    """Unified editor for Commands. Legacy Actions are automatically converted.
    """
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure some attributes exist even if UI creation fails (headless/static checks)
        self.current_item = getattr(self, 'current_item', None)
        self.generate_options_btn = getattr(self, 'generate_options_btn', None)
        self.option_count_spin = getattr(self, 'option_count_spin', None)
        self.option_count_label = getattr(self, 'option_count_label', None)
        self.stat_key_combo = getattr(self, 'stat_key_combo', None)
        self.stat_key_label = getattr(self, 'stat_key_label', None)
        self.stat_preset_btn = getattr(self, 'stat_preset_btn', None)
        self.structure_update_requested = getattr(self, 'structure_update_requested', None)
        try:
            self.setup_ui()
        except Exception:
            # Defer full UI setup in environments without a QApplication
            pass

    def _get_ui_config(self, tpe):
        # Use only COMMAND_UI_CONFIG
        cmd_cfg = COMMAND_UI_CONFIG.get(tpe, {})

        # Source visible lists
        cmd_vis = set(cmd_cfg.get('visible', []))

        # Labels
        amount_label = cmd_cfg.get('label_amount') or 'Amount'
        val2_label = cmd_cfg.get('label_val2') or 'Value 2'
        str_label = cmd_cfg.get('label_str_param') or ''
        mutation_label = cmd_cfg.get('label_mutation_kind') or ''

        return {
            'target_group_visible': ('target_group' in cmd_vis),
            'target_filter_visible': ('target_filter' in cmd_vis),
            'amount_visible': ('amount' in cmd_vis),
            'amount_label': amount_label,
            'val2_visible': ('val2' in cmd_vis),
            'val2_label': val2_label,
            'str_param_visible': ('str_param' in cmd_vis),
            'str_param_label': str_label,
            'mutation_kind_visible': ('mutation_kind' in cmd_vis),
            'mutation_kind_label': mutation_label,
            'to_zone_visible': ('to_zone' in cmd_vis),
            'from_zone_visible': ('from_zone' in cmd_vis),
            'optional_visible': ('optional' in cmd_vis),
            'input_link_visible': ('input_link' in cmd_vis),
            'produces_output': cmd_cfg.get('produces_output', False),
            'tooltip': cmd_cfg.get('tooltip', ''),
            'allowed_filter_fields': cmd_cfg.get('allowed_filter_fields', None),
            'can_be_optional': False # Commands use 'optional' flag in visible list
        }

    def setup_ui(self):
        layout = QFormLayout(self)

        from PyQt6.QtWidgets import QComboBox
        # Command Group (top-level) + Type (subtype)
        self.action_group_combo = QComboBox()
        groups = list(COMMAND_GROUPS.keys())
        self.populate_combo(self.action_group_combo, groups, data_func=lambda x: x, display_func=tr)
        self.register_widget(self.action_group_combo)
        layout.addRow(tr("Command Group"), self.action_group_combo)

        self.type_combo = QComboBox()
        # Initial population will be triggered by group change, but we need a default list
        # We'll use all keys from COMMAND_UI_CONFIG or COMMAND_TYPES
        self.known_types = sorted(list(COMMAND_UI_CONFIG.keys()))
        if "NONE" in self.known_types:
            self.known_types.remove("NONE")
            self.known_types.insert(0, "NONE")

        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        self.register_widget(self.type_combo)
        layout.addRow(tr("Command Type"), self.type_combo)

        # Scope / Target Group (Requirement: only self/opponent, civ-selector-like)
        self.scope_widget, self.scope_self_check, self.scope_opp_check = make_player_scope_selector(self)
        self.scope_self_check.setChecked(True)
        self.register_widget(self.scope_widget)
        self.register_widget(self.scope_self_check)
        self.register_widget(self.scope_opp_check)
        layout.addRow(tr("Scope"), self.scope_widget)

        # Common fields
        self.str_edit = QLineEdit()
        self.register_widget(self.str_edit)
        self.str_label = QLabel(tr("String"))
        layout.addRow(self.str_label, self.str_edit)

        # Measure Mode (for QUERY)
        self.measure_mode_combo = make_measure_mode_combo(self)
        self.register_widget(self.measure_mode_combo)
        layout.addRow(tr("Query Mode"), self.measure_mode_combo)

        # Stat Key selector for QUERY (GET_GAME_STAT)
        self.stat_key_combo = QComboBox()
        self.register_widget(self.stat_key_combo)
        try:
            stat_keys = list(CardTextGenerator.STAT_KEY_MAP.keys())
            self.populate_combo(self.stat_key_combo, stat_keys, data_func=lambda x: x,
                                display_func=lambda k: CardTextGenerator.STAT_KEY_MAP.get(k, (k, ''))[0])
        except Exception:
            pass
        self.stat_key_label = QLabel(tr("Stat Key"))

        from PyQt6.QtWidgets import QWidget, QHBoxLayout
        stat_row = QWidget()
        stat_row_layout = QHBoxLayout(stat_row)
        stat_row_layout.setContentsMargins(0, 0, 0, 0)
        stat_row_layout.addWidget(self.stat_key_combo)
        self.stat_preset_btn = QPushButton(tr("Preset: Mana Civs"))
        self.stat_preset_btn.setToolTip(tr("Set stat key to MANA_CIVILIZATION_COUNT"))
        self.stat_preset_btn.clicked.connect(lambda: self._preset_stat_key('MANA_CIVILIZATION_COUNT'))
        stat_row_layout.addWidget(self.stat_preset_btn)
        layout.addRow(self.stat_key_label, stat_row)

        # Ref Mode (for COST_REFERENCE -> MUTATE)
        self.ref_mode_combo = make_ref_mode_combo(self)
        self.register_widget(self.ref_mode_combo)
        layout.addRow(tr("Ref Mode"), self.ref_mode_combo)

        self.val1_spin = make_value_spin(self)
        self.register_widget(self.val1_spin)
        self.val1_label = QLabel(tr("Amount"))
        layout.addRow(self.val1_label, self.val1_spin)

        self.val2_spin = make_value_spin(self)
        self.register_widget(self.val2_spin)
        self.val2_label = QLabel(tr("Value 2"))
        layout.addRow(self.val2_label, self.val2_spin)

        # Option generation controls (for CHOICE/SELECT_OPTION)
        self.option_count_spin, self.generate_options_btn, self.option_count_label, self.option_gen_layout = make_option_controls(self)
        # Assuming make_option_controls returns widgets. Register them.
        self.register_widget(self.option_count_spin)
        layout.addRow(self.option_count_label, self.option_gen_layout)

        # Flags
        self.no_cost_check = QCheckBox(tr("Play without paying cost"))
        self.register_widget(self.no_cost_check)
        self.no_cost_label = QLabel("")
        layout.addRow(self.no_cost_label, self.no_cost_check)
        self.allow_duplicates_check = QCheckBox(tr("Allow Duplicates"))
        self.register_widget(self.allow_duplicates_check)
        self.allow_duplicates_label = QLabel("")
        layout.addRow(self.allow_duplicates_label, self.allow_duplicates_check)
        self.arbitrary_check = QCheckBox(tr("Optional / Arbitrary Amount"))
        self.register_widget(self.arbitrary_check)
        self.arbitrary_label = QLabel("")
        layout.addRow(self.arbitrary_label, self.arbitrary_check)

        # Zones
        self.source_zone_combo, self.dest_zone_combo = make_zone_combos(self)
        self.register_widget(self.source_zone_combo)
        self.register_widget(self.dest_zone_combo)
        self.source_zone_label = QLabel(tr("Source Zone"))
        self.dest_zone_label = QLabel(tr("Destination Zone"))
        layout.addRow(self.source_zone_label, self.source_zone_combo)
        layout.addRow(self.dest_zone_label, self.dest_zone_combo)

        # Filter
        self.filter_group = QGroupBox(tr("Filter"))
        self.filter_widget = FilterEditorWidget()
        self.register_widget(self.filter_widget)
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)
        layout.addRow(self.filter_group)

        # Mutation kind (keywords)
        self.mutation_kind_edit = QLineEdit()
        self.register_widget(self.mutation_kind_edit)
        self.mutation_kind_combo = QComboBox()
        self.register_widget(self.mutation_kind_combo)
        self.populate_combo(self.mutation_kind_combo, GRANTABLE_KEYWORDS, data_func=lambda x: x, display_func=tr)
        self.mutation_kind_label = QLabel(tr("Mutation Kind"))

        from PyQt6.QtWidgets import QStackedWidget
        self.mutation_kind_container = QStackedWidget()
        self.mutation_kind_container.addWidget(self.mutation_kind_edit)
        self.mutation_kind_container.addWidget(self.mutation_kind_combo)
        layout.addRow(self.mutation_kind_label, self.mutation_kind_container)

        # Variable link
        self.link_widget = VariableLinkWidget()
        self.register_widget(self.link_widget)
        self.link_widget.linkChanged.connect(self.update_data)
        if hasattr(self.link_widget, 'smartLinkStateChanged'):
            try:
                self.link_widget.smartLinkStateChanged.connect(self.on_smart_link_changed)
            except Exception:
                pass
        layout.addRow(self.link_widget)

        # Signals
        self.action_group_combo.currentIndexChanged.connect(self.on_group_changed)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.measure_mode_combo.currentIndexChanged.connect(self.on_measure_mode_changed)
        self.stat_key_combo.currentIndexChanged.connect(self.on_stat_key_changed)
        self.ref_mode_combo.currentIndexChanged.connect(self.update_data)
        self.generate_options_btn.clicked.connect(self.request_generate_options)
        self.no_cost_check.stateChanged.connect(self.update_data)
        self.allow_duplicates_check.stateChanged.connect(self.update_data)
        self.arbitrary_check.stateChanged.connect(self.update_data)
        self.scope_self_check.stateChanged.connect(self._on_scope_check_changed)
        self.scope_opp_check.stateChanged.connect(self._on_scope_check_changed)
        self.str_edit.textChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.source_zone_combo.currentIndexChanged.connect(self.update_data)
        self.dest_zone_combo.currentIndexChanged.connect(self.update_data)

        self.update_ui_state(self.type_combo.currentData())

    def _on_scope_check_changed(self):
        """自分/相手のチェックを排他的に扱い、常にどちらかを選ばせる。"""
        sender = self.sender()

        if sender == self.scope_self_check and self.scope_self_check.isChecked():
            self.scope_opp_check.blockSignals(True)
            self.scope_opp_check.setChecked(False)
            self.scope_opp_check.blockSignals(False)
        elif sender == self.scope_opp_check and self.scope_opp_check.isChecked():
            self.scope_self_check.blockSignals(True)
            self.scope_self_check.setChecked(False)
            self.scope_self_check.blockSignals(False)

        # If user unchecks the last remaining one, default back to self.
        if (not self.scope_self_check.isChecked()) and (not self.scope_opp_check.isChecked()):
            self.scope_self_check.blockSignals(True)
            self.scope_self_check.setChecked(True)
            self.scope_self_check.blockSignals(False)

        self.update_data()

    def on_group_changed(self):
        """When a group is selected, populate the type combo with relevant subtypes."""
        grp = self.action_group_combo.currentData()
        types = COMMAND_GROUPS.get(grp, [])
        if not types:
            types = self.known_types

        prev = self.type_combo.currentData()
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)
        if prev and prev in types:
            self.set_combo_by_data(self.type_combo, prev)
        else:
            self.type_combo.setCurrentIndex(0)

        try:
            self.update_ui_state(self.type_combo.currentData())
        except Exception:
            pass

    def on_type_changed(self):
        t = self.type_combo.currentData()
        self.update_ui_state(t)
        self.update_data()

    def on_measure_mode_changed(self):
        try:
            val = self.measure_mode_combo.currentData()
            if val:
                self.str_edit.setText(val)
        except Exception:
            pass
        self.update_data()

    def on_smart_link_changed(self, is_active: bool):
        try:
            cfg = self._get_ui_config(self.type_combo.currentData())
            amount_vis = cfg.get('amount_visible', False) and not is_active
            self.val1_label.setVisible(amount_vis)
            self.val1_spin.setVisible(amount_vis)
            if cfg.get('target_filter_visible', False) and hasattr(self.filter_widget, 'set_external_count_control'):
                self.filter_widget.set_external_count_control(is_active)
        except Exception:
            pass
        self.update_data()

    def on_stat_key_changed(self):
        try:
            val = self.stat_key_combo.currentData()
            if val:
                self.str_edit.setText(val)
        except Exception:
            pass
        self.update_data()

    def _preset_stat_key(self, key: str):
        try:
            if self.stat_key_combo:
                self.set_combo_by_data(self.stat_key_combo, key)
                self.str_edit.setText(key)
        except Exception:
            try:
                self.str_edit.setText(key)
            except Exception:
                pass
        self.update_data()

    def request_generate_options(self):
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
        data['type'] = 'CHOICE' # Unified type for options

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        try:
            self.load_data(self.current_item)
        except Exception:
            self.dataChanged.emit()

    def update_ui_state(self, t):
        cfg = self._get_ui_config(t)

        self.scope_widget.setVisible(cfg.get('target_group_visible', False))
        self.filter_group.setVisible(cfg.get('target_filter_visible', False))

        # Amount
        self.val1_label.setVisible(cfg.get('amount_visible', False))
        self.val1_spin.setVisible(cfg.get('amount_visible', False))
        self.val1_label.setText(tr(cfg.get('amount_label', 'Amount')))

        # Value 2
        self.val2_label.setVisible(cfg.get('val2_visible', False))
        self.val2_spin.setVisible(cfg.get('val2_visible', False))
        self.val2_label.setText(tr(cfg.get('val2_label', 'Value 2')))

        # String param
        self.str_label.setVisible(cfg.get('str_param_visible', False))
        self.str_edit.setVisible(cfg.get('str_param_visible', False))
        if cfg.get('str_param_label'):
            self.str_label.setText(tr(cfg.get('str_param_label')))

        # Mutation kind
        self.mutation_kind_label.setVisible(cfg.get('mutation_kind_visible', False))
        if self.mutation_kind_container:
            self.mutation_kind_container.setVisible(cfg.get('mutation_kind_visible', False))
        if cfg.get('mutation_kind_label'):
            self.mutation_kind_label.setText(tr(cfg.get('mutation_kind_label')))

        # Zones
        self.source_zone_label.setVisible(cfg.get('from_zone_visible', False))
        self.source_zone_combo.setVisible(cfg.get('from_zone_visible', False))
        self.dest_zone_label.setVisible(cfg.get('to_zone_visible', False))
        self.dest_zone_combo.setVisible(cfg.get('to_zone_visible', False))

        # Input linking
        can_link_input = cfg.get('amount_visible', False) or cfg.get('input_link_visible', False)
        self.link_widget.set_smart_link_enabled(can_link_input)

        # Produce output hint
        produces = cfg.get('produces_output', False)
        self.link_widget.set_output_hint(produces) if hasattr(self.link_widget, 'set_output_hint') else None

        # Special Modes
        is_query = (t == 'QUERY')
        self.measure_mode_combo.setVisible(is_query)

        self.ref_mode_combo.setVisible(t == 'MUTATE' and self.mutation_kind_combo.currentData() == 'COST_REFERENCE')

        # Option-related
        is_choice = (t == 'CHOICE' or t == 'SELECT_OPTION')
        self.option_count_label.setVisible(is_choice)
        self.option_count_spin.setVisible(is_choice)
        self.generate_options_btn.setVisible(is_choice)
        self.allow_duplicates_label.setVisible(is_choice)
        self.allow_duplicates_check.setVisible(is_choice)

        # Flags
        self.no_cost_label.setVisible(t == 'PLAY_FROM_ZONE')
        self.no_cost_check.setVisible(t == 'PLAY_FROM_ZONE')
        self.arbitrary_label.setVisible(cfg.get('optional_visible', False))
        self.arbitrary_check.setVisible(cfg.get('optional_visible', False))

        self.type_combo.setToolTip(tr(cfg.get('tooltip', '')))

        # Mutation kind widget switch
        try:
            if t == 'ADD_KEYWORD' and self.mutation_kind_container:
                self.mutation_kind_container.setCurrentIndex(1)
            elif self.mutation_kind_container:
                self.mutation_kind_container.setCurrentIndex(0)
        except Exception:
            pass

    def _update_ui_state(self, data):
        self.update_ui_state(data.get('type', 'NONE'))

    def _load_ui_from_data(self, data, item):
        """Load data into UI, auto-converting legacy actions if needed."""
        if item:
            self.link_widget.set_current_item(item)

        normalize_action_zone_keys(data)
        normalize_command_zone_keys(data)

        # Check for legacy and convert if necessary
        # We rely on 'format' flag or heuristic checks (legacy keys presence)
        is_legacy = data.get('format') != 'command'

        if is_legacy and data.get('type', 'NONE') != 'NONE':
            try:
                converted = ActionConverter.convert(data)
                if converted and converted.get('type') != 'NONE':
                    # Update data in-place to reflect conversion
                    data.clear()
                    data.update(converted)
                    data['format'] = 'command'
            except Exception:
                pass

        ui_type = data.get('type', 'NONE')

        # Determine group
        grp = None
        for g, types in COMMAND_GROUPS.items():
            if ui_type in types:
                grp = g
                break
        if grp is None:
            grp = 'OTHER'

        self.set_combo_by_data(self.action_group_combo, grp)
        self.set_combo_by_data(self.type_combo, ui_type)

        target_group = data.get('target_group') or 'PLAYER_SELF'
        self.scope_self_check.blockSignals(True)
        self.scope_opp_check.blockSignals(True)
        self.scope_self_check.setChecked(target_group == 'PLAYER_SELF')
        self.scope_opp_check.setChecked(target_group == 'PLAYER_OPPONENT')
        if not self.scope_self_check.isChecked() and not self.scope_opp_check.isChecked():
            self.scope_self_check.setChecked(True)
        self.scope_self_check.blockSignals(False)
        self.scope_opp_check.blockSignals(False)

        self.str_edit.setText(data.get('str_param', ''))

        # Value Mapping for specific command types
        val1 = data.get('amount', 0)
        val2 = 0

        if ui_type == 'LOOK_AND_ADD':
            val1 = data.get('look_count', 0)
            val2 = data.get('add_count', 0)
        elif ui_type == 'MEKRAID':
            val1 = data.get('look_count', 0) # Mekraid Level
            val2 = data.get('max_cost', 0) # Usually same?
            # Wait, Mekraid N means look 3, play cost < N.
            # ActionConverter: cmd['look_count'] = 3, cmd['max_cost'] = value1
            # So val1 should map to max_cost
            val1 = data.get('max_cost', 0)
            # look_count is usually fixed/derived, but if we want to edit it?
            # Let's say val2 is look_count
            val2 = data.get('look_count', 3)
        elif ui_type == 'SELECT_NUMBER':
            val1 = data.get('max', 0)
            val2 = data.get('min', 0)

        self.val1_spin.setValue(val1)
        self.val2_spin.setValue(val2)

        self.set_combo_by_data(self.source_zone_combo, data.get('from_zone', 'NONE'))
        self.set_combo_by_data(self.dest_zone_combo, data.get('to_zone', 'NONE'))

        self.filter_widget.set_data(data.get('target_filter', {}))
        self.link_widget.set_data(data)

        # Mutation kind
        mutation_kind = data.get('mutation_kind', '')
        if mutation_kind in GRANTABLE_KEYWORDS:
             self.set_combo_by_data(self.mutation_kind_combo, mutation_kind)
        else:
             self.mutation_kind_edit.setText(mutation_kind)

        # Flags
        flags = data.get('flags', [])
        self.allow_duplicates_check.setChecked('ALLOW_DUPLICATES' in flags)
        self.arbitrary_check.setChecked(data.get('optional', False) or 'OPTIONAL' in flags)

        play_flags = data.get('play_flags', [])
        self.no_cost_check.setChecked('PLAY_FOR_FREE' in play_flags)

    def _save_ui_to_data(self, data):
        """Save UI to data as a Command."""
        sel = self.type_combo.currentData()

        cmd = {}
        cmd['format'] = 'command' # Explicitly mark format
        cmd['type'] = sel

        cmd['target_group'] = 'PLAYER_SELF' if self.scope_self_check.isChecked() else 'PLAYER_OPPONENT'
        cmd['target_filter'] = self.filter_widget.get_data()

        # Standard mapping
        cmd['amount'] = self.val1_spin.value()

        # Specialized reverse mapping
        if sel == 'LOOK_AND_ADD':
            cmd['look_count'] = self.val1_spin.value()
            cmd['add_count'] = self.val2_spin.value()
            # Also keep 'amount' as legacy fallback/primary if engine expects it?
            # Engine likely expects look_count/add_count
        elif sel == 'MEKRAID':
            cmd['max_cost'] = self.val1_spin.value()
            cmd['look_count'] = self.val2_spin.value()
        elif sel == 'SELECT_NUMBER':
            cmd['max'] = self.val1_spin.value()
            cmd['min'] = self.val2_spin.value()

        # Flags
        flags = []
        if self.allow_duplicates_check.isChecked():
            flags.append('ALLOW_DUPLICATES')
        if flags:
            cmd['flags'] = flags

        if self.arbitrary_check.isChecked():
            cmd['optional'] = True

        play_flags = []
        if self.no_cost_check.isChecked():
            play_flags.append('PLAY_FOR_FREE')
        if play_flags:
            cmd['play_flags'] = play_flags

        # Params
        if sel == 'ADD_KEYWORD':
             cmd['mutation_kind'] = self.mutation_kind_combo.currentData() or self.mutation_kind_edit.text()
        elif sel == 'MUTATE':
             cfg = self._get_ui_config(sel)
             if cfg.get('mutation_kind_visible'):
                 cmd['mutation_kind'] = self.mutation_kind_edit.text()
             if cfg.get('str_param_visible'):
                 cmd['str_param'] = self.str_edit.text()
        else:
             cmd['str_param'] = self.str_edit.text()

        cmd['from_zone'] = self.source_zone_combo.currentData()
        cmd['to_zone'] = self.dest_zone_combo.currentData()

        # Variable links
        try:
            cfg = self._get_ui_config(sel)
            produces = cfg.get('produces_output', False)
            if hasattr(self.link_widget, 'ensure_output_key'):
                self.link_widget.ensure_output_key(sel, produces)
        except Exception:
            pass
        self.link_widget.get_data(cmd)

        # Clear old data and update with new command
        data.clear()
        data.update(cmd)

    def _get_display_text(self, data):
        return f"{tr('Command')}: {tr(data.get('type', 'UNKNOWN'))}"
