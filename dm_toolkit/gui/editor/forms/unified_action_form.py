# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox,
    QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys, normalize_command_zone_keys
from dm_toolkit.consts import UNIFIED_ACTION_TYPES, COMMAND_TYPES, ZONES_EXTENDED, GRANTABLE_KEYWORDS
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG
from dm_toolkit.gui.editor.action_converter import ActionConverter
from dm_toolkit.gui.editor.consts import STRUCT_CMD_REPLACE_WITH_COMMAND, STRUCT_CMD_GENERATE_BRANCHES
from dm_toolkit.gui.editor.forms.convert_preview_dialog import ConvertPreviewDialog
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_scope_combo, make_value_spin, make_measure_mode_combo,
    make_ref_mode_combo, make_zone_combos, make_option_controls
)
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

# Grouping of action types for UI: top-level category -> subtypes
ACTION_GROUPS = {
    'MOVE': [
        'MOVE_CARD', 'SEND_TO_MANA', 'SEND_TO_DECK_BOTTOM', 'RETURN_TO_HAND', 'PUT_CREATURE', 'MOVE_TO_UNDER_CARD',
        'DISCARD', 'SEARCH_DECK'
    ],
    'QUERY': [
        'MEASURE_COUNT', 'COUNT_CARDS', 'GET_GAME_STAT', 'QUERY'
    ],
    'DRAW_PLAY': [
        'DRAW_CARD', 'PLAY_FROM_ZONE', 'CAST_SPELL', 'ADD_MANA'
    ],
    'GRANT': [
        'GRANT_KEYWORD', 'APPLY_MODIFIER', 'REGISTER_DELAYED_EFFECT', 'ADD_KEYWORD'
    ],
    'EFFECT': [
        'DESTROY', 'TAP', 'UNTAP', 'MODIFY_POWER', 'BREAK_SHIELD', 'POWER_MOD', 'MUTATE'
    ],
    'CONTROL': [
        'FLOW', 'TRANSITION', 'GAME_RESULT', 'SHIELD_TRIGGER'
    ],
    'OTHER': []
}


class UnifiedActionForm(BaseEditForm):
    """Unified editor capable of editing either Command-like or Legacy Action-like defs.

    Integrates standard Action fields and Command-specific fields (Mutation, Flow)
    using a seamless UI that adapts based on the selected Type.
    """
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure some attributes exist even if UI creation fails (headless/static checks)
        self.current_item = getattr(self, 'current_item', None)
        self._is_populating = False
        try:
            self.setup_ui()
        except Exception:
            # Defer full UI setup in environments without a QApplication
            pass

    def _get_ui_config(self, tpe):
        # Normalize UI config from COMMAND_UI_CONFIG or ACTION_UI_CONFIG into unified keys
        cmd_cfg = COMMAND_UI_CONFIG.get(tpe, {})
        act_cfg = ACTION_UI_CONFIG.get(tpe, {})

        # Source visible lists
        cmd_vis = set(cmd_cfg.get('visible', []))
        act_vis = set(act_cfg.get('visible', []))

        # Labels
        amount_label = cmd_cfg.get('label_amount') or act_cfg.get('label_value1') or 'Amount'
        val2_label = act_cfg.get('label_value2') or ''
        str_label = cmd_cfg.get('label_str_param') or act_cfg.get('label_str_val') or ''
        mutation_label = cmd_cfg.get('label_mutation_kind') or ''

        return {
            'target_group_visible': ('target_group' in cmd_vis) or ('scope' in act_vis),
            'target_filter_visible': ('target_filter' in cmd_vis) or ('filter' in act_vis),
            'amount_visible': ('amount' in cmd_vis) or ('value1' in act_vis),
            'amount_label': amount_label,
            'val2_visible': ('value2' in act_vis),
            'val2_label': val2_label,
            'str_param_visible': ('str_param' in cmd_vis) or ('str_val' in act_vis),
            'str_param_label': str_label,
            'mutation_kind_visible': ('mutation_kind' in cmd_vis) or ('str_val' in act_vis and tpe in ['GRANT_KEYWORD', 'ADD_KEYWORD']),
            'mutation_kind_label': mutation_label,
            'to_zone_visible': ('to_zone' in cmd_vis) or ('destination_zone' in act_vis),
            'from_zone_visible': ('from_zone' in cmd_vis) or ('source_zone' in act_vis),
            'optional_visible': ('optional' in cmd_vis) or act_cfg.get('can_be_optional', False),
            'input_link_visible': ('input_link' in cmd_vis) or ('target_choice' in act_vis),
            'produces_output': cmd_cfg.get('produces_output', False) or act_cfg.get('produces_output', False),
            'tooltip': cmd_cfg.get('tooltip', act_cfg.get('label', '')),
            'allowed_filter_fields': cmd_cfg.get('allowed_filter_fields', None),
            'can_be_optional': act_cfg.get('can_be_optional', False),
            'query_mode_visible': ('query_mode' in cmd_vis),
            'is_flow': (tpe == 'FLOW')
        }

    def setup_ui(self):
        layout = QFormLayout(self)

        # Warning / conversion affordance
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: orange; font-weight: bold;")
        self.warning_label.setVisible(False)
        layout.addRow(self.warning_label)

        # Convert Button (Action -> Command)
        self.convert_btn = QPushButton(tr("Convert to Command"))
        self.convert_btn.setStyleSheet("background-color: #ffcc80; color: black;")
        self.convert_btn.clicked.connect(self.on_convert_clicked)
        self.convert_btn.setVisible(False)
        layout.addRow(self.convert_btn)

        # Action Group (top-level) + Type (subtype)
        self.action_group_combo = QComboBox()
        groups = list(ACTION_GROUPS.keys())
        # Put OTHER at the end if present
        if 'OTHER' in groups:
            groups = [g for g in groups if g != 'OTHER'] + ['OTHER']
        self.populate_combo(self.action_group_combo, groups, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Action Group"), self.action_group_combo)

        self.type_combo = QComboBox()
        # Merge types from both systems
        self.known_types = sorted(list(set(UNIFIED_ACTION_TYPES + COMMAND_TYPES)))

        # Initially populate with all known types; on group change we will filter
        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Action Type"), self.type_combo)

        # Scope / Target Group
        self.scope_combo = make_scope_combo(self)
        layout.addRow(tr("Scope"), self.scope_combo)

        # Common string field (String Val / String Param)
        self.str_edit = QLineEdit()
        self.str_label = QLabel(tr("String"))
        layout.addRow(self.str_label, self.str_edit)

        # Measure Mode (for COUNT/GET_STAT) & Query Mode
        self.measure_mode_combo = make_measure_mode_combo(self)
        self.measure_mode_label = QLabel(tr("Mode"))
        layout.addRow(self.measure_mode_label, self.measure_mode_combo)

        # Stat Key selector for GET_GAME_STAT
        self.stat_key_combo = QComboBox()
        # Populate with known stat keys and their JP labels
        try:
            stat_keys = list(CardTextGenerator.STAT_KEY_MAP.keys())
            # display_func reads the Japanese label from STAT_KEY_MAP
            self.populate_combo(self.stat_key_combo, stat_keys, data_func=lambda x: x,
                                display_func=lambda k: CardTextGenerator.STAT_KEY_MAP.get(k, (k, ''))[0])
        except Exception:
            # Fallback: empty combo
            pass
        self.stat_key_label = QLabel(tr("Stat Key"))

        # Place combo and a small preset button in one row
        stat_row = QWidget()
        stat_row_layout = QHBoxLayout(stat_row)
        stat_row_layout.setContentsMargins(0, 0, 0, 0)
        stat_row_layout.addWidget(self.stat_key_combo)
        self.stat_preset_btn = QPushButton(tr("Preset: Mana Civs"))
        self.stat_preset_btn.setToolTip(tr("Set stat key to MANA_CIVILIZATION_COUNT"))
        self.stat_preset_btn.clicked.connect(lambda: self._preset_stat_key('MANA_CIVILIZATION_COUNT'))
        stat_row_layout.addWidget(self.stat_preset_btn)
        layout.addRow(self.stat_key_label, stat_row)

        # Ref Mode (for COST_REFERENCE)
        self.ref_mode_combo = make_ref_mode_combo(self)
        layout.addRow(tr("Ref Mode"), self.ref_mode_combo)

        # Amount / Value 1
        self.val1_spin = make_value_spin(self)
        self.val1_label = QLabel(tr("Value 1"))
        layout.addRow(self.val1_label, self.val1_spin)

        # Value 2
        self.val2_spin = make_value_spin(self)
        self.val2_label = QLabel(tr("Value 2"))
        layout.addRow(self.val2_label, self.val2_spin)

        # Option generation controls (for SELECT_OPTION)
        self.option_count_spin, self.generate_options_btn, self.option_count_label, self.option_gen_layout = make_option_controls(self)
        layout.addRow(self.option_count_label, self.option_gen_layout)

        # Branch generation button (visible only for FLOW commands)
        self.gen_branch_btn = QPushButton(tr("Generate Branches"))
        self.gen_branch_btn.setVisible(False)
        self.gen_branch_btn.clicked.connect(self.request_generate_branches)
        layout.addRow(self.gen_branch_btn)

        # Play without cost / allow duplicates / arbitrary amount
        self.no_cost_check = QCheckBox(tr("Play without paying cost"))
        self.no_cost_label = QLabel("")
        layout.addRow(self.no_cost_label, self.no_cost_check)
        self.allow_duplicates_check = QCheckBox(tr("Allow Duplicates"))
        self.allow_duplicates_label = QLabel("")
        layout.addRow(self.allow_duplicates_label, self.allow_duplicates_check)
        self.arbitrary_check = QCheckBox(tr("Arbitrary Amount (Up to N)"))
        self.arbitrary_label = QLabel("")
        layout.addRow(self.arbitrary_label, self.arbitrary_check)

        # Zones
        self.source_zone_combo, self.dest_zone_combo = make_zone_combos(self)
        self.source_zone_label = QLabel(tr("Source Zone"))
        self.dest_zone_label = QLabel(tr("Destination Zone"))
        layout.addRow(self.source_zone_label, self.source_zone_combo)
        layout.addRow(self.dest_zone_label, self.dest_zone_combo)

        # Filter
        self.filter_group = QGroupBox(tr("Filter"))
        self.filter_widget = FilterEditorWidget()
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)
        layout.addRow(self.filter_group)

        # Mutation kind (keywords)
        self.mutation_kind_edit = QLineEdit()
        self.mutation_kind_combo = QComboBox()
        self.populate_combo(self.mutation_kind_combo, GRANTABLE_KEYWORDS, data_func=lambda x: x, display_func=tr)
        self.mutation_kind_label = QLabel(tr("Mutation Kind"))

        # Use a stacked widget to host either edit or combo (allows switching based on type)
        self.mutation_kind_container = QStackedWidget()
        self.mutation_kind_container.addWidget(self.mutation_kind_edit)
        self.mutation_kind_container.addWidget(self.mutation_kind_combo)
        layout.addRow(self.mutation_kind_label, self.mutation_kind_container)

        # Variable link
        self.link_widget = VariableLinkWidget()
        self.link_widget.linkChanged.connect(self.update_data)
        # Respond to smart-link changes to adjust UI (hide amount when linked)
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
        # preset button signal handled inline above
        self.ref_mode_combo.currentIndexChanged.connect(self.update_data)
        self.generate_options_btn.clicked.connect(self.request_generate_options)
        self.no_cost_check.stateChanged.connect(self.update_data)
        self.allow_duplicates_check.stateChanged.connect(self.update_data)
        self.arbitrary_check.stateChanged.connect(self.update_data)
        self.convert_btn.clicked.connect(self.on_convert_clicked)
        self.scope_combo.currentIndexChanged.connect(self.update_data)
        self.str_edit.textChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.source_zone_combo.currentIndexChanged.connect(self.update_data)
        self.dest_zone_combo.currentIndexChanged.connect(self.update_data)
        self.mutation_kind_edit.textChanged.connect(self.update_data)
        self.mutation_kind_combo.currentIndexChanged.connect(self.update_data)

        self.update_ui_state(self.type_combo.currentData())

    def on_group_changed(self):
        """When a group is selected, populate the type combo with relevant subtypes."""
        grp = self.action_group_combo.currentData()
        types = ACTION_GROUPS.get(grp, [])
        if not types:
            # Fallback: show all known types
            types = self.known_types
        else:
            # Ensure types in the group are actually known/available
            types = [t for t in types if t in self.known_types]

        # repopulate type combo preserving selection if possible
        prev = self.type_combo.currentData()
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)
        # restore if still present
        if prev and prev in types:
            self.set_combo_by_data(self.type_combo, prev)
        else:
            self.type_combo.setCurrentIndex(0)
        # Propagate change to UI state
        try:
            self.update_ui_state(self.type_combo.currentData())
        except Exception:
            pass
        # Only update data if we are not in population phase
        if not self._is_populating:
            self.update_data()

    def on_type_changed(self):
        t = self.type_combo.currentData()
        # Show guidance depending on whether it's natively Command or Action
        # Note: In unified form we generally don't warn unless user specifically asks
        # but for compatibility we might hide conversion button if already Command
        if t in COMMAND_TYPES:
            self.warning_label.setText("")
            self.warning_label.setVisible(False)
            self.convert_btn.setVisible(False)
        else:
            self.warning_label.setText("")
            self.warning_label.setVisible(False)
            # Allow user to attempt conversion from Action -> Command when legacy selected
            self.convert_btn.setVisible(True)

        self.update_ui_state(t)

        # Ensure output key if needed
        if not self._is_populating:
            cfg = self._get_ui_config(t)
            if cfg.get('produces_output', False):
                try:
                    self.link_widget.ensure_output_key(t, True)
                except Exception:
                    pass
            self.update_data()

    def on_measure_mode_changed(self):
        """Handle measure mode selection: update string field and propagate change."""
        try:
            val = self.measure_mode_combo.currentData()
            # If we are in QUERY mode, we might update the measure_mode
            t = self.type_combo.currentData()
            if val:
                # store the selected mode into the string/value field used by converter
                # For QUERY type, this maps to str_param often
                self.str_edit.setText(val)
        except Exception:
            pass
        self.update_data()

    def on_smart_link_changed(self, is_active: bool):
        """Adjust UI when VariableLink smart-link toggles (hide amount, sync filter external count)."""
        try:
            cfg = self._get_ui_config(self.type_combo.currentData())
            # Amount visibility respects whether smart link is active
            amount_vis = cfg.get('amount_visible', False) and not is_active
            self.val1_label.setVisible(amount_vis)
            self.val1_spin.setVisible(amount_vis)
            # If filter supports external count, inform it
            if cfg.get('target_filter_visible', False) and hasattr(self.filter_widget, 'set_external_count_control'):
                self.filter_widget.set_external_count_control(is_active)
        except Exception:
            pass
        self.update_data()

    def on_stat_key_changed(self):
        """When a stat key is selected, mirror it into the string edit for consistency."""
        try:
            val = self.stat_key_combo.currentData()
            if val:
                # keep str_edit in sync so saving logic that prefers str_edit still works
                self.str_edit.setText(val)
        except Exception:
            pass
        self.update_data()

    def _preset_stat_key(self, key: str):
        try:
            if self.stat_key_combo:
                self.set_combo_by_data(self.stat_key_combo, key)
                # keep str_edit synced
                self.str_edit.setText(key)
        except Exception:
            try:
                self.str_edit.setText(key)
            except Exception:
                pass
        self.update_data()

    def request_generate_branches(self):
        """Request generation of Flow branches."""
        self.structure_update_requested.emit(STRUCT_CMD_GENERATE_BRANCHES, {})

    def request_generate_options(self):
        """Generate placeholder option chains for `SELECT_OPTION` types."""
        if not getattr(self, 'current_item', None):
            return
        try:
            count = int(self.option_count_spin.value())
        except Exception:
            count = 1

        new_options = []
        for _ in range(max(1, count)):
            # single-placeholder action per option chain
            new_options.append([{"type": "NONE"}])

        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2) or {}
        data['options'] = new_options
        data['type'] = 'SELECT_OPTION'

        # Persist back to the item and refresh UI
        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        # Re-populate UI to reflect changes
        try:
            self.set_data(self.current_item)
        except Exception:
            # Fallback: emit dataChanged so outer logic updates
            self.dataChanged.emit()

    def on_convert_clicked(self):
        # Convert current item (legacy action) into a Command via ActionConverter
        if not getattr(self, 'current_item', None):
            return
        act_data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
        new_cmd = ActionConverter.convert(act_data)

        # Prompt with preview
        dialog = ConvertPreviewDialog(self, act_data, new_cmd)
        res = dialog.exec()
        if res == dialog.Accepted:
            payload = {
                'new_data': new_cmd,
                'target_item': self.current_item
            }
            self.structure_update_requested.emit(STRUCT_CMD_REPLACE_WITH_COMMAND, payload)

    def update_ui_state(self, t):
        # Use UI config from whichever side defines it
        cfg = self._get_ui_config(t)
        # Apply normalized config
        self.scope_combo.setVisible(cfg.get('target_group_visible', False))
        self.filter_group.setVisible(cfg.get('target_filter_visible', False))

        # Amount / value1
        self.val1_label.setVisible(cfg.get('amount_visible', False))
        self.val1_spin.setVisible(cfg.get('amount_visible', False))
        self.val1_label.setText(tr(cfg.get('amount_label', 'Amount')))

        # Value2
        self.val2_label.setVisible(cfg.get('val2_visible', False))
        self.val2_spin.setVisible(cfg.get('val2_visible', False))
        if cfg.get('val2_label'):
            self.val2_label.setText(tr(cfg.get('val2_label')))

        # String param
        self.str_label.setVisible(cfg.get('str_param_visible', False))
        self.str_edit.setVisible(cfg.get('str_param_visible', False))
        if cfg.get('str_param_label'):
            self.str_label.setText(tr(cfg.get('str_param_label')))

        # Mutation kind (use stacked container to switch between edit/combo)
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

        # Produce output guidance
        produces = cfg.get('produces_output', False)
        self.link_widget.set_output_hint(produces) if hasattr(self.link_widget, 'set_output_hint') else None

        # Measure / Ref (special cases)
        is_measure = (t == 'MEASURE_COUNT' or t == 'COUNT_CARDS' or t == 'GET_GAME_STAT' or t == 'QUERY')
        self.measure_mode_combo.setVisible(is_measure)
        self.measure_mode_label.setVisible(is_measure)

        self.stat_key_label.setVisible(t == 'GET_GAME_STAT')
        self.stat_key_combo.setVisible(t == 'GET_GAME_STAT')
        self.stat_preset_btn.setVisible(t == 'GET_GAME_STAT')
        self.ref_mode_combo.setVisible(t == 'COST_REFERENCE')

        # Option-related visibility
        self.option_count_label.setVisible(t == 'SELECT_OPTION')
        self.option_count_spin.setVisible(t == 'SELECT_OPTION')
        self.generate_options_btn.setVisible(t == 'SELECT_OPTION')
        self.allow_duplicates_label.setVisible(t == 'SELECT_OPTION')
        self.allow_duplicates_check.setVisible(t == 'SELECT_OPTION')

        # Branch generation
        self.gen_branch_btn.setVisible(cfg.get('is_flow', False))

        # No-cost / arbitrary
        self.no_cost_label.setVisible(t == 'PLAY_FROM_ZONE')
        self.no_cost_check.setVisible(t == 'PLAY_FROM_ZONE')
        self.arbitrary_label.setVisible(cfg.get('can_be_optional', False) or cfg.get('optional_visible', False))
        self.arbitrary_check.setVisible(cfg.get('can_be_optional', False) or cfg.get('optional_visible', False))

        # Tooltip
        self.type_combo.setToolTip(tr(cfg.get('tooltip', '')))

        # Mutation kind display mode
        is_add_keyword = (t == 'ADD_KEYWORD')
        try:
            if self.mutation_kind_container:
                self.mutation_kind_container.setCurrentIndex(1 if is_add_keyword else 0)
        except Exception:
            pass

    def _populate_ui(self, item):
        self._is_populating = True
        try:
            self.link_widget.set_current_item(item)
            data = item.data(Qt.ItemDataRole.UserRole + 2)

            # Normalize both action and command zone keys if present
            normalize_action_zone_keys(data)
            normalize_command_zone_keys(data)

            ui_type = data.get('type', 'NONE')
            # Set type combo and derive group selection
            self.set_combo_by_data(self.type_combo, ui_type)
            # Determine group for this type
            grp = None
            for g, types in ACTION_GROUPS.items():
                if ui_type in types:
                    grp = g
                    break
            if grp is None:
                grp = 'OTHER'
            try:
                self.set_combo_by_data(self.action_group_combo, grp)
            except Exception:
                pass

            self.set_combo_by_data(self.scope_combo, data.get('scope', data.get('target_group', 'NONE')))

            # String params
            self.str_edit.setText(data.get('str_val', data.get('str_param', '')))

            # Mode mapping
            if ui_type == 'MEASURE_COUNT' or ui_type == 'QUERY':
                val = data.get('str_param', data.get('str_val', 'CARDS_MATCHING_FILTER'))
                if not val: val = 'CARDS_MATCHING_FILTER'
                if ui_type == 'COUNT_CARDS' and not val: val = 'CARDS_MATCHING_FILTER'
                self.set_combo_by_data(self.measure_mode_combo, val)

            if ui_type == 'GET_GAME_STAT':
                # Set stat key combo from stored str_val if present
                key = data.get('str_val', '')
                if key:
                    try:
                        self.set_combo_by_data(self.stat_key_combo, key)
                    except Exception:
                        # fallback to set text
                        self.str_edit.setText(key)

            if ui_type == 'COST_REFERENCE':
                self.set_combo_by_data(self.ref_mode_combo, data.get('str_val', ''))

            self.val1_spin.setValue(data.get('value1', data.get('amount', 0)))
            self.val2_spin.setValue(data.get('value2', 0))

            self.set_combo_by_data(self.source_zone_combo, data.get('source_zone', data.get('from_zone', 'NONE')))
            self.set_combo_by_data(self.dest_zone_combo, data.get('destination_zone', data.get('to_zone', 'NONE')))

            self.filter_widget.set_data(data.get('filter', data.get('target_filter', {})))
            self.link_widget.set_data(data)

            # Mutation kind population
            mutation_kind = data.get('mutation_kind', data.get('str_val', ''))
            try:
                # prefer combo when matching known keywords
                if mutation_kind and mutation_kind in GRANTABLE_KEYWORDS:
                    self.set_combo_by_data(self.mutation_kind_combo, mutation_kind)
                else:
                    self.mutation_kind_edit.setText(mutation_kind)
            except Exception:
                try:
                    self.mutation_kind_edit.setText(mutation_kind)
                except Exception:
                    pass
            # If type is ADD_KEYWORD, switch container
            try:
                if ui_type == 'ADD_KEYWORD' and self.mutation_kind_container:
                    self.mutation_kind_container.setCurrentIndex(1)
                elif self.mutation_kind_container:
                    self.mutation_kind_container.setCurrentIndex(0)
            except Exception:
                pass

            self.update_ui_state(ui_type)

            # Option flags
            if ui_type == 'SELECT_OPTION':
                self.allow_duplicates_check.setChecked(data.get('value2', 0) == 1)
                self.option_count_spin.setValue(data.get('value1', 1))

            # Optional / Arbitrary
            self.arbitrary_check.setChecked(data.get('optional', False))

        finally:
            self._is_populating = False

    def _save_data(self, data):
        sel = self.type_combo.currentData()
        # Persist selected group for clarity (editor-only metadata)
        try:
            grp = self.action_group_combo.currentData()
            if grp:
                data['group'] = grp
        except Exception:
            pass
        data['type'] = sel

        # If this type is a native Command, write command-shaped keys
        if sel in COMMAND_TYPES:
            cmd = {}
            cmd['type'] = sel
            cmd['target_group'] = self.scope_combo.currentData()
            cmd['target_filter'] = self.filter_widget.get_data()
            cmd['amount'] = self.val1_spin.value()
            cmd['optional'] = self.arbitrary_check.isChecked()
            # mutation kind handling
            if sel == 'ADD_KEYWORD':
                cmd['mutation_kind'] = self.mutation_kind_combo.currentData() or self.mutation_kind_edit.text()
            else:
                cmd['mutation_kind'] = self.mutation_kind_edit.text()

            # str param (some commands use this)
            if sel == 'QUERY':
                mode = self.measure_mode_combo.currentData()
                if mode and mode != 'CARDS_MATCHING_FILTER':
                    cmd['str_param'] = mode
                else:
                    cmd['str_param'] = self.str_edit.text()
            else:
                cmd['str_param'] = self.str_edit.text()

            # zones
            cmd['from_zone'] = self.source_zone_combo.currentData()
            cmd['to_zone'] = self.dest_zone_combo.currentData()

            # Variable links / outputs: ensure output key if needed, then collect
            try:
                cfg = self._get_ui_config(sel)
                produces = cfg.get('produces_output', False)
                if hasattr(self.link_widget, 'ensure_output_key'):
                    self.link_widget.ensure_output_key(sel, produces)
            except Exception:
                pass
            self.link_widget.get_data(cmd)

            # Preserve legacy flags only if present
            if data.get('legacy_warning'):
                cmd['legacy_warning'] = data.get('legacy_warning')
                cmd['legacy_original_type'] = data.get('legacy_original_type')

            # Overwrite data with command structure
            data.clear()
            data.update(cmd)
            data['format'] = 'command'
            return

        # Non-command path: gather action-like fields
        data['format'] = 'action'
        data['scope'] = self.scope_combo.currentData()
        # For GET_GAME_STAT prefer stat_key_combo if visible
        try:
            if sel == 'GET_GAME_STAT' and self.stat_key_combo and self.stat_key_combo.currentData():
                data['str_val'] = self.stat_key_combo.currentData()
            else:
                data['str_val'] = self.str_edit.text()
        except Exception:
            data['str_val'] = self.str_edit.text()

        data['value1'] = self.val1_spin.value()
        data['value2'] = self.val2_spin.value()
        data['source_zone'] = self.source_zone_combo.currentData()
        data['destination_zone'] = self.dest_zone_combo.currentData()
        data['filter'] = self.filter_widget.get_data()

        # Ensure output key if query/action produces output
        try:
            cfg = self._get_ui_config(sel)
            produces = cfg.get('produces_output', False)
            if hasattr(self.link_widget, 'ensure_output_key'):
                self.link_widget.ensure_output_key(sel, produces)
        except Exception:
            pass
        self.link_widget.get_data(data)

        # Legacy marker handling
        if sel and sel not in COMMAND_TYPES:
            data['legacy_warning'] = True
            data['legacy_original_type'] = sel
        else:
            data.pop('legacy_warning', None)
            data.pop('legacy_original_type', None)

        # Option flags
        if sel == 'SELECT_OPTION':
            data['value1'] = self.option_count_spin.value()
            data['value2'] = 1 if self.allow_duplicates_check.isChecked() else 0

        # No-cost handling
        if sel == 'PLAY_FROM_ZONE' and self.no_cost_check.isChecked():
            data['value1'] = 999

    def _get_display_text(self, data):
        # Display as an Action-like label for the tree
        return f"{tr('Action')}: {tr(data.get('type', 'UNKNOWN'))}"

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.scope_combo.blockSignals(block)
        self.str_edit.blockSignals(block)
        self.val1_spin.blockSignals(block)
        self.val2_spin.blockSignals(block)
        self.source_zone_combo.blockSignals(block)
        self.dest_zone_combo.blockSignals(block)
        self.filter_widget.blockSignals(block)
        self.link_widget.blockSignals(block)
