# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
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
from dm_toolkit.gui.editor.consts import STRUCT_CMD_REPLACE_WITH_COMMAND
from dm_toolkit.gui.editor.forms.convert_preview_dialog import ConvertPreviewDialog


class UnifiedActionForm(BaseEditForm):
    """Unified editor capable of editing either Command-like or Legacy Action-like defs.

    This is a pragmatic skeleton: it exposes a unified type combo (UNIFIED_ACTION_TYPES)
    and common fields. On save it marks the data with `format: 'command'|'action'`
    so higher-level logic can serialize as CommandDef or ActionDef.
    """
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

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
            'can_be_optional': act_cfg.get('can_be_optional', False)
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

        self.type_combo = QComboBox()
        self.known_types = UNIFIED_ACTION_TYPES
        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Action Type"), self.type_combo)

        # Scope / Target Group
        self.scope_combo = QComboBox()
        scopes = ["PLAYER_SELF","PLAYER_OPPONENT","TARGET_SELECT","ALL_PLAYERS","RANDOM","ALL_FILTERED","NONE"]
        self.populate_combo(self.scope_combo, scopes, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Scope"), self.scope_combo)

        # Common fields
        self.str_edit = QLineEdit()
        self.str_label = QLabel(tr("String"))
        layout.addRow(self.str_label, self.str_edit)

        # Measure Mode (for COUNT/GET_STAT)
        self.measure_mode_combo = QComboBox()
        self.measure_mode_combo.addItem(tr("CARDS_MATCHING_FILTER"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
        self.populate_combo(self.measure_mode_combo, stats, data_func=lambda x: x, display_func=tr, clear=False)
        layout.addRow(tr("Mode"), self.measure_mode_combo)

        # Ref Mode (for COST_REFERENCE)
        self.ref_mode_combo = QComboBox()
        ref_modes = ["SYM_CREATURE", "SYM_SPELL", "G_ZERO", "HYPER_ENERGY", "NONE"]
        self.populate_combo(self.ref_mode_combo, ref_modes, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Ref Mode"), self.ref_mode_combo)

        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(-9999, 9999)
        self.val1_label = QLabel(tr("Value 1"))
        layout.addRow(self.val1_label, self.val1_spin)

        self.val2_spin = QSpinBox()
        self.val2_spin.setRange(-9999, 9999)
        self.val2_label = QLabel(tr("Value 2"))
        layout.addRow(self.val2_label, self.val2_spin)

        # Option generation controls (for SELECT_OPTION)
        self.option_count_spin = QSpinBox()
        self.option_count_spin.setRange(1, 10)
        self.option_count_spin.setValue(1)
        self.generate_options_btn = QPushButton(tr("Generate Options"))
        self.option_gen_layout = QHBoxLayout()
        self.option_gen_layout.addWidget(self.option_count_spin)
        self.option_gen_layout.addWidget(self.generate_options_btn)
        self.option_count_label = QLabel(tr("Options to Add"))
        layout.addRow(self.option_count_label, self.option_gen_layout)

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
        self.source_zone_combo = QComboBox()
        self.dest_zone_combo = QComboBox()
        self.populate_combo(self.source_zone_combo, ZONES_EXTENDED, data_func=lambda x: x, display_func=tr)
        self.populate_combo(self.dest_zone_combo, ZONES_EXTENDED, data_func=lambda x: x, display_func=tr)
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
        layout.addRow(self.mutation_kind_label, self.mutation_kind_edit)

        # Variable link
        self.link_widget = VariableLinkWidget()
        self.link_widget.linkChanged.connect(self.update_data)
        layout.addRow(self.link_widget)

        # Signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.measure_mode_combo.currentIndexChanged.connect(self.on_measure_mode_changed)
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

        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        t = self.type_combo.currentData()
        # Show guidance depending on whether it's natively Command or Action
        if t in COMMAND_TYPES:
            self.warning_label.setText(tr("This type maps to a native Command. It will be saved as a Command."))
            self.warning_label.setVisible(True)
            self.convert_btn.setVisible(False)
        else:
            self.warning_label.setText("")
            self.warning_label.setVisible(False)
            # Allow user to attempt conversion from Action -> Command when legacy selected
            self.convert_btn.setVisible(True)
        self.update_ui_state(t)
        self.update_data()

    def on_convert_clicked(self):
        # Convert current item (legacy action) into a Command via ActionConverter
        if not getattr(self, 'current_item', None):
            return
        act_data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
        new_cmd = ActionConverter.convert(act_data)

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

        # Mutation kind
        self.mutation_kind_label.setVisible(cfg.get('mutation_kind_visible', False))
        self.mutation_kind_edit.setVisible(cfg.get('mutation_kind_visible', False))
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
        self.measure_mode_combo.setVisible(t == 'MEASURE_COUNT' or t == 'COUNT_CARDS' or t == 'GET_GAME_STAT')
        self.ref_mode_combo.setVisible(t == 'COST_REFERENCE')

        # Option-related visibility
        self.option_count_label.setVisible(t == 'SELECT_OPTION')
        self.option_count_spin.setVisible(t == 'SELECT_OPTION')
        self.generate_options_btn.setVisible(t == 'SELECT_OPTION')
        self.allow_duplicates_label.setVisible(t == 'SELECT_OPTION')
        self.allow_duplicates_check.setVisible(t == 'SELECT_OPTION')

        # No-cost / arbitrary
        self.no_cost_label.setVisible(t == 'PLAY_FROM_ZONE')
        self.no_cost_check.setVisible(t == 'PLAY_FROM_ZONE')
        self.arbitrary_label.setVisible(cfg.get('can_be_optional', False))
        self.arbitrary_check.setVisible(cfg.get('can_be_optional', False))

        # Tooltip
        self.type_combo.setToolTip(tr(cfg.get('tooltip', '')))

    def _populate_ui(self, item):
        self.link_widget.set_current_item(item)
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        # Normalize both action and command zone keys if present
        normalize_action_zone_keys(data)
        normalize_command_zone_keys(data)

        ui_type = data.get('type', 'NONE')
        self.set_combo_by_data(self.type_combo, ui_type)
        self.set_combo_by_data(self.scope_combo, data.get('scope', data.get('target_group', 'NONE')))
        self.str_edit.setText(data.get('str_val', data.get('str_param', '')))
        # Mode mapping
        if ui_type == 'MEASURE_COUNT':
            val = 'CARDS_MATCHING_FILTER' if data.get('type') == 'COUNT_CARDS' else data.get('str_val', 'CARDS_MATCHING_FILTER')
            self.set_combo_by_data(self.measure_mode_combo, val)
        if ui_type == 'COST_REFERENCE':
            self.set_combo_by_data(self.ref_mode_combo, data.get('str_val', ''))
        self.val1_spin.setValue(data.get('value1', data.get('amount', 0)))
        self.val2_spin.setValue(data.get('value2', 0))
        self.set_combo_by_data(self.source_zone_combo, data.get('source_zone', data.get('from_zone', 'NONE')))
        self.set_combo_by_data(self.dest_zone_combo, data.get('destination_zone', data.get('to_zone', 'NONE')))
        self.filter_widget.set_data(data.get('filter', data.get('target_filter', {})))
        self.link_widget.set_data(data)
        self.update_ui_state(ui_type)

        # Option flags
        if ui_type == 'SELECT_OPTION':
            self.allow_duplicates_check.setChecked(data.get('value2', 0) == 1)
            self.option_count_spin.setValue(data.get('value1', 1))

    def _save_data(self, data):
        sel = self.type_combo.currentData()
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
                # where applicable, use str_edit
                cmd['str_param'] = self.str_edit.text()

            # zones
            cmd['from_zone'] = self.source_zone_combo.currentData()
            cmd['to_zone'] = self.dest_zone_combo.currentData()

            # Variable links / outputs
            self.link_widget.get_data(cmd)

            # option-specific
            if sel == 'SELECT_OPTION':
                cmd['amount'] = self.option_count_spin.value()
                if self.allow_duplicates_check.isChecked():
                    cmd.setdefault('flags', []).append('ALLOW_DUPLICATES')

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
        data['str_val'] = self.str_edit.text()
        data['value1'] = self.val1_spin.value()
        data['value2'] = self.val2_spin.value()
        data['source_zone'] = self.source_zone_combo.currentData()
        data['destination_zone'] = self.dest_zone_combo.currentData()
        data['filter'] = self.filter_widget.get_data()

        self.link_widget.get_data(data)

        # Attempt to auto-convert action-like data to Command where possible
        try:
            act_like = dict(data)
            act_like['type'] = sel
            conv = ActionConverter.convert(act_like)
            # If conversion looks fully OK, adopt silently
            if conv and conv.get('type') != 'NONE' and not conv.get('legacy_warning', False):
                data.clear()
                data.update(conv)
                data['format'] = 'command'
                return

            # Otherwise show preview dialog so user can decide
            dialog = ConvertPreviewDialog(self, act_like, conv or {})
            res = dialog.exec()
            if res == dialog.Accepted:
                # User chose to use converted command (even if partial)
                data.clear()
                data.update(conv or {})
                data['format'] = 'command'
                # ensure legacy flags if present
                if conv and conv.get('legacy_warning'):
                    data['legacy_warning'] = True
                    data['legacy_original_type'] = sel
                return
            elif res == dialog.Rejected:
                # Keep as action; mark legacy if conv indicated problems
                if conv and conv.get('legacy_warning'):
                    data['legacy_warning'] = True
                    data['legacy_original_type'] = sel
                return
            else:
                # Cancel: raise to prevent save (leave data unchanged)
                raise RuntimeError('User cancelled conversion')
        except Exception:
            # conversion failed or user cancelled; mark legacy and keep action
            data['legacy_warning'] = True
            data['legacy_original_type'] = sel

        # Save option/duplicates flags
        if sel == 'SELECT_OPTION':
            data['value1'] = self.option_count_spin.value()
            data['value2'] = 1 if self.allow_duplicates_check.isChecked() else 0

        # No-cost handling
        if sel == 'PLAY_FROM_ZONE' and self.no_cost_check.isChecked():
            data['value1'] = 999

        # If the selected type is legacy but can be converted, prefer storing as Command
        if sel and sel not in COMMAND_TYPES:
            # Build a minimal action-like dict from UI to feed converter
            act_like = dict(data)  # start from current collected data
            # Ensure keys ActionConverter expects
            act_like['type'] = sel
            # Try conversion
            try:
                conv = ActionConverter.convert(act_like)
                # If conversion produced a usable command (not NONE or still legacy_warning), adopt it
                if conv.get('type') != 'NONE' and not conv.get('legacy_warning', False):
                    # Replace data contents with converted command
                    data.clear()
                    data.update(conv)
                    data['format'] = 'command'
                else:
                    # Keep as action, but preserve legacy flags
                    data['legacy_warning'] = True
                    data['legacy_original_type'] = sel
            except Exception:
                # If conversion failed, leave as action and mark legacy
                data['legacy_warning'] = True
                data['legacy_original_type'] = sel

        # Legacy marker handling
        if sel and sel not in COMMAND_TYPES:
            data['legacy_warning'] = True
            data['legacy_original_type'] = sel
        else:
            data.pop('legacy_warning', None)
            data.pop('legacy_original_type', None)

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
