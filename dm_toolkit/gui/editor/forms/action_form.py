# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.consts import EDITOR_ACTION_TYPES, ZONES_EXTENDED
from dm_toolkit.gui.editor.consts import STRUCT_CMD_GENERATE_OPTIONS

class ActionEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        # Removed enforce_readonly_mode

    def _get_ui_config(self, action_type):
        """Normalize raw UI config with safe defaults to avoid missing keys."""
        raw = ACTION_UI_CONFIG.get(action_type, {})

        # Map UI-only types to underlying configs when needed
        if action_type == "MEASURE_COUNT":
            raw = ACTION_UI_CONFIG.get("MEASURE_COUNT", ACTION_UI_CONFIG.get("COUNT_CARDS", raw))
        elif action_type == "COST_REDUCTION":
            raw = ACTION_UI_CONFIG.get("COST_REDUCTION", raw)

        visible = raw.get("visible", [])
        vis = lambda key: key in visible

        return {
            "val1_label": raw.get("label_value1", "Value 1"),
            "val2_label": raw.get("label_value2", "Value 2"),
            "str_label": raw.get("label_str_val", "String Value"),
            "val1_visible": vis("value1"),
            "val2_visible": vis("value2"),
            "str_visible": vis("str_val"),
            "filter_visible": vis("filter"),
            "dest_zone_visible": vis("destination_zone"),
            "input_link_visible": vis("input_value_key") or raw.get("outputs", {}).get("output_value_key"), # Corrected logic
            "can_be_optional": raw.get("can_be_optional", False),
            "produces_output": raw.get("produces_output", False),
            "tooltip": raw.get("tooltip", ""),
            "allowed_filter_fields": raw.get("allowed_filter_fields", None)
            # Removed deprecated flag
        }

    def setup_ui(self):
        self.form_layout = QFormLayout(self)

        self.type_combo = QComboBox()
        self.known_types = EDITOR_ACTION_TYPES
        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        self.add_field(tr("Action Type"), self.type_combo)

        # Removed warning label

        self.scope_combo = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"]
        self.populate_combo(self.scope_combo, scopes, data_func=lambda x: x, display_func=tr)
        self.add_field(tr("Scope"), self.scope_combo)

        self.str_val_edit = QLineEdit()
        self.str_val_label = self.add_field(tr("String Value"), self.str_val_edit)

        # Measure Mode Combo (Unified)
        self.measure_mode_combo = QComboBox()
        self.measure_mode_combo.addItem(tr("CARDS_MATCHING_FILTER"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
        self.populate_combo(self.measure_mode_combo, stats, data_func=lambda x: x, display_func=tr, clear=False)
        self.mode_label = self.add_field(tr("Mode"), self.measure_mode_combo)

        self.ref_mode_combo = QComboBox()
        ref_modes = ["SYM_CREATURE", "SYM_SPELL", "G_ZERO", "HYPER_ENERGY", "NONE"]
        self.populate_combo(self.ref_mode_combo, ref_modes, data_func=lambda x: x, display_func=tr)
        self.ref_mode_label = self.add_field(tr("Ref Mode"), self.ref_mode_combo)

        # Destination Zone Combo (For MOVE_CARD)
        self.dest_zone_combo = QComboBox()
        zones = ZONES_EXTENDED
        self.populate_combo(self.dest_zone_combo, zones, data_func=lambda x: x, display_func=tr)
        self.dest_zone_label = self.add_field(tr("Destination Zone"), self.dest_zone_combo)

        # Filter Widget
        self.filter_group = QGroupBox(tr("Filter"))
        self.filter_widget = FilterEditorWidget()
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)

        self.add_field(None, self.filter_group)

        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(-9999, 9999)
        self.val1_label = self.add_field(tr("Value 1"), self.val1_spin)

        self.val2_spin = QSpinBox()
        self.val2_spin.setRange(-9999, 9999)
        self.val2_label = self.add_field(tr("Value 2"), self.val2_spin)

        # Option Generation Controls (For SELECT_OPTION)
        self.option_gen_layout = QHBoxLayout()
        self.option_count_spin = QSpinBox()
        self.option_count_spin.setRange(1, 10)
        self.option_count_spin.setValue(1)
        self.generate_options_btn = QPushButton(tr("Generate Options"))
        self.generate_options_btn.clicked.connect(self.request_generate_options)
        self.option_gen_layout.addWidget(self.option_count_spin)
        self.option_gen_layout.addWidget(self.generate_options_btn)
        self.option_count_label = self.add_field(tr("Options to Add"), self.option_gen_layout)


        self.no_cost_check = QCheckBox(tr("Play without paying cost"))
        self.add_field(None, self.no_cost_check)

        self.allow_duplicates_check = QCheckBox(tr("Allow Duplicates"))
        self.add_field(None, self.allow_duplicates_check)

        self.arbitrary_check = QCheckBox(tr("Arbitrary Amount (Up to N)"))
        self.add_field(None, self.arbitrary_check)

        # Variable Link Widget
        self.link_widget = VariableLinkWidget()
        self.link_widget.linkChanged.connect(self.update_data)
        self.link_widget.smartLinkStateChanged.connect(self.on_smart_link_changed)
        self.add_field(None, self.link_widget)

        # Define bindings for automatic data transfer
        self.bindings = {
            'scope': self.scope_combo,
            'destination_zone': self.dest_zone_combo,
            'filter': self.filter_widget,
            'value1': self.val1_spin,
            'value2': self.val2_spin,
            'optional': self.arbitrary_check,
        }

        # Connect signals - Re-enabled
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.scope_combo.currentIndexChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.str_val_edit.textChanged.connect(self.update_data)
        self.measure_mode_combo.currentIndexChanged.connect(self.on_measure_mode_changed)
        self.ref_mode_combo.currentIndexChanged.connect(self.update_data)
        self.dest_zone_combo.currentIndexChanged.connect(self.update_data)
        self.allow_duplicates_check.stateChanged.connect(self.update_data)
        self.arbitrary_check.stateChanged.connect(self.update_data)
        self.no_cost_check.stateChanged.connect(self.update_data)
        self.no_cost_check.stateChanged.connect(self.on_no_cost_changed)

        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        action_type = self.type_combo.currentData()
        self.update_ui_state(action_type)
        self.update_data()

    def on_measure_mode_changed(self):
        self.update_ui_state(self.type_combo.currentData())
        self.update_data()

    def on_smart_link_changed(self, is_active):
        action_type = self.type_combo.currentData()
        if not action_type: return

        config = self._get_ui_config(action_type)

        is_no_cost = self.no_cost_check.isChecked()
        # Hide Val1 if Smart Link is active OR Val1 is hidden by config
        self.val1_label.setVisible(config["val1_visible"] and not is_active and not is_no_cost)
        self.val1_spin.setVisible(config["val1_visible"] and not is_active and not is_no_cost)

    def on_no_cost_changed(self, state):
        self.on_smart_link_changed(self.link_widget.is_smart_link_active())

    def update_ui_state(self, action_type):
        if not action_type: return

        config = self._get_ui_config(action_type)

        self.val1_label.setText(tr(config["val1_label"]))
        self.val2_label.setText(tr(config["val2_label"]))
        self.str_val_label.setText(tr(config["str_label"]))

        # Enable input linking if Val1 is visible OR explicitly allowed (e.g. for count override)
        # Note: input_link_visible logic in _get_ui_config might need adjustment based on config structure
        can_link_input = config["val1_visible"] or config.get("input_link_visible", False)
        self.link_widget.set_smart_link_enabled(can_link_input)

        self.arbitrary_check.setVisible(config.get("can_be_optional", False))
        self.no_cost_check.setVisible(action_type == "PLAY_FROM_ZONE")

        is_smart_linked = self.link_widget.is_smart_link_active() and can_link_input
        is_no_cost = self.no_cost_check.isChecked() and (action_type == "PLAY_FROM_ZONE")

        self.val1_label.setVisible(config["val1_visible"] and not is_smart_linked and not is_no_cost)
        self.val1_spin.setVisible(config["val1_visible"] and not is_smart_linked and not is_no_cost)

        # Sync filter widget state with link state
        if config["filter_visible"]:
             self.filter_widget.set_external_count_control(is_smart_linked)

        is_select_option = (action_type == "SELECT_OPTION")
        self.allow_duplicates_check.setVisible(is_select_option)

        # Show Option Generator for SELECT_OPTION
        self.option_count_label.setVisible(is_select_option)
        self.option_count_spin.setVisible(is_select_option)
        self.generate_options_btn.setVisible(is_select_option)

        self.val2_label.setVisible(config["val2_visible"] and not is_select_option)
        self.val2_spin.setVisible(config["val2_visible"] and not is_select_option)

        # Specific Widget Visibility
        is_measure = (action_type == "MEASURE_COUNT")
        is_ref = (action_type == "COST_REFERENCE")

        self.str_val_label.setVisible(config["str_visible"] and not is_measure and not is_ref)
        self.str_val_edit.setVisible(config["str_visible"] and not is_measure and not is_ref)

        self.mode_label.setVisible(is_measure)
        self.measure_mode_combo.setVisible(is_measure)

        self.ref_mode_label.setVisible(is_ref)
        self.ref_mode_combo.setVisible(is_ref)

        self.dest_zone_label.setVisible(config.get("dest_zone_visible", False))
        self.dest_zone_combo.setVisible(config.get("dest_zone_visible", False))

        # Filter Visibility
        show_filter = config["filter_visible"]
        if is_measure:
             # Hide filter if mode is Game Stat (not CARDS_MATCHING_FILTER)
             current_mode = self.measure_mode_combo.currentData()
             if current_mode != "CARDS_MATCHING_FILTER" and current_mode is not None:
                  show_filter = False

        self.filter_group.setVisible(show_filter)
        if show_filter:
             self.filter_widget.set_allowed_fields(config.get("allowed_filter_fields", None))

        self.type_combo.setToolTip(tr(config.get("tooltip", "")))

    def _populate_ui(self, item):
        self.link_widget.set_current_item(item)
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        raw_type = data.get('type', 'NONE')
        str_val = data.get('str_val', '')

        # Map Internal Type -> UI Type
        ui_type = raw_type
        if raw_type == "COUNT_CARDS" or raw_type == "GET_GAME_STAT":
             ui_type = "MEASURE_COUNT"
        elif raw_type == "APPLY_MODIFIER" and str_val == "COST":
             ui_type = "COST_REDUCTION"
        elif raw_type in ["DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DISCARD", "SEND_TO_DECK_BOTTOM", "SEND_SHIELD_TO_GRAVE", "SEND_TO_MANA"]:
             # If legacy type is not in known_types, temporarily add it to combo
             if raw_type not in self.known_types:
                 idx = self.type_combo.findData(raw_type)
                 if idx == -1:
                     self.type_combo.addItem(f"{tr(raw_type)} (Legacy)", raw_type)

        self.set_combo_by_data(self.type_combo, ui_type)

        # Map Values to Combos
        if ui_type == "MEASURE_COUNT":
             val = "CARDS_MATCHING_FILTER" if raw_type == "COUNT_CARDS" else str_val
             self.set_combo_by_data(self.measure_mode_combo, val)
        elif ui_type == "COST_REFERENCE":
             self.set_combo_by_data(self.ref_mode_combo, str_val)

        # Use bindings for standard fields
        self._apply_bindings(data)

        # Handle special cases
        self.update_ui_state(ui_type)

        val1 = data.get('value1', 0)
        self.no_cost_check.setChecked(ui_type == "PLAY_FROM_ZONE" and val1 >= 999)

        if ui_type == "SELECT_OPTION":
             self.allow_duplicates_check.setChecked(data.get('value2', 0) == 1)

        if ui_type != "MEASURE_COUNT" and ui_type != "COST_REFERENCE":
             self.str_val_edit.setText(str_val)

        self.link_widget.set_data(data)

    def request_generate_options(self):
        count = self.option_count_spin.value()
        self.structure_update_requested.emit(STRUCT_CMD_GENERATE_OPTIONS, {"count": count})

    def _save_data(self, data):
        ui_type = self.type_combo.currentData()

        # Default mapping
        raw_type = ui_type
        str_val = self.str_val_edit.text()

        if ui_type == "MEASURE_COUNT":
            mode = self.measure_mode_combo.currentData()
            if mode == "CARDS_MATCHING_FILTER":
                raw_type = "COUNT_CARDS"
                # Keep str_val as is or empty? Usually empty for COUNT_CARDS unless specified
                str_val = ""
            else:
                raw_type = "GET_GAME_STAT"
                str_val = mode
        elif ui_type == "COST_REDUCTION":
            raw_type = "APPLY_MODIFIER"
            str_val = "COST"
        elif ui_type == "COST_REFERENCE":
             str_val = self.ref_mode_combo.currentData()

        data['type'] = raw_type
        data['str_val'] = str_val

        # Collect bindings (value1, value2, scope, dest_zone, filter, optional)
        self._collect_bindings(data)

        # Handle Link Widget
        self.link_widget.get_data(data)

        # Special overrides
        if ui_type == "PLAY_FROM_ZONE" and self.no_cost_check.isChecked():
            data['value1'] = 999

        if ui_type == "SELECT_OPTION":
             data['value2'] = 1 if self.allow_duplicates_check.isChecked() else 0

    def _get_display_text(self, data):
        return f"{tr('Action')}: {tr(data.get('type', 'UNKNOWN'))}"

    def block_signals_all(self, block):
        super().block_signals_all(block)
        # Block signals for manual connections not in bindings
        self.type_combo.blockSignals(block)
        self.str_val_edit.blockSignals(block)
        self.measure_mode_combo.blockSignals(block)
        self.ref_mode_combo.blockSignals(block)
        self.allow_duplicates_check.blockSignals(block)
        self.no_cost_check.blockSignals(block)
