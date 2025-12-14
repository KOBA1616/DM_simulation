from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget

class ActionEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

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
            "can_be_optional": raw.get("can_be_optional", False),
            "produces_output": raw.get("produces_output", False),
            "tooltip": raw.get("tooltip", ""),
            "allowed_filter_fields": raw.get("allowed_filter_fields", None)
        }

    def setup_ui(self):
        layout = QFormLayout(self)

        self.type_combo = QComboBox()
        # "MEASURE_COUNT" is a UI-only type that maps to COUNT_CARDS or GET_GAME_STAT
        types = [
            "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP",
            "COST_REFERENCE", "COST_REDUCTION", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
            "REVOLUTION_CHANGE", "MEASURE_COUNT", "APPLY_MODIFIER", "REVEAL_CARDS",
            "REGISTER_DELAYED_EFFECT", "RESET_INSTANCE", "SEND_TO_DECK_BOTTOM",
            "FRIEND_BURST", "GRANT_KEYWORD", "MOVE_CARD"
        ]
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Action Type"), self.type_combo)

        self.scope_combo = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"]
        self.populate_combo(self.scope_combo, scopes, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Scope"), self.scope_combo)

        self.str_val_label = QLabel(tr("String Value"))
        self.str_val_edit = QLineEdit()

        # Measure Mode Combo (Unified)
        self.measure_mode_combo = QComboBox()
        self.measure_mode_combo.addItem(tr("CARDS_MATCHING_FILTER"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
        self.populate_combo(self.measure_mode_combo, stats, data_func=lambda x: x, display_func=tr, clear=False)

        self.ref_mode_combo = QComboBox()
        ref_modes = ["SYM_CREATURE", "SYM_SPELL", "G_ZERO", "HYPER_ENERGY", "NONE"]
        self.populate_combo(self.ref_mode_combo, ref_modes, data_func=lambda x: x, display_func=tr)

        layout.addRow(self.str_val_label, self.str_val_edit)

        self.mode_label = QLabel(tr("Mode"))
        layout.addRow(self.mode_label, self.measure_mode_combo)

        self.ref_mode_label = QLabel(tr("Ref Mode"))
        layout.addRow(self.ref_mode_label, self.ref_mode_combo)

        # Destination Zone Combo (For MOVE_CARD)
        self.dest_zone_combo = QComboBox()
        zones = ["HAND", "BATTLE_ZONE", "GRAVEYARD", "MANA_ZONE", "SHIELD_ZONE", "DECK_BOTTOM", "DECK_TOP"]
        self.populate_combo(self.dest_zone_combo, zones, data_func=lambda x: x, display_func=tr)
        self.dest_zone_label = QLabel(tr("Destination Zone"))
        layout.addRow(self.dest_zone_label, self.dest_zone_combo)

        # Filter Widget
        self.filter_group = QGroupBox(tr("Filter"))
        self.filter_widget = FilterEditorWidget()
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)

        layout.addRow(self.filter_group)

        self.val1_label = QLabel(tr("Value 1"))
        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(-9999, 9999)
        layout.addRow(self.val1_label, self.val1_spin)

        self.val2_label = QLabel(tr("Value 2"))
        self.val2_spin = QSpinBox()
        self.val2_spin.setRange(-9999, 9999)
        layout.addRow(self.val2_label, self.val2_spin)

        self.arbitrary_check = QCheckBox(tr("Arbitrary Amount (Up to N)"))
        layout.addRow(self.arbitrary_check)

        # Play From Zone Flags
        self.pay_cost_check = QCheckBox(tr("Pay Cost"))
        self.as_summon_check = QCheckBox(tr("Treat as Summon"))
        layout.addRow(self.pay_cost_check)
        layout.addRow(self.as_summon_check)

        # Variable Link Widget
        self.link_widget = VariableLinkWidget()
        self.link_widget.linkChanged.connect(self.update_data)
        self.link_widget.smartLinkStateChanged.connect(self.on_smart_link_changed)
        layout.addRow(self.link_widget)

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.scope_combo.currentIndexChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.str_val_edit.textChanged.connect(self.update_data)
        self.measure_mode_combo.currentIndexChanged.connect(self.on_measure_mode_changed)
        self.ref_mode_combo.currentIndexChanged.connect(self.update_data)
        self.dest_zone_combo.currentIndexChanged.connect(self.update_data)
        self.arbitrary_check.stateChanged.connect(self.update_data)
        self.pay_cost_check.stateChanged.connect(self.update_data)
        self.as_summon_check.stateChanged.connect(self.update_data)

        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        action_type = self.type_combo.currentData()
        self.update_ui_state(action_type)

        if self.current_item and not self._is_populating:
            config = self._get_ui_config(action_type)
            produces = config.get("produces_output", False)

            # Use unified name for linking logic
            if action_type == "MEASURE_COUNT": produces = True

            # Delegate to link widget
            self.link_widget.ensure_output_key(action_type, produces)

        self.update_data()

    def on_measure_mode_changed(self):
        self.update_ui_state(self.type_combo.currentData())
        self.update_data()

    def on_smart_link_changed(self, is_checked):
        action_type = self.type_combo.currentData()
        config = self._get_ui_config(action_type)

        self.val1_label.setVisible(config["val1_visible"] and not is_checked)
        self.val1_spin.setVisible(config["val1_visible"] and not is_checked)

        self.update_data()

    def update_ui_state(self, action_type):
        if not action_type: return

        config = self._get_ui_config(action_type)

        self.val1_label.setText(tr(config["val1_label"]))
        self.val2_label.setText(tr(config["val2_label"]))
        self.str_val_label.setText(tr(config["str_label"]))

        can_link_input = config["val1_visible"]
        self.link_widget.set_smart_link_enabled(can_link_input)

        self.arbitrary_check.setVisible(config.get("can_be_optional", False))

        # Play From Zone Checkboxes
        is_play_zone = (action_type == "PLAY_FROM_ZONE")
        self.pay_cost_check.setVisible(is_play_zone)
        self.as_summon_check.setVisible(is_play_zone)

        is_smart_linked = self.link_widget.is_smart_link_active() and can_link_input

        self.val1_label.setVisible(config["val1_visible"] and not is_smart_linked)
        self.val1_spin.setVisible(config["val1_visible"] and not is_smart_linked)
        self.val2_label.setVisible(config["val2_visible"])
        self.val2_spin.setVisible(config["val2_visible"])

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

        self.set_combo_by_data(self.type_combo, ui_type)

        # Map Values to Combos
        if ui_type == "MEASURE_COUNT":
             # If raw is COUNT_CARDS, str_val is empty -> CARDS_MATCHING_FILTER
             # If raw is GET_GAME_STAT, str_val is stat name
             val = "CARDS_MATCHING_FILTER" if raw_type == "COUNT_CARDS" else str_val
             self.set_combo_by_data(self.measure_mode_combo, val)
        elif ui_type == "COST_REFERENCE":
             self.set_combo_by_data(self.ref_mode_combo, str_val)

        self.set_combo_by_data(self.dest_zone_combo, data.get('destination_zone', 'HAND'))

        self.update_ui_state(ui_type)

        self.set_combo_by_data(self.scope_combo, data.get('scope', 'NONE'))

        filt = data.get('filter', {})
        self.filter_widget.set_data(filt)

        self.val1_spin.setValue(data.get('value1', 0))
        self.val2_spin.setValue(data.get('value2', 0))

        if ui_type != "MEASURE_COUNT" and ui_type != "COST_REFERENCE":
             self.str_val_edit.setText(str_val)

        self.arbitrary_check.setChecked(data.get('optional', False))
        self.pay_cost_check.setChecked(data.get('pay_cost', False))
        self.as_summon_check.setChecked(data.get('as_summon', False))

        self.link_widget.set_data(data)

    def _save_data(self, data):
        action_type = self.type_combo.currentData()

        # Map UI Type -> Internal Type
        if action_type == "MEASURE_COUNT":
            selected_mode = self.measure_mode_combo.currentData()
            if selected_mode == "CARDS_MATCHING_FILTER":
                data['type'] = "COUNT_CARDS"
                data['str_val'] = ""
            else:
                data['type'] = "GET_GAME_STAT"
                data['str_val'] = selected_mode

        elif action_type == "COST_REDUCTION":
            data['type'] = "APPLY_MODIFIER"
            data['str_val'] = "COST"

        elif action_type == "COST_REFERENCE":
            data['type'] = "COST_REFERENCE"
            data['str_val'] = self.ref_mode_combo.currentData()

        else:
            data['type'] = action_type
            data['str_val'] = self.str_val_edit.text()

        data['destination_zone'] = self.dest_zone_combo.currentData()
        data['scope'] = self.scope_combo.currentData()
        data['filter'] = self.filter_widget.get_data()
        data['value1'] = self.val1_spin.value()
        data['value2'] = self.val2_spin.value()
        data['optional'] = self.arbitrary_check.isChecked()
        data['pay_cost'] = self.pay_cost_check.isChecked()
        data['as_summon'] = self.as_summon_check.isChecked()

        self.link_widget.get_data(data)

        # Auto output key generation
        out_key = data.get('output_value_key')
        if not out_key:
             # Need to use the ACTUAL saved type here, not UI type
             config = ACTION_UI_CONFIG.get(data['type'], {})
             if config.get("produces_output", False):
                  row = self.current_item.row()
                  out_key = f"var_{data['type']}_{row}"
                  data['output_value_key'] = out_key

    def _get_display_text(self, data):
        display_type = tr(data['type'])
        if data['type'] == "GET_GAME_STAT":
             display_type = f"{tr('GET_GAME_STAT')} ({tr(data['str_val'])})"
        elif data['type'] == "APPLY_MODIFIER" and data['str_val'] == "COST":
             display_type = tr("COST_REDUCTION")
        elif data['type'] == "COST_REFERENCE":
             display_type = f"{tr('COST_REFERENCE')} ({tr(data['str_val'])})"

        return f"{tr('Action')}: {display_type}"

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.scope_combo.blockSignals(block)
        self.val1_spin.blockSignals(block)
        self.val2_spin.blockSignals(block)
        self.str_val_edit.blockSignals(block)
        self.measure_mode_combo.blockSignals(block)
        self.filter_widget.blockSignals(block)
        self.link_widget.blockSignals(block)
        self.arbitrary_check.blockSignals(block)
        self.ref_mode_combo.blockSignals(block)
        self.dest_zone_combo.blockSignals(block)
        self.pay_cost_check.blockSignals(block)
        self.as_summon_check.blockSignals(block)
