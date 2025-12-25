# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QStackedWidget
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.utils import normalize_command_zone_keys
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.consts import COMMAND_TYPES, ZONES_EXTENDED, GRANTABLE_KEYWORDS
from dm_toolkit.gui.editor.consts import STRUCT_CMD_GENERATE_BRANCHES


class CommandEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def _get_ui_config(self, cmd_type):
        """
        Map Command Types to UI Configs using the dedicated COMMAND_UI_CONFIG.
        """
        raw = COMMAND_UI_CONFIG.get(cmd_type, {})
        visible = raw.get("visible", [])
        vis = lambda key: key in visible

          # Map visibility flags
        return {
            "target_group_visible": vis("target_group"),
            "target_filter_visible": vis("target_filter"),
            "amount_visible": vis("amount"),
            "amount_label": raw.get("label_amount", "Amount"),
            "str_param_visible": vis("str_param"),
            "str_param_label": raw.get("label_str_param", "String Param"),
            "mutation_kind_visible": vis("mutation_kind"),
            "mutation_kind_label": raw.get("label_mutation_kind", "Mutation Kind"),
            "to_zone_visible": vis("to_zone"),
            "from_zone_visible": vis("from_zone"),
            "optional_visible": vis("optional"),
            "input_link_visible": vis("input_link"),
            "produces_output": raw.get("produces_output", False),
            "tooltip": raw.get("tooltip", ""),
            "allowed_filter_fields": raw.get("allowed_filter_fields", None),
        }

    def setup_ui(self):
        layout = QFormLayout(self)

          # Migration / Warning Label
        self.warning_label = QLabel(tr("Warning: Imperfect Conversion"))
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.warning_label.setVisible(False)
        layout.addRow(self.warning_label)

          # Command Type
        self.type_combo = QComboBox()
          # Filter out MUTATE and FLOW from user selection
        self.known_types = [t for t in COMMAND_TYPES if t not in ["MUTATE", "FLOW"]]
        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Command Type"), self.type_combo)

          # Target Group (formerly Scope)
        self.target_group_combo = QComboBox()
        scopes = [
            "PLAYER_SELF",
            "PLAYER_OPPONENT",
            "TARGET_SELECT",
            "ALL_PLAYERS",
            "RANDOM",
            "ALL_FILTERED",
            "NONE",
        ]
        self.populate_combo(self.target_group_combo, scopes, data_func=lambda x: x, display_func=tr)
        self.target_group_label = QLabel(tr("Target Group"))
        layout.addRow(self.target_group_label, self.target_group_combo)

          # Mutation Kind / Str Param (Merged conceptual slot)
          # Using a QStackedWidget to switch between Edit and Combo occupying the same layout slot
        self.mutation_kind_container = QStackedWidget()

        self.mutation_kind_edit = QLineEdit()
        self.mutation_kind_combo = QComboBox()
        self.populate_combo(self.mutation_kind_combo, GRANTABLE_KEYWORDS, data_func=lambda x: x, display_func=tr)

        self.mutation_kind_container.addWidget(self.mutation_kind_edit)  # Index 0
        self.mutation_kind_container.addWidget(self.mutation_kind_combo)  # Index 1

        self.mutation_kind_label = QLabel(tr("Mutation Kind"))
        layout.addRow(self.mutation_kind_label, self.mutation_kind_container)

        self.str_param_edit = QLineEdit()
        self.str_param_label = QLabel(tr("String Param"))
        layout.addRow(self.str_param_label, self.str_param_edit)

          # Query Mode (For QUERY type, similar to MEASURE_COUNT)
        self.query_mode_combo = QComboBox()
        self.query_mode_combo.addItem(tr("CARDS_MATCHING_FILTER"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
        self.populate_combo(self.query_mode_combo, stats, data_func=lambda x: x, display_func=tr, clear=False)
        self.query_mode_label = QLabel(tr("Query Mode"))
        layout.addRow(self.query_mode_label, self.query_mode_combo)

          # Zones
        self.from_zone_combo = QComboBox()
        self.to_zone_combo = QComboBox()
        zones = ZONES_EXTENDED
        self.populate_combo(self.from_zone_combo, zones, data_func=lambda x: x, display_func=tr)
        self.populate_combo(self.to_zone_combo, zones, data_func=lambda x: x, display_func=tr)

        self.from_zone_label = QLabel(tr("From Zone"))
        self.to_zone_label = QLabel(tr("To Zone"))
        layout.addRow(self.from_zone_label, self.from_zone_combo)
        layout.addRow(self.to_zone_label, self.to_zone_combo)

          # Filter
        self.filter_group = QGroupBox(tr("Target Filter"))
        self.filter_widget = FilterEditorWidget()
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)
        layout.addRow(self.filter_group)

          # Amount (formerly Value 1)
        self.amount_spin = QSpinBox()
        self.amount_spin.setRange(-9999, 9999)
        self.amount_label = QLabel(tr("Amount"))
        layout.addRow(self.amount_label, self.amount_spin)

          # Optional
        self.optional_check = QCheckBox(tr("Optional (Arbitrary Amount)"))
        layout.addRow(self.optional_check)

          # Branch Generation (For FLOW)
        self.gen_branch_btn = QPushButton(tr("Generate Branches"))
        self.gen_branch_btn.clicked.connect(self.request_generate_branches)
        layout.addRow(self.gen_branch_btn)

          # Variable Linking
        self.link_widget = VariableLinkWidget()
        self.link_widget.linkChanged.connect(self.update_data)
        self.link_widget.smartLinkStateChanged.connect(self.on_smart_link_changed)
        layout.addRow(self.link_widget)

          # Signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.target_group_combo.currentIndexChanged.connect(self.update_data)
        self.mutation_kind_edit.textChanged.connect(self.update_data)
        self.mutation_kind_combo.currentIndexChanged.connect(self.update_data)
        self.str_param_edit.textChanged.connect(self.update_data)
        self.amount_spin.valueChanged.connect(self.update_data)
        self.optional_check.stateChanged.connect(self.update_data)
        self.from_zone_combo.currentIndexChanged.connect(self.update_data)
        self.to_zone_combo.currentIndexChanged.connect(self.update_data)
        self.query_mode_combo.currentIndexChanged.connect(self.on_query_mode_changed)

        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        cmd_type = self.type_combo.currentData()
        self.update_ui_state(cmd_type)

        if self.current_item and not self._is_populating:
            config = self._get_ui_config(cmd_type)
            produces = config.get("produces_output", False)
            if cmd_type == "QUERY":
                    produces = True
            self.link_widget.ensure_output_key(cmd_type, produces)

        self.update_data()

    def on_query_mode_changed(self):
          # Update visibility if mode changes (e.g. hiding filter for Game Stats)
        self.update_ui_state(self.type_combo.currentData())
        self.update_data()

    def on_smart_link_changed(self, is_active):
        cmd_type = self.type_combo.currentData()
        config = self._get_ui_config(cmd_type)

        self.amount_label.setVisible(config["amount_visible"] and not is_active)
        self.amount_spin.setVisible(config["amount_visible"] and not is_active)

        if config["target_filter_visible"]:
            self.filter_widget.set_external_count_control(is_active)

        self.update_data()

    def update_ui_state(self, cmd_type):
        if not cmd_type:
            return
        config = self._get_ui_config(cmd_type)

        self.target_group_label.setVisible(config["target_group_visible"])
        self.target_group_combo.setVisible(config["target_group_visible"])

        self.amount_label.setText(tr(config["amount_label"]))
        self.mutation_kind_label.setText(tr(config["mutation_kind_label"]))
        self.str_param_label.setText(tr(config["str_param_label"]))

          # Input Linking
        can_link_input = config["amount_visible"] or config.get("input_link_visible", False)
        self.link_widget.set_smart_link_enabled(can_link_input)
        is_smart_linked = self.link_widget.is_smart_link_active() and can_link_input

        self.amount_label.setVisible(config["amount_visible"] and not is_smart_linked)
        self.amount_spin.setVisible(config["amount_visible"] and not is_smart_linked)

        self.str_param_label.setVisible(config["str_param_visible"])
        self.str_param_edit.setVisible(config["str_param_visible"])

          # Mutation Kind
        is_add_keyword = (cmd_type == "ADD_KEYWORD")
        self.mutation_kind_label.setVisible(config["mutation_kind_visible"])
        self.mutation_kind_container.setVisible(config["mutation_kind_visible"])

          # Switch stack
        if is_add_keyword:
            self.mutation_kind_container.setCurrentIndex(1)  # Combo
        else:
            self.mutation_kind_container.setCurrentIndex(0)  # Edit

          # Query Mode
        is_query = (cmd_type == "QUERY")
        self.query_mode_label.setVisible(is_query)
        self.query_mode_combo.setVisible(is_query)

          # Zones
        self.from_zone_label.setVisible(config["from_zone_visible"])
        self.from_zone_combo.setVisible(config["from_zone_visible"])
        self.to_zone_label.setVisible(config["to_zone_visible"])
        self.to_zone_combo.setVisible(config["to_zone_visible"])

        self.optional_check.setVisible(config["optional_visible"])

          # Branch Generation Button Visibility
        self.gen_branch_btn.setVisible(cmd_type == "FLOW")

          # Filter
        show_filter = config["target_filter_visible"]
        if is_query:
               # Hide filter if mode is NOT CARDS_MATCHING_FILTER
            current_mode = self.query_mode_combo.currentData()
            if current_mode != "CARDS_MATCHING_FILTER" and current_mode is not None:
                    show_filter = False

        self.filter_group.setVisible(show_filter)
        if show_filter:
            self.filter_widget.set_allowed_fields(config.get("allowed_filter_fields", None))
               # Sync external count control
            self.filter_widget.set_external_count_control(is_smart_linked)

        self.type_combo.setToolTip(tr(config["tooltip"]))

    def _populate_ui(self, item):
        self.link_widget.set_current_item(item)
        data = item.data(Qt.ItemDataRole.UserRole + 2)

          # Normalize keys to ensure 'from_zone'/'to_zone' are used
        normalize_command_zone_keys(data)

          # Check for Legacy Warning
        legacy_warning = data.get('legacy_warning', False)
        if legacy_warning:
            orig = data.get('legacy_original_type', 'Unknown')
            self.warning_label.setText(tr(f"Warning: Imperfect Conversion from {orig}"))
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

        raw_type = data.get('type', 'NONE')

          # Populate UI Type
        self.set_combo_by_data(self.type_combo, raw_type)
        self.set_combo_by_data(self.target_group_combo, data.get('target_group', 'NONE'))
        self.set_combo_by_data(self.from_zone_combo, data.get('from_zone', 'NONE'))
        self.set_combo_by_data(self.to_zone_combo, data.get('to_zone', 'NONE'))

        self.amount_spin.setValue(data.get('amount', 0))

        mutation_kind = data.get('mutation_kind', '')
        self.mutation_kind_edit.setText(mutation_kind)

          # Check if keyword is valid for combo
        if raw_type == "ADD_KEYWORD" and mutation_kind not in GRANTABLE_KEYWORDS and mutation_kind:
               # Add temporarily to prevent data loss or display issue
            self.mutation_kind_combo.addItem(f"{mutation_kind} (Unknown)", mutation_kind)

        self.set_combo_by_data(self.mutation_kind_combo, mutation_kind)

        self.str_param_edit.setText(data.get('str_param', ''))
        self.optional_check.setChecked(data.get('optional', False))

        self.filter_widget.set_data(data.get('target_filter', {}))

          # Query Mode Mapping
        if raw_type == "QUERY":
               # We assume if target_filter is empty/default and str_param is set, it might be a stat query
               # Or we rely on a convention. For now, we assume explicit mapping isn't saved in 'type' but in 'str_param' or inferred.
               # Actually, if we use the same logic as Action, we might store "MANA_CIVILIZATION_COUNT" in str_param/mutation_kind?
               # Let's use 'str_param' for query mode if not default.
            mode = data.get('str_param', "CARDS_MATCHING_FILTER")
            if mode == "":
                    mode = "CARDS_MATCHING_FILTER"
            self.set_combo_by_data(self.query_mode_combo, mode)

        self.link_widget.set_data(data)
        self.update_ui_state(raw_type)

    def request_generate_branches(self):
        self.structure_update_requested.emit(STRUCT_CMD_GENERATE_BRANCHES, {})

    def _save_data(self, data):
      cmd_type = self.type_combo.currentData()

      data['type'] = cmd_type
      data['target_group'] = self.target_group_combo.currentData()
      data['target_filter'] = self.filter_widget.get_data()
      data['amount'] = self.amount_spin.value()
      data['optional'] = self.optional_check.isChecked()
      data['from_zone'] = self.from_zone_combo.currentData()
      data['to_zone'] = self.to_zone_combo.currentData()

      # Mutation Kind: Use Combo if ADD_KEYWORD, else Edit
      if cmd_type == "ADD_KEYWORD":
        data['mutation_kind'] = self.mutation_kind_combo.currentData()
      else:
        data['mutation_kind'] = self.mutation_kind_edit.text()

      data['str_param'] = self.str_param_edit.text()

      # Query Logic
      if cmd_type == "QUERY":
        mode = self.query_mode_combo.currentData()
        if mode != "CARDS_MATCHING_FILTER":
          data['str_param'] = mode
        else:
          data['str_param'] = ""

      self.link_widget.get_data(data)

      # Auto output key
      out_key = data.get('output_value_key')
      if not out_key:
        config = self._get_ui_config(cmd_type)
        if config.get("produces_output", False):
          # Guard current_item possibly being None
          curr = getattr(self, 'current_item', None)
          if curr is not None:
            row = curr.row()
          else:
            row = 0
          out_key = f"var_{cmd_type}_{row}"
          data['output_value_key'] = out_key

      # Preserve legacy warning flags if present (read-only)
      # Assuming we don't clear keys that are not in UI
      # BaseEditForm._save_data might be needed if we want to be safe, but here we modify 'data' dict in place
      # The 'data' dict passed to _save_data is usually the one attached to the item, or a fresh one.
      # Standard implementation of BaseEditForm.save_data copies data from UI to the dict.
      # So we don't need to explicitly save legacy flags unless we overwrite the whole dict.
    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.target_group_combo.blockSignals(block)
        self.amount_spin.blockSignals(block)
        self.mutation_kind_edit.blockSignals(block)
        self.mutation_kind_combo.blockSignals(block)
        self.str_param_edit.blockSignals(block)
        self.optional_check.blockSignals(block)
        self.from_zone_combo.blockSignals(block)
        self.to_zone_combo.blockSignals(block)
        self.query_mode_combo.blockSignals(block)
        self.filter_widget.blockSignals(block)
        self.link_widget.blockSignals(block)
