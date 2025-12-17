from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
# Variable linking might be needed later, for now sticking to basic CommandDef fields

class CommandEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def _get_ui_config(self, cmd_type):
        raw = COMMAND_UI_CONFIG.get(cmd_type, {})
        visible = raw.get("visible", [])
        vis = lambda key: key in visible

        return {
            "target_group_visible": vis("target_group"),
            "target_filter_visible": vis("target_filter"),
            "amount_visible": vis("amount"),
            "str_param_visible": vis("str_param"),
            "optional_visible": vis("optional"),
            "from_zone_visible": vis("from_zone"),
            "to_zone_visible": vis("to_zone"),
            "mutation_kind_visible": vis("mutation_kind"),
            "tooltip": raw.get("tooltip", "")
        }

    def setup_ui(self):
        layout = QFormLayout(self)

        # Type
        self.type_combo = QComboBox()
        self.known_types = [
            "TRANSITION", "MUTATE", "FLOW", "QUERY",
            "DRAW_CARD", "DISCARD", "DESTROY", "MANA_CHARGE",
            "TAP", "UNTAP", "POWER_MOD", "ADD_KEYWORD",
            "RETURN_TO_HAND", "BREAK_SHIELD", "SEARCH_DECK", "SHIELD_TRIGGER",
            "NONE"
        ]
        self.populate_combo(self.type_combo, self.known_types, data_func=lambda x: x, display_func=tr)
        layout.addRow(tr("Command Type"), self.type_combo)

        # Target Group
        self.target_group_combo = QComboBox()
        scopes = ["NONE", "SELF", "PLAYER_SELF", "PLAYER_OPPONENT", "ALL_PLAYERS", "TARGET_SELECT", "RANDOM", "ALL_FILTERED"]
        self.populate_combo(self.target_group_combo, scopes, data_func=lambda x: x, display_func=tr)
        self.target_group_label = QLabel(tr("Target Group"))
        layout.addRow(self.target_group_label, self.target_group_combo)

        # Zones
        self.from_zone_combo = QComboBox()
        self.to_zone_combo = QComboBox()
        zones = ["NONE", "HAND", "BATTLE_ZONE", "GRAVEYARD", "MANA_ZONE", "SHIELD_ZONE", "DECK", "DECK_BOTTOM", "DECK_TOP"]
        self.populate_combo(self.from_zone_combo, zones, data_func=lambda x: x, display_func=tr)
        self.populate_combo(self.to_zone_combo, zones, data_func=lambda x: x, display_func=tr)

        self.from_zone_label = QLabel(tr("From Zone"))
        self.to_zone_label = QLabel(tr("To Zone"))
        layout.addRow(self.from_zone_label, self.from_zone_combo)
        layout.addRow(self.to_zone_label, self.to_zone_combo)

        # Mutation Kind / Str Param
        self.mutation_kind_edit = QLineEdit()
        self.mutation_kind_label = QLabel(tr("Mutation Kind"))
        layout.addRow(self.mutation_kind_label, self.mutation_kind_edit)

        self.str_param_edit = QLineEdit()
        self.str_param_label = QLabel(tr("String Param"))
        layout.addRow(self.str_param_label, self.str_param_edit)

        # Amount
        self.amount_spin = QSpinBox()
        self.amount_spin.setRange(-9999, 9999)
        self.amount_label = QLabel(tr("Amount"))
        layout.addRow(self.amount_label, self.amount_spin)

        # Filter
        self.filter_group = QGroupBox(tr("Target Filter"))
        self.filter_widget = FilterEditorWidget()
        fg_layout = QVBoxLayout(self.filter_group)
        fg_layout.addWidget(self.filter_widget)
        self.filter_widget.filterChanged.connect(self.update_data)
        layout.addRow(self.filter_group)

        # Optional
        self.optional_check = QCheckBox(tr("Optional"))
        layout.addRow(self.optional_check)

        # Signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.target_group_combo.currentIndexChanged.connect(self.update_data)
        self.from_zone_combo.currentIndexChanged.connect(self.update_data)
        self.to_zone_combo.currentIndexChanged.connect(self.update_data)
        self.mutation_kind_edit.textChanged.connect(self.update_data)
        self.str_param_edit.textChanged.connect(self.update_data)
        self.amount_spin.valueChanged.connect(self.update_data)
        self.optional_check.stateChanged.connect(self.update_data)

        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        cmd_type = self.type_combo.currentData()
        self.update_ui_state(cmd_type)
        self.update_data()

    def update_ui_state(self, cmd_type):
        if not cmd_type: return
        config = self._get_ui_config(cmd_type)

        self.target_group_label.setVisible(config["target_group_visible"])
        self.target_group_combo.setVisible(config["target_group_visible"])

        self.filter_group.setVisible(config["target_filter_visible"])

        self.amount_label.setVisible(config["amount_visible"])
        self.amount_spin.setVisible(config["amount_visible"])

        self.str_param_label.setVisible(config["str_param_visible"])
        self.str_param_edit.setVisible(config["str_param_visible"])

        self.optional_check.setVisible(config["optional_visible"])

        self.from_zone_label.setVisible(config["from_zone_visible"])
        self.from_zone_combo.setVisible(config["from_zone_visible"])

        self.to_zone_label.setVisible(config["to_zone_visible"])
        self.to_zone_combo.setVisible(config["to_zone_visible"])

        self.mutation_kind_label.setVisible(config["mutation_kind_visible"])
        self.mutation_kind_edit.setVisible(config["mutation_kind_visible"])

        self.type_combo.setToolTip(tr(config["tooltip"]))

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.set_combo_by_data(self.type_combo, data.get('type', 'NONE'))
        self.set_combo_by_data(self.target_group_combo, data.get('target_group', 'NONE'))
        self.set_combo_by_data(self.from_zone_combo, data.get('from_zone', 'NONE'))
        self.set_combo_by_data(self.to_zone_combo, data.get('to_zone', 'NONE'))

        self.mutation_kind_edit.setText(data.get('mutation_kind', ''))
        self.str_param_edit.setText(data.get('str_param', ''))
        self.amount_spin.setValue(data.get('amount', 0))
        self.optional_check.setChecked(data.get('optional', False))

        self.filter_widget.set_data(data.get('target_filter', {}))

        self.update_ui_state(data.get('type', 'NONE'))

    def _save_data(self, data):
        data['type'] = self.type_combo.currentData()
        data['target_group'] = self.target_group_combo.currentData()
        data['from_zone'] = self.from_zone_combo.currentData()
        data['to_zone'] = self.to_zone_combo.currentData()
        data['mutation_kind'] = self.mutation_kind_edit.text()
        data['str_param'] = self.str_param_edit.text()
        data['amount'] = self.amount_spin.value()
        data['optional'] = self.optional_check.isChecked()
        data['target_filter'] = self.filter_widget.get_data()

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.target_group_combo.blockSignals(block)
        self.from_zone_combo.blockSignals(block)
        self.to_zone_combo.blockSignals(block)
        self.mutation_kind_edit.blockSignals(block)
        self.str_param_edit.blockSignals(block)
        self.amount_spin.blockSignals(block)
        self.optional_check.blockSignals(block)
        self.filter_widget.blockSignals(block)
