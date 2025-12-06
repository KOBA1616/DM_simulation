from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QCheckBox, QGroupBox, QGridLayout, QLabel
from PyQt6.QtCore import Qt
from gui.localization import tr
from gui.editor.forms.action_config import ACTION_UI_CONFIG

class ActionEditForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.type_combo = QComboBox()
        types = [
            "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP",
            "COST_REFERENCE", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
            "REVOLUTION_CHANGE", "COUNT_CARDS", "GET_GAME_STAT", "APPLY_MODIFIER", "REVEAL_CARDS",
            "REGISTER_DELAYED_EFFECT", "RESET_INSTANCE", "SEND_TO_DECK_BOTTOM"
        ]
        for t in types:
            self.type_combo.addItem(tr(t), t)
        layout.addRow(tr("Action Type"), self.type_combo)

        self.scope_combo = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"]
        for s in scopes:
            self.scope_combo.addItem(tr(s), s)
        layout.addRow(tr("Scope"), self.scope_combo)

        # Filter
        self.filter_group = QGroupBox(tr("Filter"))
        f_layout = QGridLayout(self.filter_group)

        # Add a help/usage label
        help_label = QLabel(tr("Select zones to target. Set Count > 0 to require a specific number of cards."))
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-style: italic;")
        f_layout.addWidget(help_label, 0, 0, 1, 2)

        self.zone_checks = {}
        zones = ["BATTLE_ZONE", "MANA_ZONE", "HAND", "GRAVEYARD", "SHIELD_ZONE", "DECK"]
        for i, z in enumerate(zones):
            cb = QCheckBox(tr(z)) # Display localized name
            cb.setToolTip(tr(f"Include {z} in target selection"))
            self.zone_checks[z] = cb
            f_layout.addWidget(cb, (i//2) + 1, i%2)
            cb.stateChanged.connect(self.update_data)

        self.filter_count = QSpinBox()
        self.filter_count.setToolTip(tr("Number of cards to select/count. 0 means all/any."))
        f_layout.addWidget(QLabel(tr("Count")), 4, 0)
        f_layout.addWidget(self.filter_count, 4, 1)
        self.filter_count.valueChanged.connect(self.update_data)

        layout.addRow(self.filter_group)

        # Labels for dynamic values, stored to update later
        self.val1_label = QLabel(tr("Value 1"))
        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(-9999, 9999) # Allow wider range
        layout.addRow(self.val1_label, self.val1_spin)

        self.val2_label = QLabel(tr("Value 2"))
        self.val2_spin = QSpinBox()
        self.val2_spin.setRange(-9999, 9999)
        layout.addRow(self.val2_label, self.val2_spin)

        self.str_val_label = QLabel(tr("String Value"))
        self.str_val_edit = QLineEdit()
        layout.addRow(self.str_val_label, self.str_val_edit)

        # Variable Linking
        self.input_key_combo = QComboBox()
        self.input_key_combo.setEditable(True)
        layout.addRow(tr("Input Key"), self.input_key_combo)

        self.output_key_edit = QLineEdit()
        layout.addRow(tr("Output Key"), self.output_key_edit)

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed) # New handler
        self.scope_combo.currentIndexChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.str_val_edit.textChanged.connect(self.update_data)
        self.input_key_combo.currentTextChanged.connect(self.update_data)
        self.output_key_edit.textChanged.connect(self.update_data)

        # Initialize UI state
        self.update_ui_state(self.type_combo.currentData())

    def on_type_changed(self):
        # Update UI state first
        self.update_ui_state(self.type_combo.currentData())
        # Then update data
        self.update_data()

    def update_ui_state(self, action_type):
        if not action_type: return

        config = ACTION_UI_CONFIG.get(action_type, ACTION_UI_CONFIG["NONE"])

        # Update Labels
        self.val1_label.setText(tr(config["val1_label"]))
        self.val2_label.setText(tr(config["val2_label"]))
        self.str_val_label.setText(tr(config["str_label"]))

        # Update Visibility
        self.val1_label.setVisible(config["val1_visible"])
        self.val1_spin.setVisible(config["val1_visible"])

        self.val2_label.setVisible(config["val2_visible"])
        self.val2_spin.setVisible(config["val2_visible"])

        self.str_val_label.setVisible(config["str_visible"])
        self.str_val_edit.setVisible(config["str_visible"])

        self.filter_group.setVisible(config["filter_visible"])

        # Update Tooltips
        self.type_combo.setToolTip(tr(config.get("tooltip", "")))

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)

        t_idx = self.type_combo.findData(data.get('type', 'NONE'))
        if t_idx >= 0:
            self.type_combo.setCurrentIndex(t_idx)
            # Explicitly update UI state when loading data
            self.update_ui_state(self.type_combo.itemData(t_idx))

        s_idx = self.scope_combo.findData(data.get('scope', 'NONE'))
        if s_idx >= 0: self.scope_combo.setCurrentIndex(s_idx)

        filt = data.get('filter', {})
        zones = filt.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)
        self.filter_count.setValue(filt.get('count', 0))

        self.val1_spin.setValue(data.get('value1', 0))
        self.val2_spin.setValue(data.get('value2', 0))
        self.str_val_edit.setText(data.get('str_val', ''))

        # Variable Linking Population
        self.populate_input_keys()
        self.input_key_combo.setCurrentText(data.get('input_value_key', ''))
        self.output_key_edit.setText(data.get('output_value_key', ''))

        self.block_signals(False)

    def populate_input_keys(self):
        self.input_key_combo.clear()
        if not self.current_item: return

        # Traverse siblings upwards
        parent = self.current_item.parent()
        if not parent: return

        row = self.current_item.row()
        for i in range(row):
            sibling = parent.child(i)
            sib_data = sibling.data(Qt.ItemDataRole.UserRole + 2)
            out_key = sib_data.get('output_value_key')
            if out_key:
                # Format: "{key} (from #{index} {type})"
                # Translate type for display
                type_disp = tr(sib_data.get('type'))
                label = f"{out_key} (from #{i} {type_disp})"
                self.input_key_combo.addItem(label, out_key) # Store actual key in UserData

    def update_data(self):
        if not self.current_item: return
        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)

        data['type'] = self.type_combo.currentData()
        data['scope'] = self.scope_combo.currentData()

        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        filt = {}
        if zones: filt['zones'] = zones
        count = self.filter_count.value()
        if count > 0: filt['count'] = count
        data['filter'] = filt

        data['value1'] = self.val1_spin.value()
        data['value2'] = self.val2_spin.value()
        data['str_val'] = self.str_val_edit.text()

        # Handle Input Key: if selected from combo, use data; if typed, use text
        idx = self.input_key_combo.currentIndex()
        if idx >= 0 and self.input_key_combo.currentText() == self.input_key_combo.itemText(idx):
             data['input_value_key'] = self.input_key_combo.itemData(idx)
        else:
             data['input_value_key'] = self.input_key_combo.currentText()

        data['output_value_key'] = self.output_key_edit.text()

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        self.current_item.setText(f"{tr('Action')}: {tr(data['type'])}")

    def block_signals(self, block):
        self.type_combo.blockSignals(block)
        self.scope_combo.blockSignals(block)
        self.val1_spin.blockSignals(block)
        self.val2_spin.blockSignals(block)
        self.str_val_edit.blockSignals(block)
        self.input_key_combo.blockSignals(block)
        self.output_key_edit.blockSignals(block)
        for cb in self.zone_checks.values():
            cb.blockSignals(block)
        self.filter_count.blockSignals(block)
