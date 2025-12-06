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
        # Removed GET_GAME_STAT from user facing list, integrated into COUNT_CARDS
        types = [
            "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP",
            "COST_REFERENCE", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
            "REVOLUTION_CHANGE", "COUNT_CARDS", "APPLY_MODIFIER", "REVEAL_CARDS",
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

        # Add a help/usage label (Localized text from localization.py)
        help_label = QLabel(tr("Filter Help"))
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

        # Filter Count Mode Selection (Option 3 Implementation)
        self.filter_mode_label = QLabel(tr("Selection Mode"))
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem(tr("All/Any"), 0)
        self.filter_mode_combo.addItem(tr("Fixed Number"), 1)

        self.filter_count_spin = QSpinBox()
        self.filter_count_spin.setRange(1, 99)
        self.filter_count_spin.setToolTip(tr("Number of cards to select/count."))

        f_layout.addWidget(self.filter_mode_label, 4, 0)
        f_layout.addWidget(self.filter_mode_combo, 4, 1)
        f_layout.addWidget(self.filter_count_spin, 5, 1)

        self.filter_mode_combo.currentIndexChanged.connect(self.on_filter_mode_changed)
        self.filter_count_spin.valueChanged.connect(self.update_data)

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
        self.str_val_combo = QComboBox() # Added ComboBox for select stats

        # Populate Stats Options
        self.str_val_combo.addItem(tr("Cards matching filter"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT"]
        for s in stats:
            self.str_val_combo.addItem(tr(s), s)

        layout.addRow(self.str_val_label, self.str_val_edit)
        layout.addRow("", self.str_val_combo) # Add to layout, label managed by update_ui_state

        # Variable Linking
        self.smart_link_check = QCheckBox(tr("Use result from previous measurement"))
        layout.addRow(self.smart_link_check)

        # Arbitrary / Optional Effect Check (Up to N)
        self.optional_check = QCheckBox(tr("Arbitrary Amount (Up to N)"))
        self.optional_check.setToolTip(tr("If checked, the effect applies to 'up to' the specified count (0 to N)."))
        layout.addRow(self.optional_check)
        self.optional_check.setVisible(False) # Hidden by default
        self.optional_check.stateChanged.connect(self.update_data)

        self.input_key_combo = QComboBox()
        self.input_key_combo.setEditable(True)
        layout.addRow(tr("Input Key"), self.input_key_combo)

        # Output Key - Hidden from user to simplify UI
        self.output_key_label = QLabel(tr("Output Key"))
        self.output_key_edit = QLineEdit()
        layout.addRow(self.output_key_label, self.output_key_edit)
        self.output_key_label.setVisible(False)
        self.output_key_edit.setVisible(False)

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.scope_combo.currentIndexChanged.connect(self.update_data)
        self.val1_spin.valueChanged.connect(self.update_data)
        self.val2_spin.valueChanged.connect(self.update_data)
        self.str_val_edit.textChanged.connect(self.update_data)
        self.str_val_combo.currentIndexChanged.connect(self.on_stat_mode_changed) # Updated handler
        self.smart_link_check.stateChanged.connect(self.on_smart_link_changed)
        self.input_key_combo.currentTextChanged.connect(self.update_data)
        self.output_key_edit.textChanged.connect(self.update_data)

        # Initialize UI state
        self.update_ui_state(self.type_combo.currentData())
        self.on_filter_mode_changed() # Init filter UI state

    def on_type_changed(self):
        # Update UI state first
        action_type = self.type_combo.currentData()
        self.update_ui_state(action_type)

        # Automated Variable Linking: Auto-generate output key if needed
        if self.current_item:
            config = ACTION_UI_CONFIG.get(action_type, {})
            produces = config.get("produces_output", False)
            if action_type == "GET_GAME_STAT": produces = True

            if produces:
                current_out = self.output_key_edit.text()
                if not current_out:
                    # Generate unique key: var_{TYPE}_{ROW}
                    row = self.current_item.row()
                    new_key = f"var_{action_type}_{row}"
                    self.output_key_edit.setText(new_key)
            else:
                pass

        # Then update data
        self.update_data()

    def on_stat_mode_changed(self):
        # Update UI visibility based on the selected mode in the unified combo
        self.update_ui_state(self.type_combo.currentData())
        self.update_data()

    def on_smart_link_changed(self):
        # When "Use result from previous" is toggled
        is_checked = self.smart_link_check.isChecked()

        # 1. Update Visibility of Value 1
        action_type = self.type_combo.currentData()
        config = ACTION_UI_CONFIG.get(action_type, ACTION_UI_CONFIG["NONE"])

        self.val1_label.setVisible(config["val1_visible"] and not is_checked)
        self.val1_spin.setVisible(config["val1_visible"] and not is_checked)

        # 2. Logic to auto-set input key if checked
        if is_checked:
            self.populate_input_keys()
            count = self.input_key_combo.count()
            if count > 0:
                last_idx = count - 1
                key = self.input_key_combo.itemData(last_idx)
                self.input_key_combo.setCurrentIndex(last_idx)
        else:
            self.input_key_combo.setCurrentText("")

        # Trigger update data to save the change
        self.update_data()

    def on_filter_mode_changed(self):
        # 0 = All/Any, 1 = Fixed Number
        mode = self.filter_mode_combo.currentData()
        is_fixed = (mode == 1)
        self.filter_count_spin.setVisible(is_fixed)
        self.update_data()

    def update_ui_state(self, action_type):
        if not action_type: return

        config = ACTION_UI_CONFIG.get(action_type, ACTION_UI_CONFIG["NONE"])

        # Unified Logic: Treat GET_GAME_STAT as COUNT_CARDS for UI config purposes
        if action_type == "GET_GAME_STAT":
            config = ACTION_UI_CONFIG.get("COUNT_CARDS", config)

        # Update Labels
        self.val1_label.setText(tr(config["val1_label"]))
        self.val2_label.setText(tr(config["val2_label"]))
        self.str_val_label.setText(tr(config["str_label"]))

        # Smart Link Visibility
        can_link_input = config["val1_visible"]
        self.smart_link_check.setVisible(can_link_input)

        is_smart_linked = self.smart_link_check.isChecked() and can_link_input

        # Optional Check Visibility
        self.optional_check.setVisible(config.get("can_be_optional", False))

        # Update Visibility
        self.val1_label.setVisible(config["val1_visible"] and not is_smart_linked)
        self.val1_spin.setVisible(config["val1_visible"] and not is_smart_linked)

        self.val2_label.setVisible(config["val2_visible"])
        self.val2_spin.setVisible(config["val2_visible"])

        # Special handling for unified COUNT_CARDS / GET_GAME_STAT
        if action_type == "GET_GAME_STAT" or action_type == "COUNT_CARDS":
            self.str_val_label.setVisible(True)
            self.str_val_edit.setVisible(False)
            self.str_val_combo.setVisible(True)

            # Determine filter visibility based on Combo Selection
            current_mode = self.str_val_combo.currentData()
            if current_mode == "CARDS_MATCHING_FILTER" or current_mode is None:
                self.filter_group.setVisible(True)
            else:
                self.filter_group.setVisible(False)
        else:
            self.str_val_label.setVisible(config["str_visible"])
            self.str_val_edit.setVisible(config["str_visible"])
            self.str_val_combo.setVisible(False)
            self.filter_group.setVisible(config["filter_visible"])

        # Update Tooltips
        self.type_combo.setToolTip(tr(config.get("tooltip", "")))

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)

        input_key = data.get('input_value_key', '')
        if input_key:
            self.smart_link_check.setChecked(True)
        else:
            self.smart_link_check.setChecked(False)

        # Set Action Type
        raw_type = data.get('type', 'NONE')

        # UI Mapping: GET_GAME_STAT -> COUNT_CARDS
        ui_type = raw_type
        if raw_type == "GET_GAME_STAT":
            ui_type = "COUNT_CARDS"

        t_idx = self.type_combo.findData(ui_type)
        if t_idx >= 0:
            self.type_combo.setCurrentIndex(t_idx)

        # Optional Flag (Stored in generic 'optional' field)
        self.optional_check.setChecked(data.get('optional', False))

        # Handle Mode Combo for Unified Type
        str_val = data.get('str_val', '')
        if raw_type == "COUNT_CARDS":
             c_idx = self.str_val_combo.findData("CARDS_MATCHING_FILTER")
             if c_idx >= 0: self.str_val_combo.setCurrentIndex(c_idx)
        elif raw_type == "GET_GAME_STAT":
             c_idx = self.str_val_combo.findData(str_val)
             if c_idx >= 0: self.str_val_combo.setCurrentIndex(c_idx)

        # Now update UI state (which relies on combo values being set)
        self.update_ui_state(ui_type)

        s_idx = self.scope_combo.findData(data.get('scope', 'NONE'))
        if s_idx >= 0: self.scope_combo.setCurrentIndex(s_idx)

        filt = data.get('filter', {})
        zones = filt.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        # Filter Count Mode Logic
        count_val = filt.get('count', 0)
        if count_val > 0:
             self.filter_mode_combo.setCurrentIndex(1) # Fixed Number
             self.filter_count_spin.setValue(count_val)
             self.filter_count_spin.setVisible(True)
        else:
             self.filter_mode_combo.setCurrentIndex(0) # All/Any
             self.filter_count_spin.setValue(1) # Default if switched
             self.filter_count_spin.setVisible(False)

        self.val1_spin.setValue(data.get('value1', 0))
        self.val2_spin.setValue(data.get('value2', 0))

        self.str_val_edit.setText(str_val)

        # Variable Linking Population
        self.populate_input_keys()
        # Intelligent setting of input key combo:
        # 1. Try to find the item data that matches the key
        # 2. If found, set index (shows friendly text)
        # 3. If not found, set text directly (legacy/fallback)
        found_idx = -1
        for i in range(self.input_key_combo.count()):
             if self.input_key_combo.itemData(i) == input_key:
                  found_idx = i
                  break

        if found_idx >= 0:
             self.input_key_combo.setCurrentIndex(found_idx)
        else:
             self.input_key_combo.setCurrentText(input_key)

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
                # Format: "Step {index}: {Type} ({key})"
                type_disp = tr(sib_data.get('type'))
                label = f"Step {i}: {type_disp} ({out_key})"
                self.input_key_combo.addItem(label, out_key)

    def update_data(self):
        if not self.current_item: return

        # Verify that the current item in the form matches the one we intend to update
        # This prevents accidental overwrites if signals fire during transition
        # (Though blockSignals usually prevents this, this is a safety check)
        # However, checking object identity is tricky if the model wraps it.
        # But self.current_item is the QStandardItem.

        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
        if data is None: data = {} # Safety

        action_type = self.type_combo.currentData()

        # Handle Unified Type Logic: Map UI to Internal Type
        if action_type == "COUNT_CARDS":
            selected_mode = self.str_val_combo.currentData()

            if selected_mode == "CARDS_MATCHING_FILTER":
                data['type'] = "COUNT_CARDS"
                data['str_val'] = "" # Clear str_val for standard count
            else:
                # It's a Stat
                data['type'] = "GET_GAME_STAT"
                data['str_val'] = selected_mode
        else:
            data['type'] = action_type
            # For non-stat types, read string edit
            # But wait, did we overwrite str_val for COUNT_CARDS above? Yes.
            # If we are here, it's NOT COUNT_CARDS (UI type).
            data['str_val'] = self.str_val_edit.text()

        data['scope'] = self.scope_combo.currentData()

        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        filt = {}
        if zones: filt['zones'] = zones

        # Filter Count Logic based on Mode
        mode = self.filter_mode_combo.currentData()
        if mode == 1: # Fixed Number
             count = self.filter_count_spin.value()
             if count > 0: filt['count'] = count
        else:
             # All/Any -> count = 0 (omitted or explicit 0)
             pass

        data['filter'] = filt

        data['value1'] = self.val1_spin.value()
        data['value2'] = self.val2_spin.value()

        # Store Optional Flag
        data['optional'] = self.optional_check.isChecked()

        # Input Key
        idx = self.input_key_combo.currentIndex()
        if idx >= 0 and self.input_key_combo.currentText() == self.input_key_combo.itemText(idx):
             data['input_value_key'] = self.input_key_combo.itemData(idx)
        else:
             data['input_value_key'] = self.input_key_combo.currentText()

        # Output Key
        # If empty and we should produce output, generate it now (safety net)
        out_key = self.output_key_edit.text()
        if not out_key:
             config = ACTION_UI_CONFIG.get(data['type'], {}) # Use actual type
             if config.get("produces_output", False):
                  row = self.current_item.row()
                  out_key = f"var_{data['type']}_{row}"
                  self.output_key_edit.setText(out_key) # Update UI too

        data['output_value_key'] = out_key

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)

        # Update display text in tree
        display_type = tr(data['type'])
        if data['type'] == "GET_GAME_STAT":
             display_type = f"{tr('GET_GAME_STAT')} ({tr(data['str_val'])})"

        self.current_item.setText(f"{tr('Action')}: {display_type}")

    def block_signals(self, block):
        self.type_combo.blockSignals(block)
        self.scope_combo.blockSignals(block)
        self.val1_spin.blockSignals(block)
        self.val2_spin.blockSignals(block)
        self.str_val_edit.blockSignals(block)
        self.str_val_combo.blockSignals(block)
        self.input_key_combo.blockSignals(block)
        self.output_key_edit.blockSignals(block)
        for cb in self.zone_checks.values():
            cb.blockSignals(block)
        self.filter_mode_combo.blockSignals(block) # New
        self.filter_count_spin.blockSignals(block) # New
        self.optional_check.blockSignals(block) # New
