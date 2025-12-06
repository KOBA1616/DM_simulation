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
            "COST_REFERENCE", "COST_REDUCTION", "NONE", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "DISCARD", "PLAY_FROM_ZONE",
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

        # Re-ordering: Mode (str_val_combo) should be above Filter
        # Since Filter is self.filter_group added via layout.addRow, we need to insert before it?
        # QFormLayout adds rows sequentially.
        # We constructed scope above.

        # New Order: Scope -> Mode -> Filter -> Values -> Variable Link

        self.str_val_label = QLabel(tr("String Value"))
        self.str_val_edit = QLineEdit()
        self.str_val_combo = QComboBox() # Added ComboBox for select stats

        # Populate Stats Options
        self.str_val_combo.addItem(tr("Cards matching filter"), "CARDS_MATCHING_FILTER")
        stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
        for s in stats:
            self.str_val_combo.addItem(tr(s), s)

        # Add Ref Mode Combo (New Requirement)
        self.ref_mode_combo = QComboBox()
        # Populate Ref Mode Options
        ref_modes = ["SYM_CREATURE", "SYM_SPELL", "G_ZERO", "HYPER_ENERGY", "NONE"]
        for rm in ref_modes:
            self.ref_mode_combo.addItem(tr(rm), rm)

        # Add Mode (str_val) related widgets to layout NOW (before filter group)
        # Note: label visibility is managed by update_ui_state, but position is here.
        layout.addRow(self.str_val_label, self.str_val_edit)

        # We need to control the visibility of the "Mode" label explicitly.
        # QFormLayout returns a QFormLayout.ItemRole (layout item), which is hard to access later.
        # Instead, we create a QLabel explicitly and add it as a row with the widget,
        # so we can reference the label later.
        self.mode_label = QLabel(tr("Mode"))
        layout.addRow(self.mode_label, self.str_val_combo)

        # Add Ref Mode Combo row (hidden by default)
        self.ref_mode_label = QLabel(tr("Ref Mode"))
        layout.addRow(self.ref_mode_label, self.ref_mode_combo)

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

        # Arbitrary Amount Checkbox
        self.arbitrary_check = QCheckBox(tr("Arbitrary Amount (Up to N)"))
        layout.addRow(self.arbitrary_check)

        # Variable Linking
        self.smart_link_check = QCheckBox(tr("Use result from previous measurement"))
        layout.addRow(self.smart_link_check)

        self.input_key_combo = QComboBox()
        self.input_key_combo.setEditable(False) # Disable editing to hide raw keys
        self.input_key_label = QLabel(tr("Input Key"))
        layout.addRow(self.input_key_label, self.input_key_combo)

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
        self.ref_mode_combo.currentIndexChanged.connect(self.update_data) # New handler
        self.arbitrary_check.stateChanged.connect(self.update_data)
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

        self.input_key_label.setVisible(is_checked)
        self.input_key_combo.setVisible(is_checked)

        # 2. Logic to auto-set input key if checked
        if is_checked:
            self.populate_input_keys()
            count = self.input_key_combo.count()
            if count > 0:
                # Default to last available output
                last_idx = count - 1
                self.input_key_combo.setCurrentIndex(last_idx)
        else:
            self.input_key_combo.setCurrentIndex(-1)

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
        elif action_type == "COST_REDUCTION":
            config = ACTION_UI_CONFIG.get("COST_REDUCTION", config)

        # Update Labels
        self.val1_label.setText(tr(config["val1_label"]))
        self.val2_label.setText(tr(config["val2_label"]))
        self.str_val_label.setText(tr(config["str_label"]))

        # Smart Link Visibility
        can_link_input = config["val1_visible"]
        self.smart_link_check.setVisible(can_link_input)

        # Arbitrary Amount Visibility
        self.arbitrary_check.setVisible(config.get("can_be_optional", False))

        is_smart_linked = self.smart_link_check.isChecked() and can_link_input

        # Update Visibility
        self.val1_label.setVisible(config["val1_visible"] and not is_smart_linked)
        self.val1_spin.setVisible(config["val1_visible"] and not is_smart_linked)

        self.val2_label.setVisible(config["val2_visible"])
        self.val2_spin.setVisible(config["val2_visible"])

        # Special handling for unified COUNT_CARDS / GET_GAME_STAT
        if action_type == "GET_GAME_STAT" or action_type == "COUNT_CARDS":
            self.str_val_label.setVisible(False) # Hide the generic label
            self.str_val_edit.setVisible(False)

            # Show Mode Combo and its label
            self.mode_label.setVisible(True)
            self.str_val_combo.setVisible(True)

            # Hide Ref Mode
            self.ref_mode_label.setVisible(False)
            self.ref_mode_combo.setVisible(False)

            # Determine filter visibility based on Combo Selection
            current_mode = self.str_val_combo.currentData()
            if current_mode == "CARDS_MATCHING_FILTER" or current_mode is None:
                self.filter_group.setVisible(True)
            else:
                self.filter_group.setVisible(False)

        elif action_type == "COST_REFERENCE":
            # New Ref Mode Combo Logic
            self.str_val_label.setVisible(False)
            self.str_val_edit.setVisible(False)

            self.mode_label.setVisible(False)
            self.str_val_combo.setVisible(False)

            self.ref_mode_label.setVisible(True)
            self.ref_mode_combo.setVisible(True)

            self.filter_group.setVisible(config["filter_visible"])

        else:
            self.str_val_label.setVisible(config["str_visible"])
            self.str_val_edit.setVisible(config["str_visible"])

            # Hide Mode Combo and its label
            self.mode_label.setVisible(False)
            self.str_val_combo.setVisible(False)

            # Hide Ref Mode
            self.ref_mode_label.setVisible(False)
            self.ref_mode_combo.setVisible(False)

            self.filter_group.setVisible(config["filter_visible"])

        # Hide Input Key stuff by default unless Smart Link is checked
        # Actually, smart_link_check visibility is controlled above.
        self.input_key_label.setVisible(self.smart_link_check.isChecked())
        self.input_key_combo.setVisible(self.smart_link_check.isChecked())

        # Update Tooltips
        self.type_combo.setToolTip(tr(config.get("tooltip", "")))

    def set_data(self, item):
        # CRITICAL FIX: Unset current_item first to prevent overwrite
        self.current_item = None

        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)

        input_key = data.get('input_value_key', '')
        if input_key:
            self.smart_link_check.setChecked(True)
        else:
            self.smart_link_check.setChecked(False)

        # Set Action Type
        raw_type = data.get('type', 'NONE')
        str_val = data.get('str_val', '')

        # UI Mapping
        ui_type = raw_type
        if raw_type == "GET_GAME_STAT":
            ui_type = "COUNT_CARDS"
        elif raw_type == "COST_REFERENCE":
             # Already handled by default mapping, but we set combo later
             pass
        elif raw_type == "APPLY_MODIFIER" and str_val == "COST":
             ui_type = "COST_REDUCTION"

        t_idx = self.type_combo.findData(ui_type)
        if t_idx >= 0:
            self.type_combo.setCurrentIndex(t_idx)

        # Handle Mode Combo for Unified Type
        if raw_type == "COUNT_CARDS":
             c_idx = self.str_val_combo.findData("CARDS_MATCHING_FILTER")
             if c_idx >= 0: self.str_val_combo.setCurrentIndex(c_idx)
        elif raw_type == "GET_GAME_STAT":
             c_idx = self.str_val_combo.findData(str_val)
             if c_idx >= 0: self.str_val_combo.setCurrentIndex(c_idx)

        # Handle Ref Mode Combo
        if raw_type == "COST_REFERENCE":
            r_idx = self.ref_mode_combo.findData(str_val)
            if r_idx >= 0:
                self.ref_mode_combo.setCurrentIndex(r_idx)
            else:
                # If custom value (unlikely but safe), add it?
                # Or default to NONE?
                # For now assume it's one of the options or we default
                self.ref_mode_combo.setCurrentIndex(self.ref_mode_combo.findData("NONE"))

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

        # Only set text if not using combos
        if ui_type != "COUNT_CARDS" and ui_type != "GET_GAME_STAT" and ui_type != "COST_REFERENCE":
             self.str_val_edit.setText(str_val)

        # Optional (Arbitrary Amount)
        self.arbitrary_check.setChecked(data.get('optional', False))

        # Variable Linking Population
        # We need item context to populate input keys
        # We temporarily set current_item to item just for this call?
        # No, populate_input_keys uses self.current_item.
        # But we set it to None.
        # We can pass item to populate_input_keys.
        self.populate_input_keys(item)

        found_idx = -1
        for i in range(self.input_key_combo.count()):
             if self.input_key_combo.itemData(i) == input_key:
                  found_idx = i
                  break

        if found_idx >= 0:
             self.input_key_combo.setCurrentIndex(found_idx)
        else:
             # Fallback if key exists but not found in tree (e.g. deleted step)
             # If we want to hide it, we might just show "Unknown" or add it as text
             if input_key:
                 self.input_key_combo.addItem(input_key, input_key)
                 self.input_key_combo.setCurrentIndex(self.input_key_combo.count()-1)

        self.output_key_edit.setText(data.get('output_value_key', ''))

        self.block_signals(False)

        # Finally set current item
        self.current_item = item

    def populate_input_keys(self, item=None):
        self.input_key_combo.clear()
        target_item = item if item else self.current_item
        if not target_item: return

        # Traverse siblings upwards
        parent = target_item.parent()
        if not parent: return

        row = target_item.row()
        for i in range(row):
            sibling = parent.child(i)
            sib_data = sibling.data(Qt.ItemDataRole.UserRole + 2)
            out_key = sib_data.get('output_value_key')
            if out_key:
                # Format: "Step {index}: {Type}"
                # Hiding the raw key from user view as requested
                type_disp = tr(sib_data.get('type'))
                label = f"Step {i}: {type_disp}"
                self.input_key_combo.addItem(label, out_key)

    def update_data(self):
        if not self.current_item: return

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

        elif action_type == "COST_REDUCTION":
            data['type'] = "APPLY_MODIFIER" # Map to engine support
            data['str_val'] = "COST" # Mode

        elif action_type == "COST_REFERENCE":
            data['type'] = "COST_REFERENCE"
            data['str_val'] = self.ref_mode_combo.currentData()

        else:
            data['type'] = action_type
            # For non-stat types, read string edit
            # But wait, did we overwrite str_val for COUNT_CARDS above? Yes.
            # If we are here, it's NOT COUNT_CARDS (UI type) and NOT COST_REFERENCE.
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

        data['optional'] = self.arbitrary_check.isChecked()

        # Input Key
        idx = self.input_key_combo.currentIndex()
        if idx >= 0:
             # Use itemData which holds the raw key
             data['input_value_key'] = self.input_key_combo.itemData(idx)
        else:
             # Fallback (should not happen with Editable=False)
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
        elif data['type'] == "APPLY_MODIFIER" and data['str_val'] == "COST":
             display_type = tr("COST_REDUCTION")
        elif data['type'] == "COST_REFERENCE":
             display_type = f"{tr('COST_REFERENCE')} ({tr(data['str_val'])})"

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
        self.filter_mode_combo.blockSignals(block)
        self.filter_count_spin.blockSignals(block)
        self.arbitrary_check.blockSignals(block) # New
        self.smart_link_check.blockSignals(block) # New
        self.ref_mode_combo.blockSignals(block) # New
