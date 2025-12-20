from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm

class KeywordEditForm(BaseEditForm):
    # Signal to request structural changes in the Logic Tree (e.g. for Revolution Change)
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyword_checks = {} # Map key -> QCheckBox
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.label = QLabel(tr("Keywords Configuration"))
        main_layout.addWidget(self.label)

        # Keywords Group
        kw_group = QGroupBox(tr("Standard Keywords"))
        kw_layout = QGridLayout(kw_group)

        keywords_list = [
            "speed_attacker", "blocker", "slayer",
            "double_breaker", "triple_breaker", "shield_trigger",
            "just_diver", "mach_fighter", "g_strike",
            "hyper_energy", "shield_burn", "power_attacker", "ex_life",
            "mega_last_burst"
        ]

        kw_map = {
            "speed_attacker": "Speed Attacker",
            "blocker": "Blocker",
            "slayer": "Slayer",
            "double_breaker": "Double Breaker",
            "triple_breaker": "Triple Breaker",
            "shield_trigger": "Shield Trigger",
            "just_diver": "Just Diver",
            "mach_fighter": "Mach Fighter",
            "g_strike": "G Strike",
            "hyper_energy": "Hyper Energy",
            "shield_burn": "Shield Burn",
            "power_attacker": "Power Attacker",
            "ex_life": "Ex-Life",
            "mega_last_burst": "Mega Last Burst"
        }

        row = 0
        col = 0
        for k in keywords_list:
            cb = QCheckBox(tr(kw_map.get(k, k)))
            kw_layout.addWidget(cb, row, col)
            self.keyword_checks[k] = cb
            cb.stateChanged.connect(self.update_data)

            col += 1
            if col > 2: # 3 columns
                col = 0
                row += 1

        main_layout.addWidget(kw_group)

        # Special Keywords Group (that imply structure)
        special_group = QGroupBox(tr("Special Keywords"))
        special_layout = QVBoxLayout(special_group)

        # Revolution Change
        self.rev_change_check = QCheckBox(tr("Revolution Change"))
        self.rev_change_check.setToolTip(tr("Enable Revolution Change to generate the necessary logic tree structure."))
        self.rev_change_check.stateChanged.connect(self.toggle_rev_change)
        special_layout.addWidget(self.rev_change_check)

        main_layout.addWidget(special_group)
        main_layout.addStretch()

    def toggle_rev_change(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit("ADD_REV_CHANGE", {})
        else:
            self.structure_update_requested.emit("REMOVE_REV_CHANGE", {})
        self.update_data() # Update the checkbox state in data too

    def _populate_ui(self, item):
        # The item data for KEYWORDS is the 'keywords' dictionary directly?
        # Or is it the card data?
        # In DataManager, we will set data to the 'keywords' dict.
        # However, BaseEditForm expects a dict.
        # Let's assume the item data IS the keywords dict.

        # NOTE: QStandardItem data is returned by value (copy) in Python often,
        # but we need to verify if modifying it here updates the model correctly via update_data.
        # update_data calls _save_data then sets it back.

        kw_data = item.data(Qt.ItemDataRole.UserRole + 2)
        if kw_data is None: kw_data = {}

        for k, cb in self.keyword_checks.items():
            is_checked = kw_data.get(k, False)
            cb.setChecked(is_checked)

        self.rev_change_check.blockSignals(True)
        self.rev_change_check.setChecked(kw_data.get('revolution_change', False))
        self.rev_change_check.blockSignals(False)

        # Check parent card for Twinpact to toggle Mega Last Burst visibility?
        # This form only sees its item. The LogicTree might need to manage this visibility or we pass context.
        # For now, let's leave it visible or always enabled. The user requirement doesn't specify strict context here.

    def _save_data(self, data):
        # 'data' is the keywords dictionary

        # Reset managed keywords
        for k in self.keyword_checks.keys():
            if k in data:
                del data[k]

        # Add checked ones
        for k, cb in self.keyword_checks.items():
            if cb.isChecked():
                data[k] = True

        # Revolution Change
        if self.rev_change_check.isChecked():
            data['revolution_change'] = True
        elif 'revolution_change' in data:
            del data['revolution_change']

    def _get_display_text(self, data):
        return tr("Keywords")

    def block_signals_all(self, block):
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.rev_change_check.blockSignals(block)
