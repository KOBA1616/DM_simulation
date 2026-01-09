from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_REV_CHANGE, STRUCT_CMD_REMOVE_REV_CHANGE

class KeywordEditForm(BaseEditForm):
    # Signal to request structural changes in the Logic Tree (e.g. for Revolution Change)
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults
        self.keyword_checks = getattr(self, 'keyword_checks', {})
        self.label = getattr(self, 'label', None)
        try:
            self.setup_ui()
        except Exception:
            pass

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
            "ex_life": "EX-Life",
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
        self.rev_change_check.setToolTip("革命チェンジを有効にすると、必要なロジックツリー構造が生成されます。")
        self.rev_change_check.stateChanged.connect(self.toggle_rev_change)
        special_layout.addWidget(self.rev_change_check)

        # Mekraid
        self.mekraid_check = QCheckBox(tr("Mekraid"))
        self.mekraid_check.setToolTip("メクレイドを有効にすると、プレイ時の効果が自動で追加されます。")
        self.mekraid_check.stateChanged.connect(self.toggle_mekraid)
        special_layout.addWidget(self.mekraid_check)

        # Friend Burst
        self.friend_burst_check = QCheckBox(tr("Friend Burst"))
        self.friend_burst_check.setToolTip("フレンド・バーストを有効にすると、攻撃時の効果が自動で追加されます。")
        self.friend_burst_check.stateChanged.connect(self.toggle_friend_burst)
        special_layout.addWidget(self.friend_burst_check)

        main_layout.addWidget(special_group)
        main_layout.addStretch()

    def toggle_rev_change(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_REV_CHANGE, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_REV_CHANGE, {})
        self.update_data() # Update the checkbox state in data too

    def toggle_mekraid(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_MEKRAID, STRUCT_CMD_REMOVE_MEKRAID
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_MEKRAID, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_MEKRAID, {})
        self.update_data()

    def toggle_friend_burst(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_FRIEND_BURST, STRUCT_CMD_REMOVE_FRIEND_BURST
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_FRIEND_BURST, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_FRIEND_BURST, {})
        self.update_data()

    def _load_ui_from_data(self, data, item):
        # The item data for KEYWORDS is the 'keywords' dictionary directly
        # data parameter is the keywords dictionary extracted from the KEYWORDS tree item
        if data is None:
            data = {}

        for k, cb in self.keyword_checks.items():
            is_checked = data.get(k, False)
            cb.setChecked(is_checked)

        self.rev_change_check.blockSignals(True)
        self.rev_change_check.setChecked(data.get('revolution_change', False))
        self.rev_change_check.blockSignals(False)

        self.mekraid_check.blockSignals(True)
        self.mekraid_check.setChecked(data.get('mekraid', False))
        self.mekraid_check.blockSignals(False)

        self.friend_burst_check.blockSignals(True)
        self.friend_burst_check.setChecked(data.get('friend_burst', False))
        self.friend_burst_check.blockSignals(False)

    def _save_ui_to_data(self, data):
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

        # Mekraid
        if self.mekraid_check.isChecked():
            data['mekraid'] = True
        elif 'mekraid' in data:
            del data['mekraid']

        # Friend Burst
        if self.friend_burst_check.isChecked():
            data['friend_burst'] = True
        elif 'friend_burst' in data:
            del data['friend_burst']

    def _get_display_text(self, data):
        return tr("Keywords")

    def block_signals_all(self, block):
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.rev_change_check.blockSignals(block)
        self.mekraid_check.blockSignals(block)
        self.friend_burst_check.blockSignals(block)
