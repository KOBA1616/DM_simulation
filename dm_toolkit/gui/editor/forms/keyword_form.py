# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QLabel, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm, get_attr, to_dict
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
            "hyper_energy", "shield_burn", "power_attacker", "ex_life"
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
            "ex_life": "EX-Life"
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
        # 再発防止: 革命チェンジ種族はノード内 target_filter で設定するためフォームに種族欄を持たない。

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

        # Friend Burst Race input (shown when friend_burst_check is checked)
        # 再発防止: フレンド・バースト種族はキーワード設定フォームで入力・保存する。
        # friend_burst_condition.races として保存され text_generator でも参照される。
        self.fb_race_label = QLabel(tr("Friend Burst Race"))
        self.fb_race_edit = QLineEdit()
        self.fb_race_edit.setPlaceholderText(tr("Comma separated races (e.g. Dragon, Cyber Lord)"))
        self.fb_race_edit.setVisible(False)
        self.fb_race_label.setVisible(False)
        self.fb_race_edit.textChanged.connect(self.update_data)
        special_layout.addWidget(self.fb_race_label)
        special_layout.addWidget(self.fb_race_edit)

        # Mega Last Burst
        self.mega_last_burst_check = QCheckBox(tr("Mega Last Burst"))
        self.mega_last_burst_check.setToolTip("メガ・ラスト・バーストを有効にすると、破壊時の効果が自動で追加されます。")
        self.mega_last_burst_check.stateChanged.connect(self.toggle_mega_last_burst)
        special_layout.addWidget(self.mega_last_burst_check)

        main_layout.addWidget(special_group)
        main_layout.addStretch()

    def toggle_rev_change(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        # 再発防止: 革命チェンジ種族はノード内 target_filter で設定するため、ここでは races を渡さない。
        self.update_data()
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_REV_CHANGE, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_REV_CHANGE, {})

    def toggle_mekraid(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_MEKRAID, STRUCT_CMD_REMOVE_MEKRAID
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        self.update_data() # Update the checkbox state in data first
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_MEKRAID, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_MEKRAID, {})

    def toggle_friend_burst(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_FRIEND_BURST, STRUCT_CMD_REMOVE_FRIEND_BURST
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        # 再発防止: フレンド・バースト種族入力フィールドはチェック時のみ表示する。
        self.fb_race_label.setVisible(is_checked)
        self.fb_race_edit.setVisible(is_checked)
        self.update_data() # Update the checkbox state in data first
        if is_checked:
            races = self._parse_races(self.fb_race_edit.text())
            self.structure_update_requested.emit(STRUCT_CMD_ADD_FRIEND_BURST, {'races': races})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_FRIEND_BURST, {})

    def toggle_mega_last_burst(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_MEGA_LAST_BURST, STRUCT_CMD_REMOVE_MEGA_LAST_BURST
        is_checked = (state == Qt.CheckState.Checked.value or state == True)
        self.update_data() # Update the checkbox state in data first
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_MEGA_LAST_BURST, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_MEGA_LAST_BURST, {})

    def _load_ui_from_data(self, data, item):
        # The item data for KEYWORDS is the 'keywords' dictionary directly
        # data parameter is the keywords dictionary extracted from the KEYWORDS tree item
        if data is None:
            data = {}
        else:
            data = to_dict(data)

        for k, cb in self.keyword_checks.items():
            is_checked = data.get(k, False)
            cb.setChecked(is_checked)

        self.rev_change_check.blockSignals(True)
        rc_checked = data.get('revolution_change', False)
        self.rev_change_check.setChecked(rc_checked)
        # 再発防止: 革命チェンジ種族はノード内 target_filter で管理。フォームに種族欄なし。
        self.rev_change_check.blockSignals(False)

        self.mekraid_check.blockSignals(True)
        self.mekraid_check.setChecked(data.get('mekraid', False))
        self.mekraid_check.blockSignals(False)

        self.friend_burst_check.blockSignals(True)
        fb_checked = data.get('friend_burst', False)
        self.friend_burst_check.setChecked(fb_checked)
        # 再発防止: チェック状態に合わせてフレンド・バースト種族入力フィールドの表示/非表示を更新する。
        self.fb_race_label.setVisible(fb_checked)
        self.fb_race_edit.setVisible(fb_checked)
        fb_cond = data.get('friend_burst_condition', {})
        if isinstance(fb_cond, dict):
            fb_races = fb_cond.get('races', [])
            self.fb_race_edit.setText(', '.join(fb_races) if fb_races else '')
        self.friend_burst_check.blockSignals(False)

        self.mega_last_burst_check.blockSignals(True)
        self.mega_last_burst_check.setChecked(data.get('mega_last_burst', False))
        self.mega_last_burst_check.blockSignals(False)

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

        # Revolution Change: 種族条件はノード内 target_filter で管理。keywords には boolean のみ保存。
        # 再発防止: revolution_change_condition をキーワードフォームから保存しないこと。
        if self.rev_change_check.isChecked():
            data['revolution_change'] = True
        elif 'revolution_change' in data:
            del data['revolution_change']
        data.pop('revolution_change_condition', None)  # 旧データのクリーンアップ

        # Mekraid
        if self.mekraid_check.isChecked():
            data['mekraid'] = True
        elif 'mekraid' in data:
            del data['mekraid']

        # Friend Burst
        if self.friend_burst_check.isChecked():
            data['friend_burst'] = True
            # 再発防止: フレンド・バースト種族は friend_burst_condition.races に保存する。
            # text_generator からも keywords dict 内の friend_burst_condition を参照できるよう変更済み。
            fb_races = self._parse_races(self.fb_race_edit.text())
            if fb_races:
                data['friend_burst_condition'] = {'races': fb_races}
            elif 'friend_burst_condition' in data:
                del data['friend_burst_condition']
        elif 'friend_burst' in data:
            del data['friend_burst']
            data.pop('friend_burst_condition', None)

        # Mega Last Burst
        if self.mega_last_burst_check.isChecked():
            data['mega_last_burst'] = True
        elif 'mega_last_burst' in data:
            del data['mega_last_burst']

    @staticmethod
    def _parse_races(text: str) -> list:
        """カンマ区切りテキストを種族リストに変換する。空文字・空白は除外。"""
        if not text or not text.strip():
            return []
        return [r.strip() for r in text.split(',') if r.strip()]

    def _get_display_text(self, data):
        return tr("Keywords")

    def block_signals_all(self, block):
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.rev_change_check.blockSignals(block)
        self.rc_race_edit.blockSignals(block)
        self.mekraid_check.blockSignals(block)
        self.friend_burst_check.blockSignals(block)
        self.fb_race_edit.blockSignals(block)
        self.mega_last_burst_check.blockSignals(block)
