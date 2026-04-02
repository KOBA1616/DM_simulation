# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QLabel, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from dataclasses import dataclass
from typing import Any
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm, get_attr, to_dict
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
from dm_toolkit.gui.editor.consts import (
    STRUCT_CMD_ADD_REV_CHANGE,
    STRUCT_CMD_REMOVE_REV_CHANGE,
    STRUCT_CMD_ADD_MEKRAID,
    STRUCT_CMD_REMOVE_MEKRAID,
    STRUCT_CMD_ADD_FRIEND_BURST,
    STRUCT_CMD_REMOVE_FRIEND_BURST,
    STRUCT_CMD_ADD_DANGEROUS_DASH,
    STRUCT_CMD_REMOVE_DANGEROUS_DASH,
)


@dataclass
class KeywordFormState:
    keyword_flags: dict[str, bool]
    revolution_change: bool
    mekraid: bool
    mekraid_races: list[str]
    friend_burst: bool
    friend_burst_races: list[str]
    dangerous_dash: bool
    dangerous_dash_civs: list[str]
    dangerous_dash_cost: int
    dangerous_dash_text: str
    mega_last_burst: bool

    def apply_to_data(self, data: dict[str, Any]) -> None:
        # 再発防止: 画面部品から直接 dict を散発更新すると条件キーの消し忘れが起きやすいため、
        # 状態オブジェクト経由で一括反映する。
        for key in list(data.keys()):
            if key in self.keyword_flags and not self.keyword_flags[key]:
                data.pop(key, None)

        for key, enabled in self.keyword_flags.items():
            if enabled:
                data[key] = True
            else:
                data.pop(key, None)

        # 再発防止: 革命チェンジはキーワード直設定ではなく
        # REVOLUTION_CHANGE ノードの有無を正とする設計に統一する。
        # KeywordForm は構造生成トリガーのみを担い、保存時に rc キーを直接更新しない。
        data.pop("revolution_change", None)
        data.pop("revolution_change_condition", None)

        if self.mekraid:
            data["mekraid"] = True
            if self.mekraid_races:
                data["mekraid_condition"] = {"races": self.mekraid_races}
            else:
                data.pop("mekraid_condition", None)
        else:
            data.pop("mekraid", None)
            data.pop("mekraid_condition", None)

        if self.friend_burst:
            data["friend_burst"] = True
            if self.friend_burst_races:
                data["friend_burst_condition"] = {"races": self.friend_burst_races}
            else:
                data.pop("friend_burst_condition", None)
        else:
            data.pop("friend_burst", None)
            data.pop("friend_burst_condition", None)

        if self.dangerous_dash:
            data["dangerous_dash"] = True
            cond = {}
            if self.dangerous_dash_civs:
                cond["civilizations"] = self.dangerous_dash_civs
            if self.dangerous_dash_cost > 0:
                cond["cost"] = self.dangerous_dash_cost
            if self.dangerous_dash_text:
                cond["raw_text"] = self.dangerous_dash_text
            if cond:
                data["dangerous_dash_condition"] = cond
            else:
                data.pop("dangerous_dash_condition", None)
        else:
            data.pop("dangerous_dash", None)
            data.pop("dangerous_dash_condition", None)

        if self.mega_last_burst:
            data["mega_last_burst"] = True
        else:
            data.pop("mega_last_burst", None)

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
            safe_connect(cb, "stateChanged", self.update_data)

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
        self.rev_change_check.setToolTip(tr("革命チェンジを有効にすると、必要なロジックツリー構造が生成されます。"))
        safe_connect(self.rev_change_check, "stateChanged", self.toggle_rev_change)
        special_layout.addWidget(self.rev_change_check)

        # Mekraid
        self.mekraid_check = QCheckBox(tr("Mekraid"))
        self.mekraid_check.setToolTip(tr("メクレイドを有効にすると、プレイ時の効果が自動で追加されます。"))
        safe_connect(self.mekraid_check, "stateChanged", self.toggle_mekraid)
        special_layout.addWidget(self.mekraid_check)
        self.mk_race_label = QLabel(tr("Mekraid Race"))
        self.mk_race_edit = QLineEdit()
        self.mk_race_edit.setPlaceholderText(tr("Comma separated races (e.g. Fire Bird, Armored Dragon)"))
        self.mk_race_label.setVisible(False)
        self.mk_race_edit.setVisible(False)
        safe_connect(self.mk_race_edit, "textChanged", self.update_data)
        safe_connect(self.mk_race_edit, "editingFinished", self._on_mekraid_race_edited)
        special_layout.addWidget(self.mk_race_label)
        special_layout.addWidget(self.mk_race_edit)

        # Friend Burst
        self.friend_burst_check = QCheckBox(tr("Friend Burst"))
        self.friend_burst_check.setToolTip(tr("フレンド・バーストを有効にすると、攻撃時の効果が自動で追加されます。"))
        safe_connect(self.friend_burst_check, "stateChanged", self.toggle_friend_burst)
        special_layout.addWidget(self.friend_burst_check)

        # Friend Burst Race input (shown when friend_burst_check is checked)
        # 再発防止: フレンド・バースト種族はキーワード設定フォームで入力・保存する。
        # friend_burst_condition.races として保存され text_generator でも参照される。
        self.fb_race_label = QLabel(tr("Friend Burst Race"))
        self.fb_race_edit = QLineEdit()
        self.fb_race_edit.setPlaceholderText(tr("Comma separated races (e.g. Dragon, Cyber Lord)"))
        self.fb_race_edit.setVisible(False)
        self.fb_race_label.setVisible(False)
        safe_connect(self.fb_race_edit, "textChanged", self.update_data)
        safe_connect(self.fb_race_edit, "editingFinished", self._on_friend_burst_race_edited)
        special_layout.addWidget(self.fb_race_label)
        special_layout.addWidget(self.fb_race_edit)

        # Dangerous Dash
        self.dangerous_dash_check = QCheckBox(tr("D・D・D"))
        self.dangerous_dash_check.setToolTip(tr("D・D・D（デデデ・デンジャラ・ダッシュ）を有効にすると、攻撃時の効果が自動で追加されます。"))
        safe_connect(self.dangerous_dash_check, "stateChanged", self.toggle_dangerous_dash)
        special_layout.addWidget(self.dangerous_dash_check)

        self.dd_civs_label = QLabel(tr("D・D・D Civilizations"))
        self.dd_civs_edit = QLineEdit()
        self.dd_civs_edit.setPlaceholderText(tr("e.g. FIRE, ZERO"))
        self.dd_civs_label.setVisible(False)
        self.dd_civs_edit.setVisible(False)
        safe_connect(self.dd_civs_edit, "textChanged", self.update_data)
        safe_connect(self.dd_civs_edit, "editingFinished", self._on_dangerous_dash_edited)
        special_layout.addWidget(self.dd_civs_label)
        special_layout.addWidget(self.dd_civs_edit)

        self.dd_cost_label = QLabel(tr("D・D・D Cost"))
        self.dd_cost_edit = QLineEdit()
        self.dd_cost_edit.setPlaceholderText(tr("Cost value"))
        self.dd_cost_label.setVisible(False)
        self.dd_cost_edit.setVisible(False)
        safe_connect(self.dd_cost_edit, "textChanged", self.update_data)
        safe_connect(self.dd_cost_edit, "editingFinished", self._on_dangerous_dash_edited)
        special_layout.addWidget(self.dd_cost_label)
        special_layout.addWidget(self.dd_cost_edit)

        self.dd_text_label = QLabel(tr("D・D・D Raw Text"))
        self.dd_text_edit = QLineEdit()
        self.dd_text_edit.setPlaceholderText(tr("Custom condition (e.g. 火のカードを1枚自分の手札から捨てる)"))
        self.dd_text_label.setVisible(False)
        self.dd_text_edit.setVisible(False)
        safe_connect(self.dd_text_edit, "textChanged", self.update_data)
        safe_connect(self.dd_text_edit, "editingFinished", self._on_dangerous_dash_edited)
        special_layout.addWidget(self.dd_text_label)
        special_layout.addWidget(self.dd_text_edit)

        # Mega Last Burst
        self.mega_last_burst_check = QCheckBox(tr("Mega Last Burst"))
        self.mega_last_burst_check.setToolTip(tr("メガ・ラスト・バーストを有効にすると、破壊時の効果が自動で追加されます。"))
        safe_connect(self.mega_last_burst_check, "stateChanged", self.toggle_mega_last_burst)
        special_layout.addWidget(self.mega_last_burst_check)

        main_layout.addWidget(special_group)
        main_layout.addStretch()

    @staticmethod
    def _is_checked_state(state: object) -> bool:
        # 再発防止: ヘッドレステスト用スタブでは Qt.CheckState が無い場合があるため、
        # bool/int/Enum いずれでも判定できる共通関数に寄せる。
        if isinstance(state, bool):
            return state
        try:
            check_state = getattr(Qt, "CheckState", None)
            checked = getattr(check_state, "Checked", None)
            checked_value = getattr(checked, "value", checked)
            if checked_value is not None and state == checked_value:
                return True
        except Exception:
            pass
        try:
            checked_const = getattr(Qt, "Checked", None)
            if checked_const is not None and state == checked_const:
                return True
        except Exception:
            pass
        return state == 2

    def toggle_rev_change(self, state):
        is_checked = self._is_checked_state(state)
        self.update_data()
        if is_checked:
            # 再発防止: 革命チェンジは種族入力を持たず、チェック操作でテンプレートを直接生成する。
            self.structure_update_requested.emit(STRUCT_CMD_ADD_REV_CHANGE, {})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_REV_CHANGE, {})

    def toggle_mekraid(self, state):
        is_checked = self._is_checked_state(state)
        self.mk_race_label.setVisible(is_checked)
        self.mk_race_edit.setVisible(is_checked)
        self.update_data() # Update the checkbox state in data first
        payload = {'races': self._parse_races(self.mk_race_edit.text())}
        if is_checked:
            self.structure_update_requested.emit(STRUCT_CMD_ADD_MEKRAID, payload)
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_MEKRAID, {})

    def toggle_friend_burst(self, state):
        is_checked = self._is_checked_state(state)
        # 再発防止: フレンド・バースト種族入力フィールドはチェック時のみ表示する。
        self.fb_race_label.setVisible(is_checked)
        self.fb_race_edit.setVisible(is_checked)
        self.update_data() # Update the checkbox state in data first
        if is_checked:
            races = self._parse_races(self.fb_race_edit.text())
            self.structure_update_requested.emit(STRUCT_CMD_ADD_FRIEND_BURST, {'races': races})
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_FRIEND_BURST, {})

    def _on_mekraid_race_edited(self):
        # 再発防止: 種族編集後に既存テンプレートを再生成し、コマンド条件へ反映する。
        if not self.mekraid_check.isChecked():
            return
        races = self._parse_races(self.mk_race_edit.text())
        self.update_data()
        self.structure_update_requested.emit(STRUCT_CMD_REMOVE_MEKRAID, {})
        self.structure_update_requested.emit(STRUCT_CMD_ADD_MEKRAID, {'races': races})

    def _on_friend_burst_race_edited(self):
        # 再発防止: 種族編集後に既存テンプレートを再生成し、コマンド条件へ反映する。
        if not self.friend_burst_check.isChecked():
            return
        races = self._parse_races(self.fb_race_edit.text())
        self.update_data()
        self.structure_update_requested.emit(STRUCT_CMD_REMOVE_FRIEND_BURST, {})
        self.structure_update_requested.emit(STRUCT_CMD_ADD_FRIEND_BURST, {'races': races})

    def toggle_dangerous_dash(self, state):
        is_checked = self._is_checked_state(state)
        self.dd_civs_label.setVisible(is_checked)
        self.dd_civs_edit.setVisible(is_checked)
        self.dd_cost_label.setVisible(is_checked)
        self.dd_cost_edit.setVisible(is_checked)
        self.dd_text_label.setVisible(is_checked)
        self.dd_text_edit.setVisible(is_checked)
        self.update_data()
        if is_checked:
            payload = {
                'civilizations': self._parse_races(self.dd_civs_edit.text()),
                'cost': int(self.dd_cost_edit.text()) if self.dd_cost_edit.text().isdigit() else 0,
                'raw_text': self.dd_text_edit.text().strip()
            }
            self.structure_update_requested.emit(STRUCT_CMD_ADD_DANGEROUS_DASH, payload)
        else:
            self.structure_update_requested.emit(STRUCT_CMD_REMOVE_DANGEROUS_DASH, {})

    def _on_dangerous_dash_edited(self):
        if not self.dangerous_dash_check.isChecked():
            return
        payload = {
            'civilizations': self._parse_races(self.dd_civs_edit.text()),
            'cost': int(self.dd_cost_edit.text()) if self.dd_cost_edit.text().isdigit() else 0,
            'raw_text': self.dd_text_edit.text().strip()
        }
        self.update_data()
        self.structure_update_requested.emit(STRUCT_CMD_REMOVE_DANGEROUS_DASH, {})
        self.structure_update_requested.emit(STRUCT_CMD_ADD_DANGEROUS_DASH, payload)

    def toggle_mega_last_burst(self, state):
        from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_MEGA_LAST_BURST, STRUCT_CMD_REMOVE_MEGA_LAST_BURST
        is_checked = self._is_checked_state(state)
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
        # 再発防止: 革命チェンジは keyword dict に保存しないため、
        # 既存ノード（REVOLUTION_CHANGE / MUTATE+mutation_kind）からチェック状態を復元する。
        rc_checked = self._has_revolution_change_node(item)
        self.rev_change_check.setChecked(rc_checked)
        self.rev_change_check.blockSignals(False)

        self.mekraid_check.blockSignals(True)
        mk_checked = data.get('mekraid', False)
        self.mekraid_check.setChecked(mk_checked)
        self.mk_race_label.setVisible(mk_checked)
        self.mk_race_edit.setVisible(mk_checked)
        mk_cond = data.get('mekraid_condition', {})
        if isinstance(mk_cond, dict):
            mk_races = mk_cond.get('races', [])
            self.mk_race_edit.setText(", ".join(mk_races) if mk_races else '')
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
            self.fb_race_edit.setText(tr(", ").join(fb_races) if fb_races else '')
        self.friend_burst_check.blockSignals(False)

        self.dangerous_dash_check.blockSignals(True)
        dd_checked = data.get('dangerous_dash', False)
        self.dangerous_dash_check.setChecked(dd_checked)
        self.dd_civs_label.setVisible(dd_checked)
        self.dd_civs_edit.setVisible(dd_checked)
        self.dd_cost_label.setVisible(dd_checked)
        self.dd_cost_edit.setVisible(dd_checked)
        self.dd_text_label.setVisible(dd_checked)
        self.dd_text_edit.setVisible(dd_checked)
        dd_cond = data.get('dangerous_dash_condition', {})
        if isinstance(dd_cond, dict):
            dd_civs = dd_cond.get('civilizations', [])
            self.dd_civs_edit.setText(", ".join(dd_civs) if dd_civs else "")
            dd_cost = dd_cond.get('cost', 0)
            self.dd_cost_edit.setText(str(dd_cost) if dd_cost > 0 else "")
            self.dd_text_edit.setText(dd_cond.get('raw_text', ""))
        self.dangerous_dash_check.blockSignals(False)

        self.mega_last_burst_check.blockSignals(True)
        self.mega_last_burst_check.setChecked(data.get('mega_last_burst', False))
        self.mega_last_burst_check.blockSignals(False)

    def _save_ui_to_data(self, data):
        state = self._collect_state_from_ui()
        state.apply_to_data(data)

    def _collect_state_from_ui(self) -> KeywordFormState:
        keyword_flags = {
            k: bool(cb.isChecked()) for k, cb in self.keyword_checks.items()
        }
        return KeywordFormState(
            keyword_flags=keyword_flags,
            revolution_change=bool(self.rev_change_check.isChecked()),
            mekraid=bool(self.mekraid_check.isChecked()),
            mekraid_races=self._parse_races(self.mk_race_edit.text()),
            friend_burst=bool(self.friend_burst_check.isChecked()),
            friend_burst_races=self._parse_races(self.fb_race_edit.text()),
            dangerous_dash=bool(self.dangerous_dash_check.isChecked()),
            dangerous_dash_civs=self._parse_races(self.dd_civs_edit.text()),
            dangerous_dash_cost=int(self.dd_cost_edit.text()) if self.dd_cost_edit.text().isdigit() else 0,
            dangerous_dash_text=self.dd_text_edit.text().strip(),
            mega_last_burst=bool(self.mega_last_burst_check.isChecked()),
        )

    @staticmethod
    def _parse_races(text: str) -> list:
        """カンマ区切りテキストを種族リストに変換する。空文字・空白は除外。"""
        if not text or not text.strip():
            return []
        normalized = text.replace('、', ',')
        return [r.strip() for r in normalized.split(',') if r.strip()]

    def _get_display_text(self, data):
        return tr("Keywords")

    def _has_revolution_change_node(self, item: Any) -> bool:
        """Detect Revolution Change from sibling effect command nodes."""
        if item is None:
            return False
        try:
            card_item = item.parent()
        except Exception:
            card_item = None
        if card_item is None:
            return False

        try:
            child_count = card_item.rowCount()
        except Exception:
            child_count = 0

        for i in range(child_count):
            try:
                child = card_item.child(i)
                role = child.data(Qt.ItemDataRole.UserRole + 1)
            except Exception:
                continue
            if role != "EFFECT":
                continue

            try:
                eff_data = to_dict(child.data(Qt.ItemDataRole.UserRole + 2) or {})
            except Exception:
                eff_data = {}
            commands = eff_data.get("commands", []) if isinstance(eff_data, dict) else []
            for cmd in commands:
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if ctype == "REVOLUTION_CHANGE":
                    return True

        return False

    def block_signals_all(self, block):
        # 再発防止: 革命チェンジ種族入力UIは削除済みのため、チェックボックスのみ制御する。
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
        self.rev_change_check.blockSignals(block)
        self.mekraid_check.blockSignals(block)
        self.mk_race_edit.blockSignals(block)
        self.friend_burst_check.blockSignals(block)
        self.fb_race_edit.blockSignals(block)
        self.dangerous_dash_check.blockSignals(block)
        self.dd_civs_edit.blockSignals(block)
        self.dd_cost_edit.blockSignals(block)
        self.dd_text_edit.blockSignals(block)
        self.mega_last_burst_check.blockSignals(block)
