# -*- coding: utf-8 -*-
"""
template_params_dialog.py
LOOK_SELECT_TO_ZONE テンプレートのパラメータ入力ダイアログ。
再発防止: ダイアログは必ず exec() 後に result()==Accepted を確認してから extra_context を使うこと。
"""
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QLineEdit, QComboBox,
    QDialogButtonBox, QGroupBox, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector
from dm_toolkit.consts import CARD_TYPES, COMMAND_ZONES
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect


class LookSelectTemplateDialog(QDialog):
    """
    めくって選ぶテンプレート (LOOK_SELECT_TO_ZONE) のパラメータを入力するダイアログ。
    'その中から_文明の_種族の_カードタイプ_カードを_数量_選ぶ' の各パラメータを設定できる。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("めくって選ぶテンプレート設定"))
        self.setMinimumWidth(380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        desc = QLabel(tr(
            "デッキからN枚めくり、条件に合うカードを選んで指定ゾーンに移動するテンプレートです。"
        ))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(desc)

        form_group = QGroupBox(tr("テンプレートパラメータ"))
        form = QFormLayout(form_group)
        layout.addWidget(form_group)

        # --- めくる枚数 ---
        self.look_spin = QSpinBox()
        self.look_spin.setRange(1, 99)
        self.look_spin.setValue(4)
        form.addRow(tr("めくる枚数:"), self.look_spin)

        # --- 選ぶ枚数 (-1=すべて) ---
        self.select_spin = QSpinBox()
        self.select_spin.setRange(-1, 99)
        self.select_spin.setSpecialValueText(tr("すべて"))
        self.select_spin.setValue(-1)
        form.addRow(tr("選ぶ枚数 (すべて=-1):"), self.select_spin)

        # --- 文明フィルタ ---
        self.civ_selector = CivilizationSelector()
        form.addRow(tr("文明フィルタ:"), self.civ_selector)

        # --- 種族フィルタ ---
        self.race_edit = QLineEdit()
        self.race_edit.setPlaceholderText(tr("カンマ区切り (例: Dragon, Cyber Lord)"))
        form.addRow(tr("種族フィルタ:"), self.race_edit)

        # --- カードタイプフィルタ ---
        self.type_combo = QComboBox()
        self.type_combo.addItem(tr("指定なし"), "")
        for ct in CARD_TYPES:
            self.type_combo.addItem(tr(ct), ct)
        form.addRow(tr("カードタイプ:"), self.type_combo)

        # --- 移動先ゾーン ---
        self.zone_combo = QComboBox()
        for z in COMMAND_ZONES:
            self.zone_combo.addItem(tr(z), z)
        self.zone_combo.setCurrentText(tr("HAND"))
        form.addRow(tr("移動先ゾーン:"), self.zone_combo)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        safe_connect(buttons, 'accepted', self.accept)
        safe_connect(buttons, 'rejected', self.reject)
        layout.addWidget(buttons)

    def get_extra_context(self) -> dict:
        """ダイアログ入力値を extra_context dict として返す。"""
        civs = self.civ_selector.get_selected_civs()
        races_raw = self.race_edit.text().strip()
        races = [r.strip() for r in races_raw.split(",") if r.strip()] if races_raw else []
        type_val = self.type_combo.currentData()
        types = [type_val] if type_val else []
        return {
            "look_amount": self.look_spin.value(),
            "select_amount": self.select_spin.value(),
            "template_civs": civs,
            "template_races": races,
            "template_types": types,
            "to_zone": self.zone_combo.currentData() or "HAND",
        }
