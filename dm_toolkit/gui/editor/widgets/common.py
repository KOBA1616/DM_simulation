# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QComboBox, QSpinBox, QCheckBox, QLineEdit, QWidget, QHBoxLayout
# from dm_toolkit.gui.editor.widgets.interfaces import EditorWidgetInterface # Removed to avoid metaclass conflict
from dm_toolkit.gui.i18n import tr
from dm_toolkit.consts import ZONES_EXTENDED, TargetScope

# Implementing the interface methods directly (Duck Typing) or just using a Mixin without ABCMeta
class EditorWidgetMixin:
    def get_value(self):
        raise NotImplementedError
    def set_value(self, value):
        raise NotImplementedError

class ZoneCombo(QComboBox, EditorWidgetMixin):
    def __init__(self, parent=None, allow_empty=False):
        super().__init__(parent)
        if allow_empty:
            self.addItem("---", None)
        for z in ZONES_EXTENDED:
            self.addItem(tr(z), z)

    def get_value(self):
        data = self.currentData()
        return None if data is None else data

    def set_value(self, value):
        if value is None:
            # Select empty item if it exists
            idx = self.findData(None)
            if idx >= 0:
                self.setCurrentIndex(idx)
            return
        
        idx = self.findData(value)
        if idx >= 0:
            self.setCurrentIndex(idx)

class MultiZoneSelector(QWidget, EditorWidgetMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checkboxes = {}
        for z in ZONES_EXTENDED:
            cb = QCheckBox(tr(z))
            self.checkboxes[z] = cb
            layout.addWidget(cb)

    def get_value(self):
        return [z for z, cb in self.checkboxes.items() if cb.isChecked()]

    def set_value(self, value):
        if not isinstance(value, list):
            value = [value] if value else []
        for z, cb in self.checkboxes.items():
            cb.setChecked(z in value)

class ScopeCombo(QComboBox, EditorWidgetMixin):
    def __init__(self, parent=None, include_zones=False, allow_empty=False):
        super().__init__(parent)
        if allow_empty:
            self.addItem("---", None)
        
        scopes = [
            TargetScope.SELF, TargetScope.OPPONENT, "TARGET_SELECT",
            "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"
        ]
        if include_zones:
            scopes += ["BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"]

        for s in scopes:
            self.addItem(tr(s), s)

    def get_value(self):
        data = self.currentData()
        return None if data is None else data

    def set_value(self, value):
        if value is None:
            # Select empty item if it exists
            idx = self.findData(None)
            if idx >= 0:
                self.setCurrentIndex(idx)
            return
        
        idx = self.findData(value)
        if idx >= 0:
            self.setCurrentIndex(idx)

class TextWidget(QLineEdit, EditorWidgetMixin):
    def get_value(self):
        return self.text()

    def set_value(self, value):
        self.setText(str(value) if value is not None else "")

class NumberWidget(QSpinBox, EditorWidgetMixin):
    def __init__(self, parent=None, min_val=-9999, max_val=9999):
        super().__init__(parent)
        self.setRange(min_val, max_val)

    def get_value(self):
        return self.value()

    def set_value(self, value):
        if value is not None:
            self.setValue(int(value))


class QuantityModeWidget(QWidget, EditorWidgetMixin):
    """数量入力とモード（固定/最大/すべて）を統合したウィジェット"""
    def __init__(self, parent=None, max_val: int = 9999):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("ちょうど (Exact)"), "EXACT")
        self.mode_combo.addItem(tr("最大 (Up to)"), "UP_TO")
        self.mode_combo.addItem(tr("すべて (All)"), "ALL")

        self.spin = QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setMaximum(max_val)

        layout.addWidget(self.mode_combo)
        layout.addWidget(self.spin)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData()
        if mode == "ALL":
            self.spin.setEnabled(False)
        else:
            self.spin.setEnabled(True)

    def get_value(self):
        mode = self.mode_combo.currentData()
        val = -1 if mode == "ALL" else self.spin.value()
        return {"amount": val, "up_to": mode == "UP_TO"}

    def set_value(self, value):
        if value is None:
            self.mode_combo.setCurrentIndex(0)
            self.spin.setValue(1)
            return

        if isinstance(value, dict):
            amt = value.get("amount", 1)
            up_to = value.get("up_to", False)
            if str(amt).upper() in ("ALL", "すべて") or amt == -1:
                self.mode_combo.setCurrentIndex(2) # ALL
                self.spin.setValue(1)
            else:
                self.spin.setValue(int(amt))
                if up_to:
                    self.mode_combo.setCurrentIndex(1) # UP_TO
                else:
                    self.mode_combo.setCurrentIndex(0) # EXACT
        else:
            # Fallback for old simple amount
            if str(value).upper() in ("ALL", "すべて") or int(value) == -1:
                self.mode_combo.setCurrentIndex(2) # ALL
                self.spin.setValue(1)
            else:
                self.spin.setValue(int(value))
                self.mode_combo.setCurrentIndex(0) # EXACT

class AmountWithAllWidget(QSpinBox, EditorWidgetMixin):
    """枚数スピンボックス。最小値（-1）を「すべて」として表示する。
    - -1 → "すべて" (特殊値テキスト = QSpinBox.specialValueText)
    -  0 → 0枚
    -  1以上 → 通常枚数
    再発防止: setSpecialValueText はミニマム値にのみ適用される PyQt6 の仕様。
              min_value は必ず -1 に設定すること。
    """
    ALL_VALUE: int = -1

    def __init__(self, parent=None, max_val: int = 9999):
        super().__init__(parent)
        self.setMinimum(self.ALL_VALUE)
        self.setMaximum(max_val)
        self.setSpecialValueText(tr("すべて"))

    def get_value(self) -> int:
        return self.value()  # -1 が "all" を意味する

    def set_value(self, value) -> None:
        if value is None:
            self.setValue(0)
        elif str(value).upper() in ("ALL", "すべて") or int(value) == self.ALL_VALUE:
            self.setValue(self.ALL_VALUE)
        else:
            self.setValue(int(value))

class BoolCheckWidget(QCheckBox, EditorWidgetMixin):
    def get_value(self):
        return self.isChecked()

    def set_value(self, value):
        self.setChecked(bool(value))
