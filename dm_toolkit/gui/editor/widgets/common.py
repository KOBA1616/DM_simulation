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
    def __init__(self, parent=None):
        super().__init__(parent)
        for z in ZONES_EXTENDED:
            self.addItem(tr(z), z)

    def get_value(self):
        return self.currentData()

    def set_value(self, value):
        idx = self.findData(value)
        if idx >= 0:
            self.setCurrentIndex(idx)

class ScopeCombo(QComboBox, EditorWidgetMixin):
    def __init__(self, parent=None, include_zones=False):
        super().__init__(parent)
        scopes = [
            TargetScope.SELF, TargetScope.OPPONENT, "TARGET_SELECT",
            "ALL_PLAYERS", "RANDOM", "ALL_FILTERED", "NONE"
        ]
        if include_zones:
            scopes += ["BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"]

        for s in scopes:
            self.addItem(tr(s), s)

    def get_value(self):
        return self.currentData()

    def set_value(self, value):
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

class BoolCheckWidget(QCheckBox, EditorWidgetMixin):
    def get_value(self):
        return self.isChecked()

    def set_value(self, value):
        self.setChecked(bool(value))
