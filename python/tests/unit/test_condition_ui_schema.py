# -*- coding: utf-8 -*-
import pytest

from dm_toolkit.gui.editor.forms.parts.condition_widget import (
    CONDITION_UI_CONFIG, ConditionEditorWidget,
)
from PyQt6 import QtWidgets

# Ensure stubbed QGroupBox has setTitle used in widget constructor
if not hasattr(QtWidgets.QGroupBox, 'setTitle'):
    def _setTitle(self, title):
        setattr(self, '_title', title)
    QtWidgets.QGroupBox.setTitle = _setTitle

# Ensure QScrollArea shim supports methods used by the widget
if not hasattr(QtWidgets.QScrollArea, 'setWidgetResizable'):
    def _setWidgetResizable(self, val):
        setattr(self, '_widget_resizable', bool(val))
    def _setWidget(self, widget):
        setattr(self, '_widget', widget)
    QtWidgets.QScrollArea.setWidgetResizable = _setWidgetResizable
    QtWidgets.QScrollArea.setWidget = _setWidget


def test_condition_types_have_ui_config():
    """各コンディションタイプが CONDITION_UI_CONFIG に定義されていることを検証する"""
    widget = ConditionEditorWidget()
    combo = widget.cond_type_combo
    # Collect all data values from combo
    data_values = [combo.itemData(i) for i in range(combo.count())]

    # Ensure that for every non-custom type, there is a config entry
    for val in data_values:
        if val == "CUSTOM":
            continue
        assert val in CONDITION_UI_CONFIG, f"Missing UI config for condition type: {val}"

    # Basic shape tests for a known type
    sample = CONDITION_UI_CONFIG.get("OPPONENT_DRAW_COUNT")
    assert isinstance(sample, dict)
    assert "show_val" in sample
    assert "label_val" in sample
