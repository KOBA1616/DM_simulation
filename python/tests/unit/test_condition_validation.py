# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets

# Provide lightweight shims for stubbed widgets in headless test environment
if not hasattr(QtWidgets.QGroupBox, 'setTitle'):
    def _setTitle(self, title):
        setattr(self, '_title', title)
    QtWidgets.QGroupBox.setTitle = _setTitle

if not hasattr(QtWidgets.QScrollArea, 'setWidgetResizable'):
    def _setWidgetResizable(self, val):
        setattr(self, '_widget_resizable', bool(val))
    def _setWidget(self, widget):
        setattr(self, '_widget', widget)
    QtWidgets.QScrollArea.setWidgetResizable = _setWidgetResizable
    QtWidgets.QScrollArea.setWidget = _setWidget

# Provide findText shim for EnhancedComboBox used in headless stubs
if not hasattr(QtWidgets.QComboBox, 'findText'):
    def _findText(self, text):
        items = getattr(self, '_items', [])
        for i, item in enumerate(items):
            # item[0] is the display label in our stubs
            if str(item[0]) == str(text):
                return i
        return -1
    QtWidgets.QComboBox.findText = _findText

if not hasattr(QtWidgets.QComboBox, 'setCurrentText'):
    def _setCurrentText(self, text):
        # store current text for tests; do not change index
        setattr(self, '_current_text', text)
    QtWidgets.QComboBox.setCurrentText = _setCurrentText

if not hasattr(QtWidgets.QCheckBox, 'setChecked'):
    def _setChecked(self, val):
        setattr(self, '_checked', bool(val))
    QtWidgets.QCheckBox.setChecked = _setChecked

from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget


def test_validate_none_condition_has_no_errors():
    w = ConditionEditorWidget()
    w.set_data({'type': 'NONE'})
    errs = w.validate_condition_model()
    assert errs == []


def test_validate_opponent_draw_count_missing_value_reports_error():
    w = ConditionEditorWidget()
    # OPPONENT_DRAW_COUNT requires 'value'
    errs = w.validate_condition_model({'type': 'OPPONENT_DRAW_COUNT'})
    assert 'missing value' in errs


def test_validate_compare_stat_complete_is_valid():
    w = ConditionEditorWidget()
    data = {
        'type': 'COMPARE_STAT',
        'stat_key': 'MY_SHIELD_COUNT',
        'op': '>=',
        'value': 1
    }
    errs = w.validate_condition_model(data)
    assert errs == []
