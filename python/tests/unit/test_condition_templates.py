# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets

# Headless shims used by ConditionEditorWidget tests
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

from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget, CONDITION_TEMPLATES


def test_condition_template_ui_has_expected_presets():
    w = ConditionEditorWidget()
    assert hasattr(w, 'template_combo')

    expected = {
        'SELF_SHIELD_3_OR_LESS',
        'OPPONENT_DRAW_2_PLUS',
        'DURING_YOUR_TURN',
        'MANA_CIV_2_PLUS',
    }
    assert expected.issubset(set(CONDITION_TEMPLATES.keys()))


def test_apply_template_by_key_reflects_immediately():
    w = ConditionEditorWidget()

    applied = w.apply_template_by_key('MANA_CIV_2_PLUS')
    assert applied is True

    data = w.get_data()
    assert data.get('type') == 'MANA_CIVILIZATION_COUNT'
    assert data.get('op') == '>='
    assert data.get('value') == 2

    # Template should also update preview text path (non-empty or no-preview fallback).
    preview = w.get_preview_text()
    assert preview is not None
