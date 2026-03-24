# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm


class _Mode:
    def __init__(self, mode: str):
        self._mode = mode

    def currentData(self):
        return self._mode


class _LineEdit:
    def text(self):
        return ""


class _FilterWidget:
    def __init__(self, payload):
        self._payload = payload

    def get_data(self):
        return self._payload


def test_static_mode_persists_target_filter_only():
    form = EffectEditForm.__new__(EffectEditForm)
    form.mode_combo = _Mode("STATIC")
    form.current_item = None
    form.layer_str_edit = _LineEdit()
    form.target_filter = _FilterWidget({"zones": ["BATTLE_ZONE"]})

    def _collect(data):
        # Simulate mixed legacy/new keys coming from bindings.
        data["filter"] = {"zones": ["HAND"]}
        data["target_filter"] = {"zones": ["BATTLE_ZONE"]}
        data["trigger_filter"] = {"types": ["CREATURE"]}

    form._collect_bindings = _collect

    out = {}
    form._save_ui_to_data(out)

    assert out.get("target_filter") == {"zones": ["BATTLE_ZONE"]}
    assert "filter" not in out
    assert "trigger_filter" not in out


def test_triggered_mode_persists_trigger_filter_only():
    form = EffectEditForm.__new__(EffectEditForm)
    form.mode_combo = _Mode("TRIGGERED")
    form.current_item = None
    form.layer_str_edit = _LineEdit()
    form.trigger_filter = _FilterWidget({"types": ["SPELL"]})

    def _collect(data):
        # Simulate mixed keys before mode-specific cleanup.
        data["filter"] = {"zones": ["HAND"]}
        data["target_filter"] = {"zones": ["BATTLE_ZONE"]}
        data["trigger_filter"] = {"types": ["CREATURE"]}
        data["trigger"] = "ON_PLAY"

    form._collect_bindings = _collect

    out = {}
    form._save_ui_to_data(out)

    assert out.get("trigger_filter") == {"types": ["SPELL"]}
    assert "target_filter" not in out
    assert "filter" not in out
