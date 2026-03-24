# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.forms.card_form import CardEditForm


class _FakeWidget:
    def __init__(self):
        self.value = None

    def set_value(self, value):
        self.value = value

    def blockSignals(self, _blocked):
        return False


def test_load_ui_resets_missing_fields_when_switching_cards():
    form = object.__new__(CardEditForm)
    form.widgets_map = {
        "name": _FakeWidget(),
        "cost": _FakeWidget(),
        "type": _FakeWidget(),
        "races": _FakeWidget(),
        "evolution_condition": _FakeWidget(),
    }
    form._update_visibility = lambda: None

    form._load_ui_from_data(
        {
            "name": "Card A",
            "cost": 5,
            "type": "EVOLUTION_CREATURE",
            "races": ["ドラゴン"],
            "evolution_condition": "ファイアー・バード",
        },
        None,
    )

    form._load_ui_from_data(
        {
            "name": "Card B",
            "cost": 3,
            "type": "CREATURE",
        },
        None,
    )

    assert form.widgets_map["name"].value == "Card B"
    assert form.widgets_map["cost"].value == 3
    assert form.widgets_map["type"].value == "CREATURE"
    assert form.widgets_map["races"].value == []
    assert form.widgets_map["evolution_condition"].value == ""