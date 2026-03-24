# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.forms.card_form import CardEditForm


class _FakeGetWidget:
    def __init__(self, value):
        self._value = value

    def get_value(self):
        return self._value


def test_save_ui_persists_cost_reductions_from_dedicated_widget():
    form = object.__new__(CardEditForm)
    form.widgets_map = {
        "name": _FakeGetWidget("Card X"),
        "type": _FakeGetWidget("CREATURE"),
        "cost_reductions": _FakeGetWidget([
            {"id": "cr1", "type": "PASSIVE", "amount": 2}
        ]),
        "keywords": _FakeGetWidget({}),
    }

    data = {"id": 100, "name": "Old", "twinpact": True}
    form._save_ui_to_data(data)

    # 再発防止: 専用ウィジェット値が保存データへ反映されないと、
    # CardEditForm で編集した cost_reductions が消失する。
    assert data["cost_reductions"] == [{"id": "cr1", "type": "PASSIVE", "amount": 2}]
    assert data["name"] == "Card X"
    assert "twinpact" not in data
