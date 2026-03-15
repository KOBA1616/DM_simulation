# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.modifier_form import ModifierEditForm


class _StubCombo:
    def __init__(self, value: str) -> None:
        self._value = value

    def currentData(self) -> str:
        return self._value


class _StubWidget:
    def __init__(self) -> None:
        self.visible = None
        self.text = None
        self.title = None
        self.range = None

    def setVisible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def setText(self, text: str) -> None:
        self.text = text

    def setTitle(self, title: str) -> None:
        self.title = title

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (minimum, maximum)


def test_grant_keyword_shows_value_as_target_count() -> None:
    form = type("DummyForm", (), {})()
    form.type_combo = _StubCombo("GRANT_KEYWORD")
    form.label_value = _StubWidget()
    form.value_spin = _StubWidget()
    form.label_keyword = _StubWidget()
    form.keyword_combo = _StubWidget()
    form.label_restriction = _StubWidget()
    form.restriction_combo = _StubWidget()
    form.filter_widget = _StubWidget()

    ModifierEditForm.update_visibility(form)

    assert form.label_value.visible is True
    assert form.value_spin.visible is True
    assert form.label_value.text == "対象数（0 = すべて）"
    assert form.value_spin.range == (0, 99999)
    assert form.label_keyword.visible is True
    assert form.keyword_combo.visible is True