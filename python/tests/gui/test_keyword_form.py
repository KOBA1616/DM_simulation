# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt

from dm_toolkit.gui.editor.forms.keyword_form import KeywordEditForm


class _DummySignal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def emit(self, command: str, payload: dict[str, Any]) -> None:
        self.calls.append((command, payload))


def test_parse_races_normalizes_japanese_separator() -> None:
    races: list[str] = KeywordEditForm._parse_races("Dragon、 Cyber Lord,  ")
    assert races == ["Dragon", "Cyber Lord"]


def test_toggle_rev_change_emits_race_payload() -> None:
    form = KeywordEditForm()
    signal = _DummySignal()
    form.structure_update_requested = signal  # type: ignore[assignment]

    form.toggle_rev_change(True)

    assert signal.calls
    cmd, payload = signal.calls[-1]
    assert cmd == "ADD_REV_CHANGE"
    assert payload == {}


def test_save_ui_to_data_persists_special_keyword_conditions() -> None:
    form = KeywordEditForm()

    # Drive state with explicit predicates because test GUI stubs do not keep widget state.
    form.rev_change_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.mekraid_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.friend_burst_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.mega_last_burst_check.isChecked = lambda: False  # type: ignore[method-assign]

    form.mk_race_edit.text = lambda: "Fire Bird"  # type: ignore[method-assign]
    form.fb_race_edit.text = lambda: "Cyber Lord, Liquid People"  # type: ignore[method-assign]

    data: dict[str, Any] = {}
    form._save_ui_to_data(data)

    # Revolution Change is now node-driven and should not be persisted from keyword form.
    assert "revolution_change" not in data
    assert data["mekraid"] is True
    assert data["friend_burst"] is True
    assert "revolution_change_condition" not in data
    assert data["mekraid_condition"] == {"races": ["Fire Bird"]}
    assert data["friend_burst_condition"] == {"races": ["Cyber Lord", "Liquid People"]}


class _FakeTreeItem:
    def __init__(self, role: str, payload: dict[str, Any] | None = None, parent: "_FakeTreeItem" | None = None) -> None:
        self._role = role
        self._payload = payload or {}
        self._parent = parent
        self._children: list[_FakeTreeItem] = []

    def data(self, role: int) -> Any:
        if role == Qt.ItemDataRole.UserRole + 1:
            return self._role
        if role == Qt.ItemDataRole.UserRole + 2:
            return self._payload
        return None

    def parent(self) -> "_FakeTreeItem" | None:
        return self._parent

    def rowCount(self) -> int:
        return len(self._children)

    def child(self, index: int) -> "_FakeTreeItem":
        return self._children[index]

    def add_child(self, child: "_FakeTreeItem") -> None:
        self._children.append(child)


class _FakeCheckBox:
    def __init__(self) -> None:
        self._checked = False

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def isChecked(self) -> bool:
        return self._checked

    def blockSignals(self, _block: bool) -> None:
        return None


def test_load_ui_restores_rev_change_checkbox_from_effect_node() -> None:
    form = KeywordEditForm()
    form.keyword_checks = {}
    form.rev_change_check = _FakeCheckBox()  # type: ignore[assignment]
    form.mekraid_check = _FakeCheckBox()  # type: ignore[assignment]
    form.friend_burst_check = _FakeCheckBox()  # type: ignore[assignment]
    form.mega_last_burst_check = _FakeCheckBox()  # type: ignore[assignment]

    class _FakeLineEdit:
        def setText(self, _text: str) -> None:
            return None

        def setVisible(self, _visible: bool) -> None:
            return None

    class _FakeLabel:
        def setVisible(self, _visible: bool) -> None:
            return None

    form.mk_race_edit = _FakeLineEdit()  # type: ignore[assignment]
    form.fb_race_edit = _FakeLineEdit()  # type: ignore[assignment]
    form.mk_race_label = _FakeLabel()  # type: ignore[assignment]
    form.fb_race_label = _FakeLabel()  # type: ignore[assignment]

    card = _FakeTreeItem("CARD")
    keywords = _FakeTreeItem("KEYWORDS", payload={}, parent=card)
    effect = _FakeTreeItem(
        "EFFECT",
        payload={"commands": [{"type": "REVOLUTION_CHANGE", "target_filter": {"races": ["Dragon"]}}]},
        parent=card,
    )
    card.add_child(effect)
    card.add_child(keywords)

    form._load_ui_from_data({}, keywords)

    assert form.rev_change_check.isChecked() is True
