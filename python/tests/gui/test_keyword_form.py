# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

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

    # Regression guard: race text must propagate to template payload.
    form.rc_race_edit.text = lambda: "Dragon, Cyber Lord"  # type: ignore[method-assign]

    form.toggle_rev_change(True)

    assert signal.calls
    cmd, payload = signal.calls[-1]
    assert cmd == "ADD_REV_CHANGE"
    assert payload == {"races": ["Dragon", "Cyber Lord"]}


def test_save_ui_to_data_persists_special_keyword_conditions() -> None:
    form = KeywordEditForm()

    # Drive state with explicit predicates because test GUI stubs do not keep widget state.
    form.rev_change_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.mekraid_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.friend_burst_check.isChecked = lambda: True  # type: ignore[method-assign]
    form.mega_last_burst_check.isChecked = lambda: False  # type: ignore[method-assign]

    form.rc_race_edit.text = lambda: "Dragon"  # type: ignore[method-assign]
    form.mk_race_edit.text = lambda: "Fire Bird"  # type: ignore[method-assign]
    form.fb_race_edit.text = lambda: "Cyber Lord, Liquid People"  # type: ignore[method-assign]

    data: dict[str, Any] = {}
    form._save_ui_to_data(data)

    assert data["revolution_change"] is True
    assert data["mekraid"] is True
    assert data["friend_burst"] is True
    assert data["revolution_change_condition"] == {"races": ["Dragon"]}
    assert data["mekraid_condition"] == {"races": ["Fire Bird"]}
    assert data["friend_burst_condition"] == {"races": ["Cyber Lord", "Liquid People"]}
