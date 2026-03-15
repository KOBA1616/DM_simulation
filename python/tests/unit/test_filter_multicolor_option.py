# -*- coding: utf-8 -*-
from __future__ import annotations

import pathlib

from dm_toolkit.gui.editor.text_resources import CardTextResources


ROOT = pathlib.Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    target = ROOT / rel_path
    assert target.exists(), f"Missing file: {target}"
    return target.read_text(encoding="utf-8")


def test_filter_widget_enables_multicolor_in_civ_selector() -> None:
    text = _read("dm_toolkit/gui/editor/forms/parts/filter_widget.py")
    assert "CivilizationSelector(allow_multicolor=True)" in text


def test_template_dialog_enables_multicolor_in_civ_selector() -> None:
    text = _read("dm_toolkit/gui/editor/template_params_dialog.py")
    assert "CivilizationSelector(allow_multicolor=True)" in text


def test_text_resources_has_multicolor_japanese_label() -> None:
    assert CardTextResources.get_civilization_text("MULTICOLOR") == "多色"
