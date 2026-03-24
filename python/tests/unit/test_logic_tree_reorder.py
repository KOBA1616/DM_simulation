# -*- coding: utf-8 -*-
from __future__ import annotations

import pathlib

LOGIC_TREE = pathlib.Path("dm_toolkit/gui/editor/logic_tree.py")
CONTEXT_MENUS = pathlib.Path("dm_toolkit/gui/editor/context_menus.py")


def _read_text(path: pathlib.Path) -> str:
    root = pathlib.Path(__file__).resolve().parents[3]
    target = root / path
    assert target.exists(), f"Target file missing: {target}"
    return target.read_text(encoding="utf-8")


def test_logic_tree_defines_keyboard_reorder_shortcuts() -> None:
    text = _read_text(LOGIC_TREE)

    assert "def move_current_item_up" in text
    assert "def move_current_item_down" in text
    assert "AltModifier" in text
    assert "Key_Up" in text
    assert "Key_Down" in text


def test_context_menu_exposes_reorder_actions() -> None:
    text = _read_text(CONTEXT_MENUS)

    assert "上へ移動 (Alt+↑)" in text
    assert "下へ移動 (Alt+↓)" in text
    assert "move_current_item_up" in text
    assert "move_current_item_down" in text
