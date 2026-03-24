# -*- coding: utf-8 -*-
"""Contract tests for MANA_SET_THIS_TURN support.

RED: ensure TurnStats has the field and editor/pipeline expose the key.
"""
from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_turnstats_has_mana_set_field():
    p = Path("src/core/card_stats.hpp")
    src = p.read_text(encoding="utf-8")
    assert "mana_set_this_turn" in src, "src/core/card_stats.hpp に mana_set_this_turn が見つかりません"


def test_editor_registry_includes_mana_set():
    assert "MANA_SET_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "MANA_SET_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_pipeline_executor_contains_mana_set_handler():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "MANA_SET_THIS_TURN" in src, "pipeline_executor.cpp に MANA_SET_THIS_TURN の扱いがありません"
    assert "mana_set_this_turn" in src, "pipeline_executor.cpp に mana_set_this_turn 参照がありません"
