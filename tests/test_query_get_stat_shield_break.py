# -*- coding: utf-8 -*-
"""Contract tests for GET_STAT SHIELD_BREAK_* support.

Ensure editor registry exposes the shield-break keys and pipeline has handlers.
"""

from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_shield_break_keys():
    assert "SHIELD_BREAK_ATTEMPT_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "SHIELD_BREAK_ATTEMPT_THIS_TURN" in CardTextResources.STAT_KEY_MAP
    assert "SHIELD_BREAK_RESOLVED_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "SHIELD_BREAK_RESOLVED_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_pipeline_executor_contains_shield_break_handlers():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "SHIELD_BREAK_ATTEMPT_THIS_TURN" in src, "pipeline_executor.cpp に SHIELD_BREAK_ATTEMPT_THIS_TURN の扱いがありません"
    assert "shield_break_attempt_count_this_turn" in src, "pipeline_executor.cpp に shield_break_attempt_count_this_turn 参照がありません"
    assert "SHIELD_BREAK_RESOLVED_THIS_TURN" in src, "pipeline_executor.cpp に SHIELD_BREAK_RESOLVED_THIS_TURN の扱いがありません"
    assert "shield_break_resolved_count_this_turn" in src, "pipeline_executor.cpp に shield_break_resolved_count_this_turn 参照がありません"
