# -*- coding: utf-8 -*-
"""Contract tests for GET_STAT DESTROY_COUNT_THIS_TURN support.

Ensure editor registry exposes the key and pipeline has a handler.
"""

from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_destroy_count():
    assert "DESTROY_COUNT_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "DESTROY_COUNT_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_pipeline_executor_contains_destroy_count_handler():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "DESTROY_COUNT_THIS_TURN" in src, "pipeline_executor.cpp に DESTROY_COUNT_THIS_TURN の扱いがありません"
    assert "creatures_destroyed_this_turn" in src, "pipeline_executor.cpp に creatures_destroyed_this_turn 参照がありません"
