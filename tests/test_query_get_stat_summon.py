# -*- coding: utf-8 -*-
"""Contract tests for GET_STAT SUMMON_COUNT_THIS_TURN support.

Ensure editor registry exposes the key and pipeline has a handler.
"""

from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_summon_count():
    assert "SUMMON_COUNT_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "SUMMON_COUNT_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_pipeline_executor_contains_summon_count_handler():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "SUMMON_COUNT_THIS_TURN" in src, "pipeline_executor.cpp に SUMMON_COUNT_THIS_TURN の扱いがありません"
    assert "summon_count_this_turn" in src, "pipeline_executor.cpp に summon_count_this_turn 参照がありません"
# -*- coding: utf-8 -*-
"""Contract tests for GET_STAT SUMMON_COUNT_THIS_TURN support.

These tests act as a lightweight RED-suite for the engine/registry:
- Ensure editor registry exposes the key
- Ensure engine pipeline source contains a handler for the key
"""

from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_summon_count():
    assert "SUMMON_COUNT_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "SUMMON_COUNT_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_pipeline_executor_contains_summon_count_handler():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "SUMMON_COUNT_THIS_TURN" in src, "pipeline_executor.cpp に SUMMON_COUNT_THIS_TURN の扱いがありません"
    assert "summon_count_this_turn" in src, "pipeline_executor.cpp に summon_count_this_turn 参照がありません"
