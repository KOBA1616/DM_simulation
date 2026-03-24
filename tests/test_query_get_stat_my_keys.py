# -*- coding: utf-8 -*-
"""
Contract tests for GET_STAT `MY_*` alias support.

Ensures the editor uses `MY_*` keys and that the pipeline executor
contains handling (or alias mapping) for those keys.
"""
from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_my_keys():
    # Sanity: editor expects MY_* style keys available for compare-stat
    assert "MY_SHIELD_COUNT" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "MY_MANA_COUNT" in CardTextResources.COMPARE_STAT_EDITOR_KEYS


def test_pipeline_executor_contains_my_alias_handling():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    # The C++ handler should contain logic to map or handle MY_ aliases
    assert 'rfind("MY_", 0)' in src or 'MY_SHIELD_COUNT' in src, \
        "pipeline_executor.cpp に MY_* の扱いがありません"
