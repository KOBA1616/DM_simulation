# -*- coding: utf-8 -*-
"""Contract tests for SPELL_CAST_THIS_TURN stat key support.

Checks editor registry, query schema, pipeline stat resolver, and compare-stat evaluator.
"""

from pathlib import Path

from dm_toolkit.consts import QUERY_MODES
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_registry_includes_spell_cast_stat_key() -> None:
    assert "SPELL_CAST_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "SPELL_CAST_THIS_TURN" in CardTextResources.STAT_KEY_MAP


def test_query_modes_include_spell_cast_stat_key() -> None:
    assert "SPELL_CAST_THIS_TURN" in QUERY_MODES


def test_pipeline_executor_contains_spell_cast_handler() -> None:
    src = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp").read_text(encoding="utf-8")
    assert "SPELL_CAST_THIS_TURN" in src
    assert "spells_cast_this_turn" in src


def test_compare_stat_evaluator_contains_spell_cast_handler() -> None:
    src = Path("src/engine/systems/rules/condition_system.cpp").read_text(encoding="utf-8")
    assert "SPELL_CAST_THIS_TURN" in src
    assert "spells_cast_this_turn" in src
