# -*- coding: utf-8 -*-
"""Contract tests for SUMMON_COUNT_THIS_TURN canonicalization.

Verify that TurnStats declares `summon_count_this_turn` and that the
pipeline executor references it for `SUMMON_COUNT_THIS_TURN`.
"""

from pathlib import Path


def test_turnstats_has_summon_field():
    p = Path("src/core/card_stats.hpp")
    src = p.read_text(encoding="utf-8")
    assert "summon_count_this_turn" in src, "TurnStats に summon_count_this_turn がありません"


def test_pipeline_reads_summon_field():
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    src = p.read_text(encoding="utf-8")
    assert "SUMMON_COUNT_THIS_TURN" in src, "pipeline_executor.cpp に SUMMON_COUNT_THIS_TURN の扱いがありません"
    assert "summon_count_this_turn" in src, "pipeline_executor.cpp に summon_count_this_turn 参照がありません"
