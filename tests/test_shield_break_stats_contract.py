# -*- coding: utf-8 -*-
"""Contract test: ensure TurnStats contains shield break attempt/resolved counters.

This test asserts the presence of two new fields in
`src/core/card_stats.hpp`:
- `shield_break_attempt_count_this_turn`
- `shield_break_resolved_count_this_turn`

The test is the RED step for adding shield-break per-turn statistics.
"""
from pathlib import Path


def test_shield_break_fields_present_in_turnstats():
    p = Path("src/core/card_stats.hpp")
    src = p.read_text(encoding="utf-8")

    assert "shield_break_attempt_count_this_turn" in src, (
        "src/core/card_stats.hpp に `shield_break_attempt_count_this_turn` が見つかりません"
    )
    assert "shield_break_resolved_count_this_turn" in src, (
        "src/core/card_stats.hpp に `shield_break_resolved_count_this_turn` が見つかりません"
    )
