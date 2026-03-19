# -*- coding: utf-8 -*-
"""Contract test: ensure replacement handling (G-Neo) occurs before destroyed stat increment.

This is a lightweight check that the implementation applies replacement logic
(e.g. setting `g_neo_activated`) before incrementing
`creatures_destroyed_this_turn` so replacement prevents counting.
"""
from pathlib import Path


def test_replacement_handling_occurs_before_destroy_increment():
    p = Path("src/engine/infrastructure/commands/definitions/commands.cpp")
    src = p.read_text(encoding="utf-8")
    assert "g_neo_activated" in src, "commands.cpp に g_neo_activated フラグが見つかりません"
    assert "creatures_destroyed_this_turn" in src, "commands.cpp に creatures_destroyed_this_turn の参照が見つかりません"

    # Ensure the g_neo handling appears earlier in the file than the destroyed increment
    # Narrow to the TransitionCommand::execute body, ensure g_neo handling appears
    parts = src.split("TransitionCommand::execute")
    assert len(parts) > 1, "TransitionCommand::execute が見つかりません"
    exec_block = parts[1]
    if "TransitionCommand::invert" in exec_block:
        exec_block = exec_block.split("TransitionCommand::invert")[0]

    idx_replace = exec_block.find("g_neo_activated")
    idx_helper = exec_block.find("add_turn_destroyed_count")
    assert idx_replace >= 0 and idx_helper >= 0, "置換処理または破壊カウント呼び出しが見つかりません"
    assert idx_replace < idx_helper, "置換処理が破壊カウント加算より後に来ています — 置換後に統計加算する設計を守ってください"
