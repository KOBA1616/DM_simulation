# -*- coding: utf-8 -*-
"""Contract test: ensure a common helper for turn-stat increments exists and is used.

This test checks that `add_turn_destroyed_count` is defined in
`commands.cpp` and that `TransitionCommand::execute` delegates to it
instead of directly mutating `turn_stats.creatures_destroyed_this_turn`.
"""
from pathlib import Path


def test_destroy_increment_helper_exists_and_used():
    p = Path("src/engine/infrastructure/commands/definitions/commands.cpp")
    src = p.read_text(encoding="utf-8")
    assert "add_turn_destroyed_count" in src, "commands.cpp に add_turn_destroyed_count が見つかりません"

    # Ensure helper is called rather than direct field mutation
    # (we look for a call site add_turn_destroyed_count(state, ) and ensure
    # there is no remaining direct ".creatures_destroyed_this_turn +=" in the
    # transition handling block.)
    assert "add_turn_destroyed_count(state" in src, "add_turn_destroyed_count の呼び出しが見つかりません"
    # It's acceptable if the field is referenced elsewhere; ensure no direct ++ in transition
    parts = src.split("TransitionCommand::execute")
    assert len(parts) > 1, "TransitionCommand::execute が見つかりません"
    transition_block = parts[1]
    # Trim at the next method (invert) to limit to the execute body
    if "TransitionCommand::invert" in transition_block:
        transition_block = transition_block.split("TransitionCommand::invert")[0]

    assert "add_turn_destroyed_count(state" in transition_block, "TransitionCommand 内で add_turn_destroyed_count の呼び出しが見つかりません"
    assert ".creatures_destroyed_this_turn" not in transition_block, "TransitionCommand 内で直接 creatures_destroyed_this_turn に加算があります"
