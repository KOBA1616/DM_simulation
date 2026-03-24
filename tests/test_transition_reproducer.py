# Focused reproducer for SELECT -> TRANSITION propagation
from __future__ import annotations
import pytest
from typing import Any

from tests.test_card1_hand_quality import _setup, _cmd, _all_cmds, _instance_ids_in_zone


def test_reproducer_transition_selection_logging() -> None:
    game, db = _setup(1, 2, seed=20)
    s = game.state

    hand_before_ids = set(_instance_ids_in_zone(s, 0, "hand"))

    play_cmd = _cmd(game, db, "PLAY")
    if play_cmd is None:
        pytest.skip("PLAY コマンドなし")
    game.resolve_command(play_cmd)

    # record hand after play
    hand_after_play_ids = set(_instance_ids_in_zone(s, 0, "hand"))
    old_hand_ids = hand_before_ids & hand_after_play_ids

    resolve_cmd = _cmd(game, db, "RESOLVE")
    if resolve_cmd is None:
        pytest.skip("RESOLVE_EFFECT コマンドなし")
    game.resolve_command(resolve_cmd)

    # Show legal commands after resolve (for debugging)
    legal_pre = _all_cmds(game, db)
    print("LEGAL_BEFORE_SELECT:", [str(getattr(c, 'type', '?')) for c in legal_pre])

    sel_num = _cmd(game, db, "SELECT_NUMBER")
    if sel_num is not None:
        choose_1 = next(
            (
                c for c in _all_cmds(game, db)
                if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
                and getattr(c, "target_instance", -1) == 1
            ),
            sel_num,
        )
        game.resolve_command(choose_1)

    legal_mid = _all_cmds(game, db)
    print("LEGAL_AFTER_SELECT_NUMBER:", [str(getattr(c, 'type', '?')) for c in legal_mid])

    # perform select target if present
    chosen_id = None
    for _ in range(5):
        legal = _all_cmds(game, db)
        target_cmds = [
            c for c in legal
            if "SELECT_TARGET" in str(getattr(c, "type", "")).upper()
        ]
        if not target_cmds:
            break
        best = next(
            (c for c in target_cmds if getattr(c, "instance_id", None) in old_hand_ids),
            target_cmds[0],
        )
        chosen_id = getattr(best, "instance_id", None)
        print("RESOLVING_SELECT_TARGET chosen_id=", chosen_id)
        game.resolve_command(best)
        if not s.waiting_for_user_input:
            break

    deck_ids = set(_instance_ids_in_zone(s, 0, "deck"))
    print("DECK_IDS_HEAD:", list(deck_ids)[:10])
    assert chosen_id in deck_ids, f"選択したカード (instance_id={chosen_id}) が山札に見つかりません。"


def test_reproducer_selected_card_leaves_hand_after_transition() -> None:
    game, db = _setup(1, 2, seed=21)
    s = game.state

    hand_before_ids = set(_instance_ids_in_zone(s, 0, "hand"))

    play_cmd = _cmd(game, db, "PLAY")
    if play_cmd is None:
        pytest.skip("PLAY コマンドなし")
    game.resolve_command(play_cmd)

    hand_after_play_ids = set(_instance_ids_in_zone(s, 0, "hand"))
    old_hand_ids = hand_before_ids & hand_after_play_ids

    resolve_cmd = _cmd(game, db, "RESOLVE")
    if resolve_cmd is None:
        pytest.skip("RESOLVE_EFFECT コマンドなし")
    game.resolve_command(resolve_cmd)

    sel_num = _cmd(game, db, "SELECT_NUMBER")
    if sel_num is not None:
        choose_1 = next(
            (
                c
                for c in _all_cmds(game, db)
                if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
                and getattr(c, "target_instance", -1) == 1
            ),
            sel_num,
        )
        game.resolve_command(choose_1)

    chosen_id = None
    for _ in range(5):
        legal = _all_cmds(game, db)
        target_cmds = [
            c for c in legal if "SELECT_TARGET" in str(getattr(c, "type", "")).upper()
        ]
        if not target_cmds:
            break
        best = next(
            (c for c in target_cmds if getattr(c, "instance_id", None) in old_hand_ids),
            target_cmds[0],
        )
        chosen_id = getattr(best, "instance_id", None)
        game.resolve_command(best)
        if not s.waiting_for_user_input:
            break

    deck_ids = set(_instance_ids_in_zone(s, 0, "deck"))
    hand_final_ids = set(_instance_ids_in_zone(s, 0, "hand"))

    # 再発防止: SELECT→TRANSITION の回帰で、選択カードが deck へ移動したのに
    # hand に残留する不整合が再発しないことを固定化する。
    assert chosen_id in deck_ids
    assert chosen_id not in hand_final_ids, (
        f"選択カード(instance_id={chosen_id})が hand に残留しています"
    )
