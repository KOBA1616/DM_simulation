# -*- coding: utf-8 -*-
import inspect

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
def _call_format_command(cmd_type, action, is_spell, look_count=0, add_count=0, target_str="", unit="", input_key="", input_usage="", sample=None):
    from dm_toolkit.gui.editor.text_generator import CardTextGenerator
    action = action.copy() if action else {}
    action["type"] = cmd_type
    if look_count: action["look_count"] = look_count
    if add_count: action["add_count"] = add_count
    if input_key: action["input_value_key"] = input_key
    if input_usage: action["input_usage"] = input_usage
    # For testing, we mock the resolution
    ctx = TextGenerationContext({}, sample)
    ctx.is_spell = is_spell
    return CardTextGenerator._format_command(action, ctx)

def _call_format_buffer(cmd_type, action, is_spell, look_count=0):
    return _call_format_command(cmd_type, action, is_spell, look_count)

def _call_format_mutate(cmd_type, action, is_spell, look_count=0, add_count=0, target_str="", unit=""):
    return _call_format_command(cmd_type, action, is_spell, look_count, add_count, target_str, unit)



def test_game_action_command_legacy_duplicate_branches_removed():
    src = inspect.getsource(CardTextGenerator._format_game_action_command)

    # These commands are now expected to be handled by ACTION_HANDLER_MAP dispatch,
    # not duplicated in legacy if/elif branches.
    removed = [
        'elif atype == "SEARCH_DECK"',
        'elif atype == "LOOK_AND_ADD"',
        'elif atype == "PUT_CREATURE"',
        'elif atype == "SHUFFLE_DECK"',
        'elif atype == "BOOST_MANA"',
        'elif atype == "BREAK_SHIELD"',
        'elif atype == "ADD_SHIELD"',
        'elif atype == "SHIELD_BURN"',
        'elif atype == "REVEAL_CARDS"',
        'elif atype == "COUNT_CARDS"',
        'elif atype == "LOCK_SPELL"',
        'elif atype in ["SPELL_RESTRICTION", "CANNOT_PUT_CREATURE", "CANNOT_SUMMON_CREATURE", "PLAYER_CANNOT_ATTACK"]',
        'elif atype == "STAT"',
        'elif atype == "GET_GAME_STAT"',
        'elif atype == "FLOW"',
        'elif atype == "GAME_RESULT"',
        'elif atype == "DECLARE_NUMBER"',
        'elif atype == "DECIDE"',
        'elif atype == "DECLARE_REACTION"',
        'elif atype == "ATTACH"',
        'elif atype == "MOVE_TO_UNDER_CARD"',
        'elif atype == "RESET_INSTANCE"',
        'elif atype == "SELECT_TARGET"',
    ]

    for token in removed:
        assert token not in src


def test_game_action_command_map_handles_boost_mana_and_search_deck():
    boost = _call_format_command(
        "BOOST_MANA", {}, False, 2, 0, "カード", "つ", "", "", []
    )
    assert boost == "自分のマナを2つ増やす。"

    search = _call_format_command(
        "SEARCH_DECK", {"destination_zone": "HAND"}, False, 1, 0, "カード", "枚", "", "", []
    )
    assert "山札" in search and "手札" in search


def test_game_action_command_map_handles_lock_restriction_and_stat_series():
    lock_text = _call_format_command(
        "LOCK_SPELL",
        {"target_group": "OPPONENT", "duration": "UNTIL_END_OF_TURN"},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "相手" in lock_text and "呪文を唱えられない" in lock_text

    restriction_text = _call_format_command(
        "SPELL_RESTRICTION",
        {"target_group": "OPPONENT", "duration": "UNTIL_END_OF_TURN", "filter": {"exact_cost": 5}},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "相手" in restriction_text and "コスト5の呪文" in restriction_text

    stat_text = _call_format_command(
        "STAT",
        {"stat": "MANA_COUNT", "amount": 2},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "統計更新" in stat_text and "+= 2" in stat_text

    stat_ref_text = _call_format_command(
        "GET_GAME_STAT",
        {"str_val": "MANA_COUNT"},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        ["FIRE", "NATURE"],
    )
    assert "マナ" in stat_ref_text


def test_game_action_command_map_handles_flow_result_and_declare_series():
    flow_text = _call_format_command(
        "FLOW",
        {"flow_type": "PHASE_CHANGE", "value1": 2},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "フェーズへ移行する" in flow_text

    result_text = _call_format_command(
        "GAME_RESULT",
        {"result": "WIN"},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "ゲームを終了する" in result_text

    declare_number_text = _call_format_command(
        "DECLARE_NUMBER",
        {"value1": 3, "value2": 7},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "3～7" in declare_number_text

    decide_text = _call_format_command(
        "DECIDE",
        {"selected_option_index": 1},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "選択肢1" in decide_text

    reaction_text = _call_format_command(
        "DECLARE_REACTION",
        {"reaction_index": 2},
        False,
        0,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "インデックス 2" in reaction_text


def test_game_action_command_map_handles_attach_selection_and_reset_series():
    attach_text = _call_format_command(
        "ATTACH",
        {},
        False,
        0,
        0,
        "このカード",
        "枚",
        "",
        "",
        [],
    )
    assert "カードの下に重ねる" in attach_text

    under_text = _call_format_command(
        "MOVE_TO_UNDER_CARD",
        {},
        False,
        2,
        0,
        "カード",
        "枚",
        "",
        "",
        [],
    )
    assert "2枚カードの下に重ねる" in under_text

    reset_text = _call_format_command(
        "RESET_INSTANCE",
        {},
        False,
        0,
        0,
        "このクリーチャー",
        "枚",
        "",
        "",
        [],
    )
    assert "状態を初期化する" in reset_text

    select_text = _call_format_command(
        "SELECT_TARGET",
        {},
        False,
        2,
        0,
        "クリーチャー",
        "体",
        "",
        "",
        [],
    )
    assert "2体選ぶ" in select_text
