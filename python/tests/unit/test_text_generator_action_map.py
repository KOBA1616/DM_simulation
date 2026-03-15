import pytest

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_boost_mana_handler():
    res = CardTextGenerator._format_game_action_command(
        "BOOST_MANA", {}, False, 1, 0, "カード", "つ", "", "", None
    )
    assert res == "自分のマナを1つ増やす。"


def test_shuffle_deck_handler():
    res = CardTextGenerator._format_game_action_command(
        "SHUFFLE_DECK", {}, False, 0, 0, "カード", "", "", "", None
    )
    assert res == "山札をシャッフルする。"


def test_break_shield_handler_defaults_to_opponent():
    # No explicit target_group/scope -> should prefix with 相手の
    res = CardTextGenerator._format_game_action_command(
        "BREAK_SHIELD", {}, False, 2, 0, "", "", "", "", None
    )
    assert res == "相手のシールドを2つブレイクする。"


def test_add_shield_from_deck():
    res = CardTextGenerator._format_game_action_command(
        "ADD_SHIELD", {}, False, 3, 0, "山札", "", "", "", None
    )
    assert res == "山札の上から3枚をシールド化する。"
