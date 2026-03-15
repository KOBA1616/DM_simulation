import pytest

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_draw_card_handler():
    res = CardTextGenerator._format_game_action_command(
        "DRAW_CARD", {}, False, 2, 0, "カード", "枚", "", "", None
    )
    assert res == "2枚引く。"


def test_discard_handler_defaults_to_hand():
    res = CardTextGenerator._format_game_action_command(
        "DISCARD", {}, False, 3, 0, "", "枚", "", "", None
    )
    assert res == "手札を3枚捨てる。"
