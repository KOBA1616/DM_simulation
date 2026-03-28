import pytest

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



def test_boost_mana_handler():
    res = _call_format_command(
        "BOOST_MANA", {}, False, 1, 0, "カード", "つ", "", "", None
    )
    assert res == "自分のマナを1つ増やす。"


def test_shuffle_deck_handler():
    res = _call_format_command(
        "SHUFFLE_DECK", {}, False, 0, 0, "カード", "", "", "", None
    )
    assert res == "山札をシャッフルする。"


def test_break_shield_handler_defaults_to_opponent():
    # No explicit target_group/scope -> should prefix with 相手の
    res = _call_format_command(
        "BREAK_SHIELD", {}, False, 2, 0, "", "", "", "", None
    )
    assert res == "相手のシールドを2つブレイクする。"


def test_add_shield_from_deck():
    res = _call_format_command(
        "ADD_SHIELD", {}, False, 3, 0, "山札", "", "", "", None
    )
    assert res == "山札の上から3枚をシールド化する。"
