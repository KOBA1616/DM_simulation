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



def test_transition_battle_to_graveyard_contains_expected_parts():
    action = {"from_zone": "BATTLE_ZONE", "to_zone": "GRAVEYARD"}
    res = _call_format_command(
        "TRANSITION", action, False, 1, 0, "カード", "", "", "", None
    )
    assert "バトルゾーン" in res and "墓地" in res and "置く" in res


def test_move_card_to_hand_contains_expected_parts():
    action = {"to_zone": "HAND"}
    res = _call_format_command(
        "MOVE_CARD", action, False, 2, 0, "自分のクリーチャー", "", "", "", None
    )
    assert "手札" in res or "手札に" in res


def test_transition_deck_to_hand_includes_explicit_selection_wording():
    action = {"from_zone": "DECK", "to_zone": "HAND"}
    res = _call_format_command(
        "TRANSITION", action, False, 2, 0, "カード", "枚", "", "", None
    )
    assert "選び" in res and "手札に加える" in res


def test_move_buffer_to_zone_with_filter_includes_explicit_selection_wording():
    action = {
        "to_zone": "HAND",
        "filter": {"civilizations": ["FIRE"], "types": ["CREATURE"]},
    }
    res = _call_format_buffer(
        "MOVE_BUFFER_TO_ZONE", action, False, 2
    )
    assert "選び" in res and "手札" in res
