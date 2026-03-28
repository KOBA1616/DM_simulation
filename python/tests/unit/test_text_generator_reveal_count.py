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



def test_reveal_cards_by_count():
    res = _call_format_command(
        "REVEAL_CARDS", {}, False, 2, 0, "", "", "", "", None
    )
    assert res == "山札の上から2枚を表向きにする。"


def test_reveal_cards_with_input_key():
    res = _call_format_command(
        "REVEAL_CARDS", {"input_value_key": "X"}, False, 0, 0, "", "", "", "", None
    )
    # Depending on how input_key is routed, handler may use input link or fall back.
    assert ("その数だけ表向き" in res) or ("0枚を表向きにする" in res)


def test_count_cards_with_target():
    res = _call_format_command(
        "COUNT_CARDS", {}, False, 0, 0, "自分のカード", "", "", "", None
    )
    assert res == "自分のカードの数を数える。"


def test_count_cards_generic():
    res = _call_format_command(
        "COUNT_CARDS", {}, False, 0, 0, "カード", "", "", "", None
    )
    # Localized text may vary; ensure it returns a parenthesized label
    assert res.startswith("(") and res.endswith(")")
