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



def test_summon_token_with_name():
    action = {"str_val": "ドラゴン"}
    res = _call_format_command(
        "SUMMON_TOKEN", action, False, 2, 0, "クリーチャー", "体", "", "", None
    )
    assert res == "ドラゴンを2体出す。"


def test_summon_token_generic_fallback():
    action = {"str_val": "GENERIC_TOKEN"}
    res = _call_format_command(
        "SUMMON_TOKEN", action, False, 0, 0, "クリーチャー", "体", "", "", None
    )
    assert res == "トークンを1体出す。"
