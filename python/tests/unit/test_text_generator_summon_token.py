from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_summon_token_with_name():
    action = {"str_val": "ドラゴン"}
    res = CardTextGenerator._format_game_action_command(
        "SUMMON_TOKEN", action, False, 2, 0, "クリーチャー", "体", "", "", None
    )
    assert res == "ドラゴンを2体出す。"


def test_summon_token_generic_fallback():
    action = {"str_val": "GENERIC_TOKEN"}
    res = CardTextGenerator._format_game_action_command(
        "SUMMON_TOKEN", action, False, 0, 0, "クリーチャー", "体", "", "", None
    )
    assert res == "トークンを1体出す。"
