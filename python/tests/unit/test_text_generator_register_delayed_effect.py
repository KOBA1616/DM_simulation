from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_register_delayed_effect_fallback():
    action = {"str_val": "UNKNOWN_DELAY"}
    res = CardTextGenerator._format_game_action_command(
        "REGISTER_DELAYED_EFFECT", action, False, 3, 0, "", "", "", "", None
    )
    assert res == "遅延効果（UNKNOWN_DELAY）を3ターン登録する。"
