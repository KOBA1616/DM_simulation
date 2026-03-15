from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_apply_modifier_cost_effect():
    action = {"str_param": "COST", "amount": 2}
    res = CardTextGenerator._format_game_action_command(
        "APPLY_MODIFIER", action, False, 0, 0, "クリーチャー", "つ", "", "", None
    )
    assert res == "クリーチャーを2つは、そのクリーチャーにコスト修正（2）を与える。"
