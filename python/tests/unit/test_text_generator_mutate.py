from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_mutate_fallback_handler():
    action = {"mutation_kind": "FOO"}
    res = CardTextGenerator._format_game_action_command(
        "MUTATE", action, False, 2, 0, "クリーチャー", "つ", "", "", None
    )
    assert res == "状態変更(FOO): クリーチャー (値:2)"
