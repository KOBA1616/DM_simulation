from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_friend_burst_with_str_val():
    action = {"str_val": "ドラゴン"}
    res = CardTextGenerator._format_game_action_command(
        "FRIEND_BURST", action, False, 0, 0, "クリーチャー", "体", "", "", None
    )
    assert res.startswith("＜ドラゴン＞のフレンド・バースト")


def test_friend_burst_fallback_race_from_filter():
    action = {"filter": {"races": ["ビースト"]}}
    res = CardTextGenerator._format_game_action_command(
        "FRIEND_BURST", action, False, 0, 0, "クリーチャー", "体", "", "", None
    )
    assert res.startswith("＜ビースト＞のフレンド・バースト")
