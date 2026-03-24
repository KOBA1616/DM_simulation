from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_transition_battle_to_graveyard_contains_expected_parts():
    action = {"from_zone": "BATTLE_ZONE", "to_zone": "GRAVEYARD"}
    res = CardTextGenerator._format_game_action_command(
        "TRANSITION", action, False, 1, 0, "カード", "", "", "", None
    )
    assert "バトルゾーン" in res and "墓地" in res and "置く" in res


def test_move_card_to_hand_contains_expected_parts():
    action = {"to_zone": "HAND"}
    res = CardTextGenerator._format_game_action_command(
        "MOVE_CARD", action, False, 2, 0, "自分のクリーチャー", "", "", "", None
    )
    assert "手札" in res or "手札に" in res


def test_transition_deck_to_hand_includes_explicit_selection_wording():
    action = {"from_zone": "DECK", "to_zone": "HAND"}
    res = CardTextGenerator._format_game_action_command(
        "TRANSITION", action, False, 2, 0, "カード", "枚", "", "", None
    )
    assert "選び" in res and "手札に加える" in res


def test_move_buffer_to_zone_with_filter_includes_explicit_selection_wording():
    action = {
        "to_zone": "HAND",
        "filter": {"civilizations": ["FIRE"], "types": ["CREATURE"]},
    }
    res = CardTextGenerator._format_buffer_command(
        "MOVE_BUFFER_TO_ZONE", action, False, 2
    )
    assert "選び" in res and "手札" in res
