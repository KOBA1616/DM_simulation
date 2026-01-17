from dm_toolkit.action_to_command import map_action


def test_convert_move_card_return_to_hand():
    act = {
        "type": "MOVE_CARD",
        "destination_zone": "HAND",
        "source_zone": "BATTLE_ZONE",
    }

    cmd = map_action(act)
    assert isinstance(cmd, dict)
    assert cmd.get('type') == 'RETURN_TO_HAND'
    assert 'uid' in cmd


def test_map_action_produces_uid():
    cmd = map_action({"type": "DRAW_CARD", "value1": 1})
    assert isinstance(cmd, dict)
    assert cmd.get('type') in ("DRAW_CARD", "NONE")
    assert 'uid' in cmd
