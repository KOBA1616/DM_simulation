from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_editor_exposes_my_and_opponent_attacked_keys():
    assert "MY_ATTACKED_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "OPPONENT_ATTACKED_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "MY_ATTACKED_THIS_TURN" in CardTextResources.STAT_KEY_MAP
    assert "OPPONENT_ATTACKED_THIS_TURN" in CardTextResources.STAT_KEY_MAP
