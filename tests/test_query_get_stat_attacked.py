from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_attacked_stat_key_present():
    """RED: Editor/resource registry must expose ATTACKED_THIS_TURN key."""
    assert "ATTACKED_THIS_TURN" in CardTextResources.COMPARE_STAT_EDITOR_KEYS
    assert "ATTACKED_THIS_TURN" in CardTextResources.STAT_KEY_MAP
