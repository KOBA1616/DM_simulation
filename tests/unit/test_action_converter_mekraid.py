from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_mekraid_basic_conversion():
    act = {
        'type': 'MEKRAID',
        'value1': 4,  # max cost allowed
        'scope': 'PLAYER_SELF',
        'filter': {'card_type': 'CREATURE'}
    }

    cmd = ActionConverter.convert(act)

    assert cmd['type'] == 'MEKRAID'
    assert cmd.get('look_count') == 3
    assert cmd.get('max_cost') == 4
    assert cmd.get('select_count') == 1
    assert cmd.get('play_for_free') is True
    assert cmd.get('target_group') == 'PLAYER_SELF'
    assert 'target_filter' in cmd
    assert cmd.get('rest_zone') == 'DECK_BOTTOM'
    assert cmd.get('legacy_warning') is False
