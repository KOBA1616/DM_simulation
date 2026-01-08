from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_play_from_zone_conversion_with_cost_and_source():
    act = {
        'type': 'PLAY_FROM_ZONE',
        'source_zone': 'GRAVEYARD',
        'destination_zone': 'BATTLE_ZONE',
        'value1': 3,
        'scope': 'PLAYER_SELF'
    }

    cmd = ActionConverter.convert(act)

    assert cmd['type'] == 'PLAY_FROM_ZONE'
    assert cmd.get('from_zone') == 'GRAVEYARD'
    # Zone names are normalized to canonical forms (BATTLE_ZONE -> BATTLE, etc)
    assert cmd.get('to_zone') in ('BATTLE_ZONE', 'BATTLE')
    assert cmd.get('max_cost') == 3
    assert cmd.get('target_group') == 'PLAYER_SELF'


def test_look_and_add_conversion_basic():
    act = {
        'type': 'LOOK_AND_ADD',
        'value1': 3,
        'value2': 1,
        'scope': 'PLAYER_SELF',
        'filter': {'color': 'LIGHT'}
    }

    cmd = ActionConverter.convert(act)

    assert cmd['type'] == 'LOOK_AND_ADD'
    assert cmd.get('look_count') == 3
    assert cmd.get('add_count') == 1
    assert cmd.get('target_group') == 'PLAYER_SELF'
    assert 'target_filter' in cmd
    assert cmd.get('legacy_warning') is False
