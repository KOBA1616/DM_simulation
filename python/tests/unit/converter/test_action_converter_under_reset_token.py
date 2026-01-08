from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_move_to_under_card_conversion():
    act = {
        'type': 'MOVE_TO_UNDER_CARD',
        'base_target': 'Card123',
        'value1': 1,
        'scope': 'PLAYER_SELF'
    }

    cmd = ActionConverter.convert(act)

    # MOVE_TO_UNDER_CARD maps to TRANSITION in unified map_action
    assert cmd['type'] in ('ATTACH', 'TRANSITION')
    assert cmd.get('base_target') == 'Card123'
    assert cmd.get('amount') == 1
    assert cmd.get('target_group') == 'PLAYER_SELF'


def test_reset_instance_conversion():
    act = {'type': 'RESET_INSTANCE', 'scope': 'PLAYER_SELF'}
    cmd = ActionConverter.convert(act)
    # RESET_INSTANCE consolidates to MUTATE with mutation_kind='RESET_INSTANCE'
    assert cmd['type'] in ('RESET_INSTANCE', 'MUTATE')
    if cmd['type'] == 'MUTATE':
        assert cmd.get('mutation_kind') == 'RESET_INSTANCE'
    assert cmd.get('target_group') == 'PLAYER_SELF'


def test_summon_token_conversion():
    act = {'type': 'SUMMON_TOKEN', 'value1': 2, 'str_val': 'TokenA', 'scope': 'PLAYER_SELF'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'SUMMON_TOKEN'
    assert cmd.get('amount') == 2
    assert cmd.get('token_id') == 'TokenA'
    assert cmd.get('target_group') == 'PLAYER_SELF'
