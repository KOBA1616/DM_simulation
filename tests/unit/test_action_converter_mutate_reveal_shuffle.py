from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_mutate_tap_untap_conversion():
    for sval in ("TAP", "UNTAP"):
        act = {'type': 'MUTATE', 'str_val': sval, 'scope': 'PLAYER_SELF'}
        cmd = ActionConverter.convert(act)
        assert cmd['type'] == sval
        assert cmd.get('target_group') == 'PLAYER_SELF'


def test_mutate_shield_burn_conversion():
    act = {'type': 'MUTATE', 'str_val': 'SHIELD_BURN', 'value1': 1, 'scope': 'PLAYER_OPPONENT'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'SHIELD_BURN'
    assert cmd.get('amount') == 1
    assert cmd.get('target_group') == 'PLAYER_OPPONENT'


def test_reveal_and_shuffle_conversion():
    act_reveal = {'type': 'REVEAL_CARDS', 'value1': 2, 'scope': 'PLAYER_SELF'}
    cmdr = ActionConverter.convert(act_reveal)
    assert cmdr['type'] == 'REVEAL_CARDS'
    assert cmdr.get('amount') == 2

    act_shuffle = {'type': 'SHUFFLE_DECK', 'scope': 'PLAYER_SELF'}
    cmds = ActionConverter.convert(act_shuffle)
    assert cmds['type'] == 'SHUFFLE_DECK'
