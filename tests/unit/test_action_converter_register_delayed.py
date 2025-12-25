import pytest
from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_register_delayed_effect_conversion():
    act = {
        'type': 'REGISTER_DELAYED_EFFECT',
        'str_val': 'SampleEffect',
        'value1': 2,
        'scope': 'PLAYER_SELF'
    }

    cmd = ActionConverter.convert(act)

    assert cmd['type'] == 'REGISTER_DELAYED_EFFECT'
    assert cmd.get('str_val') == 'SampleEffect'
    assert cmd.get('value1') == 2
    assert cmd.get('target_group') == 'PLAYER_SELF'
    assert cmd.get('legacy_warning') is False


def test_revolution_change_conversion():
    act = {
        'type': 'REVOLUTION_CHANGE',
        'filter': {'card_type': 'CREATURE'},
        'value1': 1,
        'str_val': 'rev-flag'
    }

    cmd = ActionConverter.convert(act)

    assert cmd['type'] == 'MUTATE'
    assert cmd.get('mutation_kind') == 'REVOLUTION_CHANGE'
    assert 'target_filter' in cmd
    assert cmd.get('amount') == 1
    assert cmd.get('str_param') == 'rev-flag'
    assert cmd.get('legacy_warning') is False
