import pytest
from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_look_to_buffer_conversion():
    act = {'type': 'LOOK_TO_BUFFER', 'value1': 3, 'filter': {'zones': ['DECK']}}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'LOOK_TO_BUFFER'
    assert cmd['look_count'] == 3
    assert 'target_filter' in cmd


def test_select_from_buffer_conversion():
    act = {'type': 'SELECT_FROM_BUFFER', 'value1': 2, 'value2': 0}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'SELECT_FROM_BUFFER'
    assert cmd['amount'] == 2


def test_play_from_buffer_conversion():
    act = {'type': 'PLAY_FROM_BUFFER', 'destination_zone': 'BATTLE_ZONE', 'value1': 0}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'PLAY_FROM_BUFFER'
    assert cmd['from_zone'] == 'BUFFER'
    assert cmd['to_zone'] == 'BATTLE_ZONE'


def test_move_buffer_to_zone_conversion():
    act = {'type': 'MOVE_BUFFER_TO_ZONE', 'destination_zone': 'HAND', 'value1': 2}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'TRANSITION'
    assert cmd['from_zone'] == 'BUFFER'
    assert cmd['to_zone'] == 'HAND'
    assert cmd['amount'] == 2
