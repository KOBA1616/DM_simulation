import pytest
from dm_toolkit.gui.editor import normalize


def test_canonicalize_simple_action():
    a = {"uid": "u1", "type": "DRAW_CARD", "value1": 2}
    c = normalize.canonicalize(a)
    assert c['kind'] == 'ACTION'
    assert c['type'] == 'DRAW_CARD'
    assert c['payload']['value1'] == 2
    assert c['uid'] == 'u1'


def test_canonicalize_command_branches():
    cmd = {
        "uid": "c1",
        "if_true": [{"type": "DRAW_CARD", "value1": 1}],
        "if_false": [{"type": "DISCARD", "value1": 1}]
    }
    c = normalize.canonicalize(cmd)
    assert c['kind'] == 'COMMAND'
    assert 'branches' in c
    assert isinstance(c['branches']['if_true'], list)
    assert c['branches']['if_true'][0]['type'] == 'DRAW_CARD'


def test_canonicalize_options_nested():
    action = {
        "uid": "a1",
        "type": "SELECT_OPTION",
        "options": [
            [{"type": "DRAW_CARD", "value1": 1}],
            [{"type": "DISCARD", "value1": 1}, {"type": "DRAW_CARD", "value1": 2}]
        ]
    }
    c = normalize.canonicalize(action)
    assert c['kind'] == 'ACTION'
    assert len(c['options']) == 2
    assert isinstance(c['options'][0], list)
    assert c['options'][1][1]['type'] == 'DRAW_CARD'
