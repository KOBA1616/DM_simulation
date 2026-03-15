import pytest
from dm_toolkit.gui.editor.models import CommandModel


def test_move_buffer_remain_to_zone_typed_params():
    data = {
        'type': 'MOVE_BUFFER_REMAIN_TO_ZONE',
        'to_zone': 'DECK',
        'filter': {'zones': ['HAND']}
    }

    cmd = CommandModel.model_validate(data)
    assert hasattr(cmd, 'params')
    params = cmd.params
    # Typed model should expose fields
    assert getattr(params, 'to_zone', None) == 'DECK'
    assert getattr(params, 'filter', None) == {'zones': ['HAND']}

    dumped = cmd.model_dump()
    assert dumped['type'] == 'MOVE_BUFFER_REMAIN_TO_ZONE'
    assert dumped.get('to_zone') == 'DECK'
    # legacy 'filter' should be preserved on serialization
    assert dumped.get('filter') == {'zones': ['HAND']}
