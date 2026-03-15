import pytest
from dm_toolkit.gui.editor.models import CommandModel


def test_play_from_buffer_typed_params():
    # Legacy dict with play-from-buffer fields
    data = {
        'type': 'PLAY_FROM_BUFFER',
        'buffer_index': 0,
        'to_zone': 'BATTLE_ZONE',
        'tapped': True,
        'summoned_for_free': False,
    }

    cmd = CommandModel.model_validate(data)
    # After ingest, params should be a PlayFromBufferParams model
    assert hasattr(cmd, 'params')
    params = cmd.params
    # Pydantic model should expose attributes
    assert getattr(params, 'buffer_index', None) == 0
    assert getattr(params, 'to_zone', None) == 'BATTLE_ZONE'
    assert getattr(params, 'tapped', None) is True

    # Serialization should flatten params back into top-level keys
    dumped = cmd.model_dump()
    assert dumped['type'] == 'PLAY_FROM_BUFFER'
    assert dumped.get('buffer_index') == 0
    assert dumped.get('to_zone') == 'BATTLE_ZONE'
    assert dumped.get('tapped') is True
