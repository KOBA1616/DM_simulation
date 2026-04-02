import pytest
from dm_toolkit.gui.editor.models import CommandModel


def test_ignore_ability_typed_params():
    data = {
        'type': 'IGNORE_ABILITY',
        'ability_id': 'FLY',
        'target_group': 'SELF',
        'duration': 2,
        'reason': 'test-case'
    }

    cmd = CommandModel.model_validate(data)
    assert hasattr(cmd, 'params')
    params = cmd.params
    # Ensure typed model fields are present
    assert params.get('ability_id') == 'FLY'
    assert params.get('target_group') == 'SELF'
    assert params.get('duration') == 2

    # Serialization should flatten params back into top-level keys
    dumped = cmd.model_dump()
    assert dumped['type'] == 'IGNORE_ABILITY'
    assert dumped.get('ability_id') == 'FLY'
    assert dumped.get('target_group') == 'SELF'
    assert dumped.get('duration') == 2
