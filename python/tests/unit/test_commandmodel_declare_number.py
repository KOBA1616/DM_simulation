import pytest
from dm_toolkit.gui.editor.models import CommandModel


def test_declare_number_typed_params():
    data = {
        'type': 'DECLARE_NUMBER',
        'value1': 1,
        'value2': 5
    }

    cmd = CommandModel.model_validate(data)
    assert hasattr(cmd, 'params')
    params = cmd.params
    assert getattr(params, 'value1', None) == 1
    assert getattr(params, 'value2', None) == 5

    dumped = cmd.model_dump()
    assert dumped['type'] == 'DECLARE_NUMBER'
    assert dumped.get('value1') == 1
    assert dumped.get('value2') == 5
