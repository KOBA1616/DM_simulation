import pytest

from dm_toolkit.gui.editor.models import CommandModel, QueryParams


def test_commandmodel_ingests_query_params():
    data = {
        'type': 'QUERY',
        'params': {
            'query_string': 'find creatures',
            'options': ['active', 'owner_self']
        }
    }

    cmd = CommandModel.model_validate(data)

    # After ingest, params should be a QueryParams instance
    assert isinstance(cmd.params, QueryParams)
    assert cmd.params.query_string == 'find creatures'
    assert cmd.params.options == ['active', 'owner_self']


def test_serialize_flattens_query_params():
    data = {
        'type': 'QUERY',
        'params': {
            'query_string': 'find x',
            'options': ['o1']
        }
    }
    cmd = CommandModel.model_validate(data)
    dumped = cmd.model_dump()

    # serialized shape should flatten params into top-level keys
    assert dumped['type'] == 'QUERY'
    assert dumped['query_string'] == 'find x'
    assert dumped['options'] == ['o1']
