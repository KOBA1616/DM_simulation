from dm_toolkit.gui.editor.models import CommandModel, QueryParams


def test_typed_params_serialize_flattened():
    qp = QueryParams(query_string='find', options=['a', 'b'])
    cmd = CommandModel(type='QUERY', params=qp)
    out = cmd.model_dump()  # uses model_serializer
    assert out['type'] == 'QUERY'
    assert out['query_string'] == 'find'
    assert out['options'] == ['a', 'b']
