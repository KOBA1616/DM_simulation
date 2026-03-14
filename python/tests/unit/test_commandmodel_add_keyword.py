from dm_toolkit.gui.editor.models import CommandModel, AddKeywordParams


def test_add_keyword_ingest_and_serialize():
    data = {
        'type': 'ADD_KEYWORD',
        'keyword': 'SPEED_ATTACK',
        'target': 'SELF',
        'duration': 2
    }

    cmd = CommandModel.model_validate(data)

    # params should be converted to AddKeywordParams
    assert isinstance(cmd.params, AddKeywordParams)
    assert cmd.params.keyword == 'SPEED_ATTACK'
    assert cmd.params.target == 'SELF'
    assert cmd.params.duration == 2

    dumped = cmd.model_dump()
    assert dumped['type'] == 'ADD_KEYWORD'
    assert dumped['keyword'] == 'SPEED_ATTACK'
    assert dumped['target'] == 'SELF'
    assert dumped['duration'] == 2
