from dm_toolkit.gui.editor.models import CommandModel, LookAndAddParams


def test_look_and_add_ingest_and_serialize():
    data = {
        'type': 'LOOK_AND_ADD',
        'look_count': 5,
        'add_count': 2,
        'rest_zone': 'DECK_TOP'
    }

    cmd = CommandModel.model_validate(data)

    # params should be converted to LookAndAddParams
    assert isinstance(cmd.params, LookAndAddParams)
    assert cmd.params.look_count == 5
    assert cmd.params.add_count == 2
    assert cmd.params.rest_zone == 'DECK_TOP'

    # serialization (model_dump) should flatten params into top-level keys
    dumped = cmd.model_dump()
    assert dumped['type'] == 'LOOK_AND_ADD'
    assert dumped['look_count'] == 5
    assert dumped['add_count'] == 2
    assert dumped['rest_zone'] == 'DECK_TOP'
