from dm_toolkit.gui.editor.models import CommandModel, MekraidParams


def test_mekraid_ingest_and_serialize():
    data = {
        'type': 'MEKRAID',
        'reveal_count': 4,
        'evolution_cost': 2
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, MekraidParams)
    assert cmd.params.reveal_count == 4
    assert cmd.params.evolution_cost == 2

    dumped = cmd.model_dump()
    assert dumped['type'] == 'MEKRAID'
    assert dumped['reveal_count'] == 4
    assert dumped['evolution_cost'] == 2
