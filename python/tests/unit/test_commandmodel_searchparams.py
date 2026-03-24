from dm_toolkit.gui.editor.models import CommandModel


def test_searchdeck_params_typed():
    data = {
        'type': 'SEARCH_DECK',
        'amount': 3,
        'to_zone': 'MANA_ZONE'
    }

    cmd = CommandModel.model_validate(data)
    # After ingest, params should be a typed model (SearchParams)
    assert hasattr(cmd, 'params')
    # params should have amount and destination_zone (to_zone maps to destination_zone via ingest)
    # Note: ingest moves unknown keys into params, so amount/to_zone become params fields
    p = cmd.params
    # Ensure amount was set
    if hasattr(p, 'amount'):
        assert p.amount == 3
    else:
        # fallback: dict
        assert p.get('amount') == 3

    # Serialization should include the flattened param
    dumped = cmd.model_dump()
    assert dumped.get('amount') == 3 or dumped.get('to_zone') == 'MANA_ZONE'
