from dm_toolkit.gui.editor.models import CommandModel


def test_play_from_zone_params_ingest_and_serialize():
    data = {
        'type': 'PLAY_FROM_ZONE',
        'source_zone': 'DECK',
        'destination_zone': 'HAND',
        'amount': 2,
    }

    cm = CommandModel.model_validate(data)
    assert cm.type == 'PLAY_FROM_ZONE'
    params = cm.params
    if hasattr(params, 'source_zone'):
        assert params.source_zone == 'DECK'
        assert params.destination_zone == 'HAND'
        assert params.amount == 2
    else:
        assert params.get('source_zone') == 'DECK'

    out = cm.model_dump()
    assert out.get('type') == 'PLAY_FROM_ZONE'
    assert out.get('source_zone') == 'DECK' or out.get('params', {}).get('source_zone') == 'DECK'
