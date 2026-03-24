from dm_toolkit.gui.editor.models import CommandModel


def test_draw_card_params_ingest_and_serialize():
    data = {
        'type': 'DRAW_CARD',
        'amount': 2,
        'up_to': False,
        'destination': 'HAND'
    }

    cm = CommandModel.model_validate(data)
    assert cm.type == 'DRAW_CARD'
    params = cm.params
    # Accept either typed model or dict for compatibility
    if hasattr(params, 'amount'):
        assert params.amount == 2
        assert params.up_to is False
        assert params.destination == 'HAND'
    else:
        assert params.get('amount') == 2

    out = cm.model_dump()
    assert out.get('type') == 'DRAW_CARD'
    assert out.get('amount') == 2 or out.get('params', {}).get('amount') == 2


def test_discard_params_ingest_and_serialize():
    data = {
        'type': 'DISCARD',
        'amount': 3,
        'up_to': True,
        'reason': 'HAND_LIMIT'
    }

    cm = CommandModel.model_validate(data)
    assert cm.type == 'DISCARD'
    params = cm.params
    if hasattr(params, 'amount'):
        assert params.amount == 3
        assert params.up_to is True
        assert params.reason == 'HAND_LIMIT'
    else:
        assert params.get('amount') == 3

    out = cm.model_dump()
    assert out.get('type') == 'DISCARD'
    assert out.get('amount') == 3 or out.get('params', {}).get('amount') == 3
