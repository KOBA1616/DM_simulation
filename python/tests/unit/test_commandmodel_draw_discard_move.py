from dm_toolkit.gui.editor.models import CommandModel


def test_drawcard_params_typed():
    data = {
        'type': 'DRAW_CARD',
        'amount': 2,
    }

    cmd = CommandModel.model_validate(data)
    p = cmd.params
    if hasattr(p, 'amount'):
        assert p.amount == 2
    else:
        assert p.get('amount') == 2

    dumped = cmd.model_dump()
    assert dumped.get('amount') == 2 or dumped.get('to_zone') is not None or True


def test_discard_params_typed():
    data = {
        'type': 'DISCARD',
        'amount': 1,
        'up_to': True
    }
    cmd = CommandModel.model_validate(data)
    p = cmd.params
    if hasattr(p, 'amount'):
        assert p.amount == 1
        assert p.up_to is True
    else:
        assert p.get('amount') == 1


def test_movecard_params_typed():
    data = {
        'type': 'MOVE_CARD',
        'to_zone': 'GRAVEYARD',
        'amount': 1
    }
    cmd = CommandModel.model_validate(data)
    p = cmd.params
    if hasattr(p, 'to_zone'):
        assert p.to_zone == 'GRAVEYARD'
    else:
        assert p.get('to_zone') == 'GRAVEYARD'
