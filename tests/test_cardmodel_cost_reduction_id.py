from dm_toolkit.gui.editor.models import CardModel


def test_cardmodel_auto_assigns_cost_reduction_id():
    raw = {
        'id': 10,
        'name': 'AutoCR',
        'cost': 3,
        'cost_reductions': [
            {'type': 'PASSIVE', 'min_mana_cost': 0}
        ]
    }
    card = CardModel(**raw)
    assert card.cost_reductions
    cr = card.cost_reductions[0]
    assert isinstance(cr.id, str) and cr.id