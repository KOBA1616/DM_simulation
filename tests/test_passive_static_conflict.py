from dm_toolkit.gui.editor import validators_shared as v


def test_detects_conflict_when_both_present():
    card = {
        'id': 'c1',
        'name': 'ConflictCard',
        'cost_reductions': [
            {'type': 'PASSIVE', 'id': 'p1', 'min_mana_cost': 0}
        ],
        'static_abilities': [
            {'type': 'COST_MODIFIER', 'value': 1}
        ]
    }
    warns = v.detect_passive_static_conflicts(card)
    assert len(warns) == 1
    assert warns[0].startswith('ERROR:')
    assert 'PASSIVE' in warns[0] and 'COST_MODIFIER' in warns[0]


def test_no_conflict_when_only_one_side():
    card1 = {'cost_reductions': [{'type': 'PASSIVE'}]}
    card2 = {'static_abilities': [{'type': 'COST_MODIFIER', 'value': 1}]}
    assert v.detect_passive_static_conflicts(card1) == []
    assert v.detect_passive_static_conflicts(card2) == []
