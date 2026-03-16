from dm_toolkit import payment


def test_cannot_pay_when_missing_required_civ_even_if_total_sufficient():
    card = {'id': 1, 'name': 'Aqua', 'cost': 3, 'civilization': 'WATER'}
    mana_pool = {'NATURE': 3}
    assert not payment.can_pay_with_mana(card, mana_pool, base_cost=3)


def test_can_pay_when_has_required_civ_and_total_sufficient():
    card = {'id': 2, 'name': 'Aqua', 'cost': 3, 'civilization': 'WATER'}
    mana_pool = {'WATER': 1, 'NATURE': 2}
    assert payment.can_pay_with_mana(card, mana_pool, base_cost=3)


def test_int_mana_pool_is_conservative_for_civ_requirement():
    card = {'id': 3, 'name': 'Aqua', 'cost': 2, 'civilization': 'WATER'}
    # integer total should be treated as lacking per-civ info -> conservative False
    assert not payment.can_pay_with_mana(card, 2, base_cost=2)


def test_passive_reduction_still_requires_civ_in_pool():
    card = {
        'id': 4,
        'name': 'AquaBoost',
        'cost': 4,
        'civilizations': ['WATER'],
        'cost_reductions': [
            {'type': 'PASSIVE', 'amount': 2, 'min_mana_cost': 0}
        ]
    }
    mana_pool = {'NATURE': 2}
    # adjusted cost becomes 2, total mana ==2 but lacks WATER -> should be False
    assert not payment.can_pay_with_mana(card, mana_pool, base_cost=4)
