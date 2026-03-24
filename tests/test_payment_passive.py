from dm_toolkit import payment


def test_passive_amount_reduction():
    card = {"cost": 5, "cost_reductions": [{"type": "PASSIVE", "amount": 2, "id": "p1"}]}
    adjusted = payment.apply_passive_reductions(card, base_cost=5, units=1)
    assert adjusted == 3


def test_passive_unit_cost_and_max_units():
    card = {"cost": 7, "cost_reductions": [{"type": "PASSIVE", "unit_cost": 1, "max_units": 2, "id": "p2"}]}
    # units=3 but max_units=2 -> reduction = 1 * 2 = 2
    adjusted = payment.apply_passive_reductions(card, base_cost=7, units=3)
    assert adjusted == 5


def test_passive_min_mana_floor():
    card = {"cost": 10, "cost_reductions": [{"type": "PASSIVE", "amount": 8, "min_mana_cost": 4, "id": "p3"}]}
    # reduction 8 -> 10-8=2 but floor min_mana_cost=4 -> adjusted 4
    adjusted = payment.apply_passive_reductions(card, base_cost=10, units=1)
    assert adjusted == 4


def test_can_pay_with_mana_true_false():
    card = {"cost": 6, "cost_reductions": [{"type": "PASSIVE", "amount": 2, "id": "p4"}]}
    assert payment.can_pay_with_mana(card, mana_available=4, base_cost=6)
    assert not payment.can_pay_with_mana(card, mana_available=3, base_cost=6)
