from dm_toolkit import payment


def test_multiple_passive_reductions_sum():
    card = {
        "cost": 10,
        "cost_reductions": [
            {"type": "PASSIVE", "amount": 2, "id": "p1"},
            {"type": "PASSIVE", "reduction_per_unit": 1, "max_units": 3, "id": "p2"},
        ],
    }
    # units=2 -> second CR reduces by 1*2=2, total reduction = 2+2=4 -> adjusted 6
    adjusted = payment.apply_passive_reductions(card, base_cost=10, units=2)
    assert adjusted == 6


def test_non_passive_ignored():
    card = {
        "cost": 8,
        "cost_reductions": [
            {"type": "ACTIVE_PAYMENT", "amount": 3, "id": "a1"},
            {"type": "PASSIVE", "amount": 1, "id": "p1"},
        ],
    }
    # Only PASSIVE applies
    adjusted = payment.apply_passive_reductions(card, base_cost=8, units=1)
    assert adjusted == 7
