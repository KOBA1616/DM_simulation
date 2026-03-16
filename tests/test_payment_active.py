from dm_toolkit import payment


def test_active_reduction_per_unit_and_max_units():
    card = {
        "cost": 10,
        "cost_reductions": [
            {"type": "ACTIVE_PAYMENT", "id": "a1", "reduction_per_unit": 2, "max_units": 3},
        ],
    }
    # select units=2 -> reduction = 2*2 = 4 -> adjusted 6
    adjusted = payment.apply_active_payment(card, base_cost=10, reduction_id='a1', units=2)
    assert adjusted == 6


def test_active_units_field_override_when_no_units_selected():
    card = {
        "cost": 8,
        "cost_reductions": [
            {"type": "ACTIVE_PAYMENT", "id": "a2", "reduction_per_unit": 1, "units": 3},
        ],
    }
    # if units param omitted (None), uses cr.units == 3
    adjusted = payment.apply_active_payment(card, base_cost=8, reduction_id='a2', units=None)
    assert adjusted == 5


def test_active_amount_one_shot():
    card = {
        "cost": 6,
        "cost_reductions": [
            {"type": "ACTIVE_PAYMENT", "id": "a3", "amount": 3},
        ],
    }
    adjusted = payment.apply_active_payment(card, base_cost=6, reduction_id='a3', units=1)
    assert adjusted == 3
