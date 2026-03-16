from dm_toolkit import payment


def test_multiple_passives_max_floor_and_sum_reduction():
    card = {
        "cost": 20,
        "cost_reductions": [
            {"type": "PASSIVE", "amount": 5, "min_mana_cost": 10, "id": "p1"},
            {"type": "PASSIVE", "unit_cost": 3, "max_units": 4, "min_mana_cost": 6, "id": "p2"},
        ],
    }
    # units=3 -> second reduces by 3*3=9, first reduces by 5 => total reduction 14
    # base 20 - 14 = 6; floors are 10 and 6 -> max floor = 10 -> adjusted 10
    adjusted = payment.apply_passive_reductions(card, base_cost=20, units=3)
    assert adjusted == 10


def test_passive_then_active_combination_respects_floors():
    card = {
        "cost": 12,
        "cost_reductions": [
            {"type": "PASSIVE", "amount": 2, "min_mana_cost": 3, "id": "p1"},
            {"type": "ACTIVE_PAYMENT", "amount": 4, "id": "a1", "min_mana_cost": 1},
        ],
    }
    # Apply passive: 12-2=10 (floor 3 ignored)
    after_passive = payment.apply_passive_reductions(card, base_cost=12, units=1)
    assert after_passive == 10

    # Then apply active payment a1 amount=4 -> 10-4 = 6, floor from active is 1
    after_active = payment.apply_active_payment(card, base_cost=after_passive, reduction_id='a1', units=1)
    assert after_active == 6
