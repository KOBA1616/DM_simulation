from dm_toolkit import payment


def test_evaluate_cost_passive_only():
    card = {
        'cost': 5,
        'cost_reductions': [
            {'id': 'p1', 'type': 'PASSIVE', 'amount': 2},
            {'id': 'p2', 'type': 'PASSIVE', 'reduction_per_unit': 1, 'max_units': 2},
        ]
    }
    plan = payment.evaluate_cost(card, base_cost=5, units=3)
    # p1 gives 2, p2 gives min(3,2)*1 =2 => total 4, adjusted floor 1
    assert plan.total_passive_reduction == 4
    assert plan.adjusted_after_passive == 1
    assert plan.final_cost == 1


def test_evaluate_cost_with_active():
    card = {
        'cost': 6,
        'cost_reductions': [
            {'id': 'p1', 'type': 'PASSIVE', 'amount': 1},
            {'id': 'a1', 'type': 'ACTIVE_PAYMENT', 'reduction_per_unit': 2, 'max_units': 2, 'min_mana_cost': 1},
        ]
    }
    plan = payment.evaluate_cost(card, base_cost=6, units=2, active_reduction_id='a1', active_units=2)
    # passive 1 => adjusted_after_passive 5; active reduces by 2*2=4 => final 1, with min floor 1 -> stays 1
    assert plan.total_passive_reduction == 1
    assert plan.active_reduction_amount == 4
    assert plan.final_cost == 1


def test_can_pay_with_mana_uses_plan():
    card = {'cost': 4, 'civilization': 'WATER', 'cost_reductions': [{'type': 'PASSIVE', 'amount': 1}]}
    mana_pool = {'WATER': 1, 'NATURE': 2}
    # base 4 -> passive 1 -> final 3 -> total mana 3 -> True
    assert payment.can_pay_with_mana(card, mana_pool, base_cost=4)
