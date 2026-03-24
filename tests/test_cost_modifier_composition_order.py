from dm_toolkit.payment import evaluate_cost


def test_composition_order_passive_then_static_then_active():
    # explicit PASSIVE reduces by 2
    # static COST_MODIFIER FIXED reduces by 3
    card = {
        "cost_reductions": [
            {"type": "PASSIVE", "id": "p1", "amount": 2}
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value_mode": "FIXED", "value": 3}
        ]
    }

    base = 10
    plan = evaluate_cost(card, base)

    # After PASSIVE only: 10 - 2 = 8
    assert plan.adjusted_after_passive == 8
    # static reductions applied next: final cost = 8 - 3 = 5
    assert plan.final_cost == 5
    # total_passive_reduction should reflect only explicit passives
    assert plan.total_passive_reduction == 2
from dm_toolkit.payment import evaluate_cost


def test_cost_modifier_composition_order():
    # Base cost 10
    # PASSIVE reduces by 1
    # STATIC COST_MODIFIER (value) reduces by 2 (merged into passive by toolkit)
    # ACTIVE_PAYMENT reduces by 3
    card = {
        "cost_reductions": [
            {"type": "PASSIVE", "id": "pass1", "amount": 1},
            {"type": "ACTIVE_PAYMENT", "id": "act1", "amount": 3},
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value": 2}
        ],
    }

    plan = evaluate_cost(card, base_cost=10, units=1, active_reduction_id="act1", active_units=1)

    # With explicit PASSIVE applied first: after PASSIVE = 10 - 1 = 9
    assert plan.adjusted_after_passive == 9
    # total_passive_reduction should reflect only explicit PASSIVE entries
    assert plan.total_passive_reduction == 1
    # After static (2) then active (3): final = 9 - 2 - 3 = 4
    assert plan.final_cost == 4
