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

    # After passive + static (merged) reductions: 10 - (1 + 2) = 7
    assert plan.adjusted_after_passive == 7
    # total_passive_reduction should include the static conversion
    assert plan.total_passive_reduction == 3
    # After active reduction 3: final = 7 - 3 = 4
    assert plan.final_cost == 4
