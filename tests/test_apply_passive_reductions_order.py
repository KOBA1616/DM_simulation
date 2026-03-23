from dm_toolkit.payment import evaluate_cost, apply_passive_reductions


def test_apply_passive_reductions_only_applies_explicit_passives():
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

    # evaluate_cost.adjusted_after_passive should reflect explicit PASSIVE only
    assert plan.adjusted_after_passive == 8

    # apply_passive_reductions should also apply only explicit PASSIVE reductions
    adj = apply_passive_reductions(card, base)
    assert adj == 8
