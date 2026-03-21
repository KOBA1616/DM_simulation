from dm_toolkit.payment import evaluate_cost


def test_stat_scaled_reduction_applied():
    card = {
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "MY_STAT",
                "per_value": 1,
                "min_stat": 1,
                "max_reduction": 3,
            }
        ]
    }

    plan = evaluate_cost(card, base_cost=5, units=1, stat_values={"MY_STAT": 2})
    # reduction = (2 - 1 + 1) * 1 = 2 => adjusted_after_passive = 5 - 2 = 3
    assert plan.adjusted_after_passive == 3


def test_stat_scaled_respects_max_reduction():
    card = {
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "MY_STAT",
                "per_value": 2,
                "min_stat": 1,
                "max_reduction": 3,
            }
        ]
    }

    # stat value would produce (5 - 1 + 1) * 2 = 10, but max_reduction=3 clamps it
    plan = evaluate_cost(card, base_cost=5, units=1, stat_values={"MY_STAT": 5})
    assert plan.adjusted_after_passive == 2  # 5 - 3 = 2
