from dm_toolkit.payment import evaluate_cost


def test_stat_scaled_no_reduction_below_min_stat():
    card = {
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "MY_STAT",
                "per_value": 1,
                # min_stat omitted -> defaults to 1
            }
        ]
    }

    # stat value 0 should produce no reduction when min_stat defaults to 1
    plan = evaluate_cost(card, base_cost=4, units=1, stat_values={"MY_STAT": 0})
    assert plan.adjusted_after_passive == 4


def test_stat_scaled_applies_without_max_reduction():
    card = {
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "MY_STAT",
                "per_value": 2,
                "min_stat": 1,
                # max_reduction omitted
            }
        ]
    }

    # stat value 4 -> (4 - 1 + 1) * 2 = 8 reduction
    plan = evaluate_cost(card, base_cost=12, units=1, stat_values={"MY_STAT": 4})
    assert plan.adjusted_after_passive == 4
