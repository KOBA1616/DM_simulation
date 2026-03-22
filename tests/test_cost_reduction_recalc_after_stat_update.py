from dm_toolkit.payment import evaluate_cost


def make_stat_scaled_modifier(stat_key: str, per_value: int = 1, min_stat: int = 1, max_reduction: int | None = None):
    m = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": stat_key,
        "per_value": per_value,
        "min_stat": min_stat,
    }
    if max_reduction is not None:
        m["max_reduction"] = max_reduction
    return m


def test_evaluate_cost_reflects_stat_values_change():
    card = {
        "id": 9999,
        "static_abilities": [make_stat_scaled_modifier("SUMMON_COUNT_THIS_TURN", per_value=1, min_stat=1, max_reduction=3)],
    }

    base = 5

    # initial stats: zero summons -> no reduction
    plan0 = evaluate_cost(card, base_cost=base, stat_values={"SUMMON_COUNT_THIS_TURN": 0})
    assert plan0.final_cost == base, f"Expected no reduction when stat is 0, got {plan0.final_cost}"

    # after one summon -> reduction = (1 -1 +1)*1 =1
    plan1 = evaluate_cost(card, base_cost=base, stat_values={"SUMMON_COUNT_THIS_TURN": 1})
    assert plan1.final_cost == base - 1, f"Expected final_cost {base-1}, got {plan1.final_cost}"

    # after three summons -> reduction = min(max_reduction, (3-1+1)*1) = min(3,3)=3
    plan3 = evaluate_cost(card, base_cost=base, stat_values={"SUMMON_COUNT_THIS_TURN": 3})
    assert plan3.final_cost == max(base - 3, 0), f"Expected final_cost {max(base-3,0)}, got {plan3.final_cost}"
from dm_toolkit.payment import evaluate_cost


def test_stat_scaled_recalc_changes_cost():
    card = {
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "attacks",
                "per_value": 1,
                "min_stat": 1,
                "max_reduction": 3,
            }
        ]
    }

    base_cost = 5

    # No attacks -> no reduction
    plan_no = evaluate_cost(card, base_cost=base_cost, stat_values={"attacks": 0})
    assert plan_no.final_cost == base_cost

    # Two attacks -> reduction = (2 - 1 + 1) * 1 = 2 => final cost = 3
    plan_two = evaluate_cost(card, base_cost=base_cost, stat_values={"attacks": 2})
    assert plan_two.final_cost == 3
