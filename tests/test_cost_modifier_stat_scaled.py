import pytest

from dm_toolkit.payment import evaluate_cost


def test_stat_scaled_applies_correctly():
    card = {
        'static_abilities': [
            {
                'type': 'COST_MODIFIER',
                'value_mode': 'STAT_SCALED',
                'stat_key': 'summon_count',
                'per_value': 1,
                'min_stat': 1,
            }
        ]
    }

    base_cost = 5
    stat_values = {'summon_count': 3}

    plan = evaluate_cost(card, base_cost, units=1, stat_values=stat_values)

    # calculated reduction = (3 - 1 + 1) * 1 = 3 -> final_cost = 5 - 3 = 2
    assert plan.final_cost == 2


def test_stat_scaled_respects_max_reduction():
    card = {
        'static_abilities': [
            {
                'type': 'COST_MODIFIER',
                'value_mode': 'STAT_SCALED',
                'stat_key': 'summon_count',
                'per_value': 1,
                'min_stat': 1,
                'max_reduction': 2,
            }
        ]
    }

    base_cost = 5
    stat_values = {'summon_count': 5}

    plan = evaluate_cost(card, base_cost, units=1, stat_values=stat_values)

    # raw calc = (5 -1 +1)*1 =5 -> clamped to max_reduction 2 -> final_cost = 5 - 2 = 3
    assert plan.final_cost == 3
# End of tests
 
