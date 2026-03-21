from dm_toolkit import payment


def make_card_with_conditional_fixed():
    return {
        "id": 9001,
        "name": "Support Static",
        "type": "CREATURE",
        "cost": 5,
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "FIXED",
                "value": 3,
                "condition": {
                    "type": "CARDS_MATCHING_FILTER",
                    "op": ">=",
                    "value": 1,
                    "filter": {
                        "owner": "SELF",
                        "zones": ["BATTLE_ZONE"],
                        "races": ["DRAGON"]
                    }
                }
            }
        ]
    }


def test_conditional_fixed_reduction_applies():
    card = make_card_with_conditional_fixed()
    # zone_state: SELF has 1 DRAGON in BATTLE_ZONE
    zone_state = {"SELF": {"BATTLE_ZONE": {"DRAGON": 1}}}
    plan = payment.evaluate_cost(card, base_cost=5, units=1, stat_values=None, zone_state=zone_state)
    assert plan.final_cost == 2


def test_conditional_fixed_reduction_not_applies_when_missing():
    card = make_card_with_conditional_fixed()
    # zone_state: no DRAGON present
    zone_state = {"SELF": {"BATTLE_ZONE": {"DRAGON": 0}}}
    plan = payment.evaluate_cost(card, base_cost=5, units=1, stat_values=None, zone_state=zone_state)
    assert plan.final_cost == 5
