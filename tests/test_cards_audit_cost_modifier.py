import json
from tools.cards_audit import audit_cost_modifier_fields_from_json


def test_audit_detects_missing_fields_for_stat_scaled():
    cards = [
        {
            "id": 1001,
            "static_abilities": [
                {"type": "COST_MODIFIER", "value_mode": "STAT_SCALED"}
            ],
        },
        {
            "id": 1002,
            "cost_reductions": [
                {"id": "cr_1", "value_mode": "STAT_SCALED", "stat_key": "CREATURES_PLAYED"}
            ],
        },
        {
            "id": 1003,
            "static_abilities": [
                {"type": "COST_MODIFIER", "value_mode": "STAT_SCALED", "stat_key": "CREATURES_PLAYED", "per_value": 1}
            ],
        },
    ]

    issues = audit_cost_modifier_fields_from_json(json.dumps(cards))
    # Expect two issues: card 1001 missing stat_key & per_value, card 1002 missing per_value
    ids = sorted([it['card_id'] for it in issues])
    assert 1001 in ids
    assert 1002 in ids
    # Ensure 1003 has no issues
    assert all(it['card_id'] != 1003 for it in issues)
