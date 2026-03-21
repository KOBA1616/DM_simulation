from tools.data_migration_fix_to_stat_scaled import migrate_card


def test_migrate_fixed_cost_modifier_to_stat_scaled():
    card = {
        "id": 2001,
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "FIXED",
                "value": 2,
                "condition": {
                    "type": "CARDS_MATCHING_FILTER",
                    "op": ">=",
                    "value": 1,
                    "filter": {"owner": "SELF", "zones": ["BATTLE_ZONE"], "races": ["DRAGON"]}
                }
            }
        ]
    }

    migrated = migrate_card(card)
    # Expect static_abilities[0] converted to STAT_SCALED with per_value equal to value and min_stat=1
    sab = migrated["static_abilities"][0]
    assert sab.get("value_mode") == "STAT_SCALED"
    assert sab.get("per_value") == 2
    assert sab.get("min_stat") == 1
    assert sab.get("stat_key") is not None


def test_migrate_no_change_when_already_stat_scaled():
    card = {
        "id": 2002,
        "static_abilities": [
            {"type": "COST_MODIFIER", "value_mode": "STAT_SCALED", "stat_key": "CREATURES_PLAYED", "per_value": 1}
        ]
    }
    migrated = migrate_card(card)
    assert migrated == card
