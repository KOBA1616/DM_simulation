import os
import json

from dm_ai_module import JsonLoader, GameInstance, StatCommand, StatType


def test_stat_scaled_fallback_active_modifiers():
    # Ensure Python fallback path
    os.environ['DM_DISABLE_NATIVE'] = '1'

    card = {
        "id": 1001,
        "name": "Test Creature",
        "type": "CREATURE",
        "cost": 3,
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CREATURES_PLAYED",
                "per_value": 1,
                "min_stat": 1,
                "max_reduction": 3,
            }
        ],
    }

    db = JsonLoader.load_cards(json.dumps([card]))
    gi = GameInstance(None, db)

    # place the card into player 0's battle zone so its static ability is active
    gi.add_test_card_to_battle(0, 1001, 1, tapped=False, sick=False)

    # initially no stats -> reduction should be zero
    initial_mods = getattr(gi.state, 'active_modifiers', [])
    assert all(getattr(m, 'reduction_amount', 0) == 0 for m in initial_mods)

    # apply a StatCommand to increment CREATURES_PLAYED to 2
    gi.execute_command(StatCommand(StatType.CREATURES_PLAYED, 2))

    mods = getattr(gi.state, 'active_modifiers', [])
    # at least one modifier should now report a positive reduction
    assert any(getattr(m, 'reduction_amount', 0) > 0 for m in mods)
