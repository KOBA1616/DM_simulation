import sys
import os
import json

# Add bin/ to path
bin_path = os.path.join(os.path.dirname(__file__), '../bin')
sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Failed to import dm_ai_module from {bin_path}")
    sys.exit(1)

def verify_trigger():
    print("Starting Trigger Verification...")

    cards_data = [
        {
            "id": 100, "name": "SelfTrigger", "type": "CREATURE", "cost": 1, "civilizations": ["FIRE"], "power": 1000,
            "effects": [{"trigger": "ON_PLAY", "trigger_scope": "NONE", "actions": []}]
        },
        {
            "id": 101, "name": "OppTrigger", "type": "CREATURE", "cost": 1, "civilizations": ["FIRE"], "power": 1000,
            "effects": [{"trigger": "ON_PLAY", "trigger_scope": "PLAYER_OPPONENT", "actions": []}]
        },
        {
            "id": 102, "name": "DrawTrigger", "type": "CREATURE", "cost": 1, "civilizations": ["FIRE"], "power": 1000,
            "effects": [{"trigger": "ON_DRAW", "trigger_scope": "PLAYER_OPPONENT", "actions": []}]
        },
        {
            "id": 103, "name": "AllTrigger", "type": "CREATURE", "cost": 1, "civilizations": ["FIRE"], "power": 1000,
            "effects": [{"trigger": "ON_PLAY", "trigger_scope": "ALL_PLAYERS", "actions": []}]
        }
    ]

    temp_json = "temp_cards.json"
    with open(temp_json, 'w') as f:
        json.dump(cards_data, f)

    db = dm_ai_module.JsonLoader.load_cards(temp_json)

    game = dm_ai_module.GameInstance(0, db)
    game.start_game()
    state = game.state

    tm = dm_ai_module.TriggerManager()

    # Helper to get pending count/info
    def get_pending():
        return state.get_pending_effects_info()

    # Test 1: Self Trigger (Card 100)
    print("\n--- Test 1: Self Trigger ---")
    state.add_test_card_to_battle(0, 100, 0, False, True)

    evt = dm_ai_module.GameEvent()
    evt.type = dm_ai_module.EventType.ZONE_ENTER
    evt.instance_id = 0
    evt.player_id = 0
    evt.context = {
        "to_zone": int(dm_ai_module.Zone.BATTLE),
        "from_zone": int(dm_ai_module.Zone.HAND),
        "instance_id": 0
    }

    tm.check_triggers(evt, state, db)

    pending = get_pending()
    print("Pending:", pending)
    triggered = [p for p in pending if p['source_instance_id'] == 0]
    assert len(triggered) >= 1
    print("SUCCESS: Self Trigger fired.")

    # state.pending_effects.clear()
    base_idx = len(pending)

    # Test 2: Opponent Trigger (Card 101)
    print("\n--- Test 2: Opponent Trigger ---")
    state.add_test_card_to_battle(0, 101, 1, False, False)
    # Opponent plays card 100 (instance 2)
    state.add_test_card_to_battle(1, 100, 2, False, True)

    evt = dm_ai_module.GameEvent()
    evt.type = dm_ai_module.EventType.ZONE_ENTER
    evt.instance_id = 2
    evt.player_id = 1
    evt.context = {
        "to_zone": int(dm_ai_module.Zone.BATTLE),
        "from_zone": int(dm_ai_module.Zone.HAND),
        "instance_id": 2
    }

    tm.check_triggers(evt, state, db)

    pending = get_pending()
    print("Pending:", pending)
    # Expect instance 1 (Observer) to trigger because P1 played
    # Filter pending to only new ones or check specific ID
    triggered = [p for p in pending[base_idx:] if p['source_instance_id'] == 1]
    assert len(triggered) >= 1
    print("SUCCESS: Opponent Trigger fired.")
    base_idx = len(pending)

    # Test 3: ON_DRAW Trigger (Card 102)
    print("\n--- Test 3: ON_DRAW Trigger ---")
    state.add_test_card_to_battle(0, 102, 3, False, False)

    # Opponent draws card (instance 4)
    # Note: Draw event context check: to_zone=HAND, from_zone=DECK
    evt = dm_ai_module.GameEvent()
    evt.type = dm_ai_module.EventType.ZONE_ENTER
    evt.instance_id = 4
    evt.player_id = 1
    evt.context = {
        "to_zone": int(dm_ai_module.Zone.HAND),
        "from_zone": int(dm_ai_module.Zone.DECK),
        "instance_id": 4
    }

    tm.check_triggers(evt, state, db)

    pending = get_pending()
    print("Pending:", pending)
    triggered = [p for p in pending[base_idx:] if p['source_instance_id'] == 3]
    assert len(triggered) >= 1
    print("SUCCESS: ON_DRAW Trigger fired.")

    print("\nVerification Successful!")
    os.remove(temp_json)

if __name__ == "__main__":
    verify_trigger()
