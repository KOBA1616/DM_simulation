
import sys
import os

# Add bin/ to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), "../bin"))

try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module. Make sure it is built and in bin/")
    sys.exit(1)

def test_trigger_system():
    print("Testing TriggerSystem...")

    # 1. Create GameState/GameInstance
    # Note: GameInstance signature: (seed, card_db)
    # We need a dummy card_db
    card_db = {}

    # Create dummy card definition
    # (id, name, civ, races, cost, power, keywords, effects)
    # Using CardData wrapper for convenience if available, or direct CardDefinition
    # Python binding for CardDefinition is:
    # (id, name, civ, races, cost, power, keywords, effects)
    keywords = dm_ai_module.CardKeywords()
    cdef = dm_ai_module.CardDefinition(
        1, "Dummy", dm_ai_module.Civilization.FIRE, ["Dragon"], 5, 5000, keywords, []
    )
    card_db[1] = cdef

    game = dm_ai_module.GameInstance(42, card_db)

    # 2. Access TriggerManager
    tm = game.trigger_manager
    if tm is None:
        print("Error: trigger_manager is None")
        sys.exit(1)

    print("TriggerManager accessed successfully.")

    # 3. Define a callback
    # Callback signature: (GameEvent, GameState)
    received_events = []

    def on_zone_enter(event, state):
        print(f"Callback received event: Type={event.type}, Source={event.source_id}")
        received_events.append(event)

    # 4. Subscribe
    tm.subscribe(dm_ai_module.EventType.ZONE_ENTER, on_zone_enter)
    print("Subscribed to ZONE_ENTER")

    # 5. Dispatch an event
    # GameEvent(type, source_id, target_id, player_id)
    evt = dm_ai_module.GameEvent(
        dm_ai_module.EventType.ZONE_ENTER,
        100, # Source ID (instance)
        -1,
        0    # Player ID
    )

    print("Dispatching event...")
    tm.dispatch(evt, game.state)

    # 6. Verify
    if len(received_events) == 1:
        e = received_events[0]
        if e.type == dm_ai_module.EventType.ZONE_ENTER and e.source_id == 100:
            print("SUCCESS: Event received correctly.")
        else:
            print(f"FAILURE: Event data mismatch. Got Type={e.type}, Source={e.source_id}")
            sys.exit(1)
    else:
        print(f"FAILURE: Expected 1 event, got {len(received_events)}")
        sys.exit(1)

if __name__ == "__main__":
    test_trigger_system()
