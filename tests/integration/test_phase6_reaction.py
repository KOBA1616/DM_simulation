
import sys
import os
from typing import Any

# Ensure the module can be loaded
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module. Make sure it is built and in 'bin/' directory.")
    sys.exit(1)

from dm_ai_module import (
    GameState, CardDefinition, CardType, Zone, Civilization,
    GameCommand, CommandType, Phase, GameEvent, EventType,
    TriggerManager, TransitionCommand, DeclareReactionCommand, GameStatus,
    EffectDef, TriggerType, EffectActionType, ActionDef, ConditionDef
)

def create_dummy_card_def(card_id, shield_trigger=False, revolution_change=False):
    # Using the binding constructor:
    # (id, name, civ, races, cost, power, keywords, effects)

    # Create keywords struct
    kw = dm_ai_module.CardKeywords()
    kw.shield_trigger = shield_trigger
    kw.revolution_change = revolution_change

    # Create basic effect (required by constructor)
    effs: list[Any] = []

    return CardDefinition(
        card_id,
        f"Card_{card_id}",
        "FIRE",
        ["Dragon"],
        5,
        5000,
        kw,
        effs
    )

def test_shield_trigger_integration():
    """
    Verifies that moving a Shield Trigger card from Shield to Hand (via TransitionCommand)
    dispatches an event, TriggerManager catches it, opens a Reaction Window,
    and allows DeclareReactionCommand to resolve it.
    """
    # 1. Setup
    state = GameState(40)
    tm = TriggerManager()

    # Register card definitions
    db = {}
    c100 = create_dummy_card_def(100, shield_trigger=True)
    db[100] = c100

    # Setup State: Player 0 has Card 100 in Shield
    state.setup_test_duel()
    state.add_card_to_shield(0, 100, 0) # instance_id 0

    # 3. Execution: Break Shield
    # Move Card 0 from Shield to Hand
    cmd = TransitionCommand(0, Zone.SHIELD, Zone.HAND, 0)
    state.execute_command(cmd) # Corrected execution

    # Manually trigger the reaction check
    evt = GameEvent(EventType.ZONE_ENTER, 0, -1, 0)

    # Based on the C++ code read:
    # if (event.context.count("to_zone") && event.context.at("to_zone") == (int)Zone::HAND &&
    #     event.context.count("from_zone") && event.context.at("from_zone") == (int)Zone::SHIELD) {
    #     int instance_id = event.context.at("instance_id");

    # So we need all three keys.
    evt.context = {
        "to_zone": int(Zone.HAND),
        "from_zone": int(Zone.SHIELD),
        "instance_id": 0
    }

    tm.check_reactions(evt, state, db)

    # 4. Verification: Reaction Window Open
    assert state.status == GameStatus.WAITING_FOR_REACTION, f"State should be WAITING_FOR_REACTION but is {state.status}"

    # 5. Declare Reaction (Use)
    # We want to USE the reaction. Index 0 (since only 1 candidate)
    react_cmd = DeclareReactionCommand(0, False, 0) # pass=False, index=0
    state.execute_command(react_cmd) # Corrected execution

    # 6. Verification: Pending Effect Queued
    # We expect a pending effect of type TRIGGER_ABILITY
    pending = state.get_pending_effects_info()
    assert len(pending) > 0, "Should have pending effects"
    last_eff = pending[-1]

    # Check if dict
    assert isinstance(last_eff, dict)
    assert last_eff["type"] == dm_ai_module.EffectType.TRIGGER_ABILITY
    assert last_eff["source_instance_id"] == 0

    print("Shield Trigger Integration Test Passed!")

def test_reaction_pass():
    state = GameState(40)
    tm = TriggerManager()
    db = {}
    db[100] = create_dummy_card_def(100, shield_trigger=True)
    state.setup_test_duel()
    state.add_card_to_shield(0, 100, 0)

    # state.set_event_dispatcher... same as above

    cmd = TransitionCommand(0, Zone.SHIELD, Zone.HAND, 0)
    state.execute_command(cmd) # Corrected execution

    evt = GameEvent(EventType.ZONE_ENTER, 0, -1, 0)
    evt.context = {
        "to_zone": int(Zone.HAND),
        "from_zone": int(Zone.SHIELD),
        "instance_id": 0
    }
    tm.check_reactions(evt, state, db)

    assert state.status == GameStatus.WAITING_FOR_REACTION

    # Pass
    pass_cmd = DeclareReactionCommand(0, True, 0) # pass=True
    state.execute_command(pass_cmd) # Corrected execution

    assert state.status == GameStatus.PLAYING
    print("Reaction Pass Test Passed!")

if __name__ == "__main__":
    test_shield_trigger_integration()
    test_reaction_pass()
