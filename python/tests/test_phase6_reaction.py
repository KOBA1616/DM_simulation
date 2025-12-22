
import sys
import os

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
    TriggerManager, TransitionCommand, DeclareReactionCommand, Status,
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
    effs = []

    return CardDefinition(
        card_id,
        f"Card_{card_id}",
        Civilization.FIRE,
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
    state.add_test_card_to_shield(0, 100, 0) # instance_id 0

    # 2. Hook up Event Dispatcher
    # We define a python callback that delegates to TriggerManager
    def dispatch_cb(event):
        # Check reactions
        tm.check_reactions(event, state, db)

    state.set_event_dispatcher(dispatch_cb)

    # 3. Execution: Break Shield
    # Move Card 0 from Shield to Hand
    cmd = TransitionCommand(0, Zone.SHIELD, Zone.HAND, 0)
    cmd.execute(state)

    # 4. Verification: Reaction Window Open
    assert state.status == Status.WAITING_FOR_REACTION, "State should be WAITING_FOR_REACTION"
    # Cannot easily access reaction_stack via bindings unless exposed as list...
    # bindings.cpp did not expose reaction_stack directly as a list we can inspect easily?
    # Actually it's hidden. We rely on status.

    # 5. Declare Reaction (Use)
    # We want to USE the reaction. Index 0 (since only 1 candidate)
    react_cmd = DeclareReactionCommand(0, False, 0) # pass=False, index=0
    react_cmd.execute(state)

    # 6. Verification: Pending Effect Queued
    # We expect a pending effect of type TRIGGER_ABILITY
    pending = dm_ai_module.get_pending_effects_info(state)
    assert len(pending) > 0, "Should have pending effects"
    last_eff = pending[-1]
    # Tuple: (type, source, controller)
    # EffectType.TRIGGER_ABILITY is what we used
    assert last_eff[0] == dm_ai_module.EffectType.TRIGGER_ABILITY
    assert last_eff[1] == 0 # instance_id

    print("Shield Trigger Integration Test Passed!")

def test_reaction_pass():
    state = GameState(40)
    tm = TriggerManager()
    db = {}
    db[100] = create_dummy_card_def(100, shield_trigger=True)
    state.setup_test_duel()
    state.add_test_card_to_shield(0, 100, 0)

    state.set_event_dispatcher(lambda e: tm.check_reactions(e, state, db))

    cmd = TransitionCommand(0, Zone.SHIELD, Zone.HAND, 0)
    cmd.execute(state)

    assert state.status == Status.WAITING_FOR_REACTION

    # Pass
    pass_cmd = DeclareReactionCommand(0, True) # pass=True
    pass_cmd.execute(state)

    assert state.status == Status.PLAYING
    print("Reaction Pass Test Passed!")

if __name__ == "__main__":
    test_shield_trigger_integration()
    test_reaction_pass()
