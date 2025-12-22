
import pytest
from dm_ai_module import GameState, CardDefinition, CardData, EffectDef, TriggerType, EffectActionType, ActionDef, GenericCardSystem, EffectResolver, Action, ActionType, Phase, Zone, Civilization, CardRegistry, get_pending_effects_info

def test_trigger_stack_behavior():
    # Setup
    state = GameState(100)
    state.active_player_id = 0

    import json
    card_json = {
        "id": 100,
        "name": "Trigger Creature",
        "cost": 5,
        "civilization": "FIRE",
        "type": "CREATURE",
        "power": 5000,
        "races": ["Human"],
        "effects": [
            {
                "trigger": "ON_PLAY",
                "actions": [
                    {
                        "type": "DRAW_CARD",
                        "value1": 1
                    }
                ]
            }
        ]
    }
    # Load into registry. New bindings ensure correct conversion to definitions.
    CardRegistry.load_from_json(json.dumps(card_json))

    # Add card to hand
    state.add_card_to_hand(0, 100, 1)
    # Add card to deck for drawing
    state.add_card_to_deck(0, 101, 2)

    # Verify initial state
    print(f"Hand size before play: {len(state.players[0].hand)}")

    # Play Card
    action = Action()
    action.type = ActionType.PLAY_CARD_INTERNAL
    action.source_instance_id = 1
    action.target_player = 0

    print("Resolving PLAY_CARD_INTERNAL action...")
    # Using registry implicitly by passing empty db (if supported) or retrieving it.
    # But wait, python binding for resolve_action requires db?
    # No, I updated binding to default to Registry if omitted.
    # BUT EffectResolver.resolve_action might not have the default bound or I need to check binding again.
    # EffectResolver binding: .def_static("resolve_action", &EffectResolver::resolve_action);
    # This takes 3 args: state, action, card_db.
    # I did NOT update EffectResolver binding. I updated GenericCardSystem bindings.
    # I should update EffectResolver binding too or pass Registry explicitly.
    # Let's pass Registry explicitly.

    # Wait, CardRegistry.get_all_definitions() is not bound to Python.
    # But CardRegistry.get_all_cards() IS bound.
    # Wait, EffectResolver expects map<CardID, CardDefinition>.
    # CardRegistry.get_all_cards() returns map<int, CardData>.
    # They are NOT compatible in Python either (unless pybind does magic conversion, which it likely doesn't for maps).

    # So I cannot easily pass Registry to EffectResolver from Python if I don't bind get_all_definitions.
    # AND if I don't bind it, I can't use it.

    # However, GenericCardSystem.resolve_trigger fallback logic (which I added to C++) uses Registry internally.
    # So if I pass an empty DB (or a dummy DB) to EffectResolver, it will call resolve_trigger.
    # resolve_trigger will check DB, fail, and fallback to Registry.
    # So this should work!

    card_db = {}

    EffectResolver.resolve_action(state, action, card_db)

    # Check pending effects
    pending = get_pending_effects_info(state)
    print(f"Pending effects: {pending}")

    assert len(pending) > 0, "Stack behavior failed: No pending effects."

    # Verify hand size (card moved to battle)
    print(f"Hand size after play: {len(state.players[0].hand)}")

    # Resolve the pending effect
    print("Resolving Pending Effect (Trigger)...")
    resolve_action = Action()
    resolve_action.type = ActionType.RESOLVE_EFFECT
    resolve_action.target_player = 0
    resolve_action.slot_index = 0

    EffectResolver.resolve_action(state, resolve_action, card_db)

    # Check if executed
    pending_after = get_pending_effects_info(state)
    print(f"Pending effects after resolution: {pending_after}")
    assert len(pending_after) == 0, "Effect was not consumed from stack."

    # Check outcome (Draw Card)
    # Hand size should be 1 (drawn card)
    hand_size = len(state.players[0].hand)
    print(f"Hand size after resolution: {hand_size}")
    assert hand_size == 1, f"Effect did not execute properly (Hand size {hand_size} != 1)"

    print("Test Passed: Trigger Stack Logic Verified.")

if __name__ == "__main__":
    test_trigger_stack_behavior()
