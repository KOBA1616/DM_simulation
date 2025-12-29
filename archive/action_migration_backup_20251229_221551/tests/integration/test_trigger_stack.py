
import pytest
from dm_ai_module import GameInstance, GameState, CardDefinition, CardData, EffectDef, TriggerType, EffectActionType, ActionDef, EffectResolver, Action, ActionType, Phase, Zone, Civilization, CardRegistry

@pytest.mark.skip(reason="Work in progress: Trigger detection for ON_PLAY via GameInstance needs debugging of PLAY_CARD vs PLAY_CARD_INTERNAL flow")
def test_trigger_stack_behavior():
    # Setup using GameInstance to ensure TriggerManager is wired up

    # 1. Define Card Data
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
    # Load into registry
    CardRegistry.load_from_json(json.dumps(card_json))

    # 2. Initialize GameInstance
    # This automatically uses CardRegistry.get_all_definitions()
    game = GameInstance(100)
    state = game.state
    state.active_player_id = 0

    # 3. Setup Initial State
    # Add card to hand (ID 100, Instance ID 1)
    state.add_card_to_hand(0, 100, 1)
    # Add card to deck for drawing (ID 101, Instance ID 2)
    state.add_card_to_deck(0, 101, 2)

    # Verify initial state
    print(f"Hand size before play: {len(state.players[0].hand)}")

    # 4. Execute Play Action via GameInstance
    action = Action()
    action.type = ActionType.PLAY_CARD_INTERNAL
    action.source_instance_id = 1
    action.target_player = 0

    print("Resolving PLAY_CARD_INTERNAL action...")
    game.resolve_action(action)

    # 5. Check Pending Effects
    # The GameInstance wiring should have triggered TriggerManager -> PendingEffect
    pending = state.get_pending_effects_info()
    print(f"Pending effects: {pending}")

    assert len(pending) > 0, "Stack behavior failed: No pending effects."

    # Verify card moved to battle zone (Hand size decreases)
    print(f"Hand size after play: {len(state.players[0].hand)}")
    # It should be 0 now (1 played)
    assert len(state.players[0].hand) == 0

    # 6. Resolve Pending Effect
    print("Resolving Pending Effect (Trigger)...")
    resolve_action = Action()
    resolve_action.type = ActionType.RESOLVE_EFFECT
    resolve_action.target_player = 0
    resolve_action.slot_index = 0

    game.resolve_action(resolve_action)

    # 7. Verify Outcome
    # Check if executed (Pending effect removed)
    pending_after = state.get_pending_effects_info()
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
