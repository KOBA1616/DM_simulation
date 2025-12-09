
import sys
import os

# Add bin to path for dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
import dm_ai_module

def test_variable_linking_count_cards():
    """Verify COUNT_CARDS and DRAW_CARD with input variable works."""
    # 1. Setup Game
    state = dm_ai_module.GameState(100)

    # 2. Register dummy cards
    # ID 1: Creature (to be counted)
    # ID 2: Spell (to trigger effect)
    card_db = {}

    # Dummy Creature
    creature_def = dm_ai_module.CardData(1, "Dummy Creature", 1, "Fire", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(creature_def)

    # Spell with Variable Effect
    # Action 1: Count Creatures in Battle Zone -> "count_val"
    # Action 2: Draw Cards using "count_val"
    effect = dm_ai_module.EffectDef()
    effect.trigger = dm_ai_module.TriggerType.ON_PLAY
    effect.condition = dm_ai_module.ConditionDef()
    effect.condition.type = "NONE"

    # Act 1: COUNT_CARDS
    act1 = dm_ai_module.ActionDef()
    act1.type = dm_ai_module.EffectActionType.COUNT_CARDS
    act1.filter = dm_ai_module.FilterDef()
    act1.filter.zones = ["BATTLE_ZONE"]
    act1.filter.types = ["CREATURE"]
    act1.filter.owner = "SELF"
    act1.output_value_key = "my_creatures"

    # Act 2: DRAW_CARD
    act2 = dm_ai_module.ActionDef()
    act2.type = dm_ai_module.EffectActionType.DRAW_CARD
    act2.input_value_key = "my_creatures"

    effect.actions = [act1, act2]

    spell_def = dm_ai_module.CardData(2, "Variable Spell", 1, "Water", 0, "SPELL", [], [effect])
    dm_ai_module.register_card_data(spell_def)

    # 3. Setup State
    # Active player needs some creatures in battle zone
    state.add_test_card_to_battle(0, 1, 100, False, True) # 1
    state.add_test_card_to_battle(0, 1, 101, False, True) # 2
    state.add_test_card_to_battle(0, 1, 102, False, True) # 3

    # Player needs cards in deck to draw
    for i in range(10):
        state.add_card_to_deck(0, 1, i) # ID 1

    # Execute Effect
    # Manually trigger the effect of the spell

    dm_ai_module.GenericCardSystem.resolve_effect(state, effect, 999)

    # 4. Verify
    # Should have drawn 3 cards.
    hand_size = len(state.players[0].hand)

    assert hand_size == 3, f"Expected 3 cards drawn, got {hand_size}"

def test_variable_linking_get_stat():
    """Verify GET_GAME_STAT (Mana Civs) and SEND_TO_DECK_BOTTOM works."""
    state = dm_ai_module.GameState(100)

    # Setup Mana Zone with different civs
    # Fire
    c1 = dm_ai_module.CardData(10, "Fire Card", 1, "Fire", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c1)
    state.add_card_to_mana(0, 10, 200)

    # Water
    c2 = dm_ai_module.CardData(11, "Water Card", 1, "Water", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c2)
    state.add_card_to_mana(0, 11, 201)

    # Fire again (duplicate civ)
    state.add_card_to_mana(0, 10, 202)

    # Nature
    c3 = dm_ai_module.CardData(12, "Nature Card", 1, "Nature", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c3)
    state.add_card_to_mana(0, 12, 203)

    # Unique Civs: Fire, Water, Nature = 3

    # Define Effect
    # Act 1: Get Mana Civ Count -> "civ_count"
    # Act 2: Draw "civ_count" (3)
    # Act 3: Send "civ_count" cards from Hand to Deck Bottom

    effect = dm_ai_module.EffectDef()
    effect.trigger = dm_ai_module.TriggerType.ON_PLAY
    effect.condition.type = "NONE"

    act1 = dm_ai_module.ActionDef()
    act1.type = dm_ai_module.EffectActionType.GET_GAME_STAT
    act1.str_val = "MANA_CIVILIZATION_COUNT"
    act1.output_value_key = "civ_count"

    act2 = dm_ai_module.ActionDef()
    act2.type = dm_ai_module.EffectActionType.DRAW_CARD
    act2.input_value_key = "civ_count"

    act3 = dm_ai_module.ActionDef()
    act3.type = dm_ai_module.EffectActionType.SEND_TO_DECK_BOTTOM
    act3.scope = dm_ai_module.TargetScope.TARGET_SELECT
    act3.filter = dm_ai_module.FilterDef()
    act3.filter.zones = ["HAND"]
    act3.input_value_key = "civ_count" # Select 3

    effect.actions = [act1, act2, act3]

    # Add cards to deck for drawing
    for i in range(10):
        state.add_card_to_deck(0, 10, 300+i)

    # We need to simulate the "Target Selection" part of Act 3.
    # GenericCardSystem.resolve_effect will queue a PendingEffect for Act 3.

    # 1. Resolve Effect
    dm_ai_module.GenericCardSystem.resolve_effect(state, effect, 999)

    # Check intermediate state: Drawn 3 cards?
    # Hand should have 3 cards now.
    assert len(state.players[0].hand) == 3

    # Check Pending Effects
    pending = dm_ai_module.get_pending_effects_info(state)
    assert len(pending) == 1
    # Type should be NONE (from select_targets wrapper) or similar?
    # GenericCardSystem::select_targets uses EffectType::NONE for the pending effect wrapper usually.
    # The tuple is (type, source_id, controller)

    # Resolve the selection
    # We need to pick 3 cards.
    # Targets are instance ids.
    hand = state.players[0].hand
    target_ids = [c.instance_id for c in hand] # Select all 3

    # Use GenericCardSystem.resolve_effect_with_targets to finish it?
    # But wait, the PendingEffect in C++ stores the continuation "effect_def".
    # And it stores "execution_context" (we added this).
    # Does resolve_effect_with_targets pull context from pending effect?
    # NO. The C++ binding for resolve_effect_with_targets likely doesn't support passing context manually from Python easily,
    # OR the binding calls the version without context?

    # Wait, the PendingEffect holds the context.
    # But who calls resolve_effect_with_targets?
    # Usually EffectResolver.resolve_action(RESOLVE_EFFECT) calls it.

    # So we should generate an action to resolve the pending effect.
    # ActionType.RESOLVE_EFFECT with slot_index 0.

    # ActionGenerator usually does this?
    # We can manually create the action.

    resolve_action = dm_ai_module.Action()
    resolve_action.type = dm_ai_module.ActionType.RESOLVE_EFFECT
    resolve_action.slot_index = 0
    # The action itself doesn't carry targets?
    # Wait, ActionType.SELECT_TARGET adds targets to the pending effect.
    # Then RESOLVE_EFFECT executes it.

    # So:
    # 1. Add targets to pending effect.
    # ActionType.SELECT_TARGET
    for tid in target_ids:
        sel_act = dm_ai_module.Action()
        sel_act.type = dm_ai_module.ActionType.SELECT_TARGET
        sel_act.slot_index = 0
        sel_act.target_instance_id = tid
        dm_ai_module.EffectResolver.resolve_action(state, sel_act, {})

    # 2. Resolve
    res_act = dm_ai_module.Action()
    res_act.type = dm_ai_module.ActionType.RESOLVE_EFFECT
    res_act.slot_index = 0
    dm_ai_module.EffectResolver.resolve_action(state, res_act, {})

    # Verify: Hand should be 0 (3 drawn, 3 sent to deck bottom).
    assert len(state.players[0].hand) == 0
    # Deck should be 10 (started 10, drew 3, returned 3).
    assert len(state.players[0].deck) == 10

    print("Test Passed")

if __name__ == "__main__":
    test_variable_linking_count_cards()
    test_variable_linking_get_stat()
