
import pytest
import dm_ai_module

def test_cost_modifier_static_reduction():
    """
    Test that a CostModifier correctly reduces the cost of a card.
    """
    state = dm_ai_module.GameState(0)

    mod = dm_ai_module.CostModifier()
    mod.reduction_amount = 2
    mod.turns_remaining = 1
    mod.controller = 0
    mod.source_instance_id = -1

    f = dm_ai_module.FilterDef()
    mod.condition_filter = f

    modifiers = state.active_modifiers
    modifiers.append(mod)
    state.active_modifiers = modifiers

    card_def = dm_ai_module.CardDefinition(
        100, "Test Dragon", "FIRE", ["Dragon"], 5, 5000,
        dm_ai_module.CardKeywords(), []
    )

    # Use helper method to modify C++ state correctly
    for i in range(3):
        state.add_card_to_mana(0, 1, i)

    state.add_card_to_hand(0, 100, 10)

    card_db = {
        1: dm_ai_module.CardDefinition(1, "Dummy Mana", "FIRE", [], 1, 1000, dm_ai_module.CardKeywords(), []),
        100: card_def
    }

    state.active_player_id = 0
    state.current_phase = dm_ai_module.Phase.MAIN

    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)

    can_play = any(a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 100 for a in actions)
    assert can_play, "Should be able to play Cost 5 card with 3 mana and -2 reduction"

def test_cost_modifier_filtered():
    state = dm_ai_module.GameState(0)

    mod = dm_ai_module.CostModifier()
    mod.reduction_amount = 2
    mod.turns_remaining = 1
    mod.controller = 0
    mod.source_instance_id = -1

    f = dm_ai_module.FilterDef()
    f.races = ["Dragon"]
    mod.condition_filter = f

    modifiers = state.active_modifiers
    modifiers.append(mod)
    state.active_modifiers = modifiers

    dragon_def = dm_ai_module.CardDefinition(
        100, "Test Dragon", "FIRE", ["Dragon"], 5, 5000, dm_ai_module.CardKeywords(), []
    )
    human_def = dm_ai_module.CardDefinition(
        101, "Test Human", "FIRE", ["Human"], 5, 2000, dm_ai_module.CardKeywords(), []
    )

    card_db = {
        1: dm_ai_module.CardDefinition(1, "Dummy Mana", "FIRE", [], 1, 1000, dm_ai_module.CardKeywords(), []),
        100: dragon_def,
        101: human_def
    }

    for i in range(3):
        state.add_card_to_mana(0, 1, i)

    # Test Dragon
    state.add_card_to_hand(0, 100, 10)
    state.active_player_id = 0
    state.current_phase = dm_ai_module.Phase.MAIN

    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
    can_play_dragon = any(a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 100 for a in actions)
    assert can_play_dragon, "Dragon should get cost reduction"

    # Test Human
    state.add_card_to_hand(0, 101, 11)

    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
    can_play_human = any(a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 101 for a in actions)
    assert not can_play_human, "Human should NOT get cost reduction"

if __name__ == "__main__":
    test_cost_modifier_static_reduction()
    test_cost_modifier_filtered()
