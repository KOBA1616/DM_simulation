import sys
import os
import pytest

# Add bin to path
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module as m
except ImportError:
    m = None

@pytest.mark.skipif(m is None, reason="dm_ai_module not found")
def test_if_else_action():
    state = m.GameState(100)
    native_state = getattr(state, '_native', state)
    # Use registry's DB object to satisfy type check
    db = m.CardRegistry.get_all_cards()
    m.initialize_card_stats(native_state, db, 100)

    # Setup Deck
    native_state.add_card_to_deck(0, 1, 10)
    native_state.add_card_to_deck(0, 1, 11)
    native_state.add_card_to_deck(0, 1, 12)
    native_state.add_card_to_deck(0, 1, 13)

    # Register card
    card_data = m.CardData(
        1, "Test Card", 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], []
    )
    m.register_card_data(card_data)

    # Setup Hand: 1 card
    native_state.add_card_to_hand(0, 1, 0)

    # Condition: Hand Count >= 1 (TRUE)
    cond_true = m.ConditionDef()
    cond_true.type = "COMPARE_STAT"
    cond_true.stat_key = "MY_HAND_COUNT"
    cond_true.op = ">="
    cond_true.value = 1

    # Action: IF_ELSE
    action = m.ActionDef()
    action.type = m.EffectPrimitive.IF_ELSE
    action.condition = cond_true
    action.target_player = "PLAYER_SELF"

    # Option 0 (THEN): Draw 1
    then_act = m.ActionDef()
    then_act.type = m.EffectPrimitive.DRAW_CARD
    then_act.value1 = 1

    # Option 1 (ELSE): Draw 2
    else_act = m.ActionDef()
    else_act.type = m.EffectPrimitive.DRAW_CARD
    else_act.value1 = 2

    action.options = [[then_act], [else_act]]

    # Execute
    m.GenericCardSystem.resolve_action(native_state, action, 0)

    # Hand: 1 (init) + 1 (drawn) = 2
    assert len(state.players[0].hand) == 2

    # Condition: Hand Count >= 10 (FALSE)
    cond_false = m.ConditionDef()
    cond_false.type = "COMPARE_STAT"
    cond_false.stat_key = "MY_HAND_COUNT"
    cond_false.op = ">="
    cond_false.value = 10

    action.condition = cond_false

    # Execute
    m.GenericCardSystem.resolve_action(native_state, action, 0)

    # Hand: 2 (prev) + 2 (drawn via ELSE) = 4
    assert len(state.players[0].hand) == 4

@pytest.mark.skipif(m is None, reason="dm_ai_module not found")
def test_if_implicit_filter_condition():
    state = m.GameState(100)
    native_state = getattr(state, '_native', state)
    db = m.CardRegistry.get_all_cards()
    m.initialize_card_stats(native_state, db, 100)

    native_state.add_card_to_deck(0, 1, 10)
    native_state.add_card_to_deck(0, 1, 11)

    card_data = m.CardData(
        1, "Test Card", 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], []
    )
    m.register_card_data(card_data)

    # Setup Mana: 1 Fire
    native_state.add_card_to_mana(0, 1, 0)

    # Action: IF (Filter: Fire in Mana) -> Draw 1
    action = m.ActionDef()
    action.type = m.EffectPrimitive.IF
    action.target_player = "PLAYER_SELF"

    filter_def = m.FilterDef()
    filter_def.zones = ["MANA_ZONE"]
    filter_def.civilizations = m.CivilizationList([m.Civilization.FIRE])
    action.filter = filter_def

    then_act = m.ActionDef()
    then_act.type = m.EffectPrimitive.DRAW_CARD
    then_act.value1 = 1

    action.options = [[then_act]]

    # Execute
    m.GenericCardSystem.resolve_action(native_state, action, 0)

    # Hand: 0 + 1 = 1
    assert len(state.players[0].hand) == 1

    # Test False case: Water in Mana
    filter_false = m.FilterDef()
    filter_false.zones = ["MANA_ZONE"]
    filter_false.civilizations = m.CivilizationList([m.Civilization.WATER])
    action.filter = filter_false

    m.GenericCardSystem.resolve_action(native_state, action, 0)

    # Hand: 1 (unchanged)
    assert len(state.players[0].hand) == 1

if __name__ == "__main__":
    # Manually run if executed directly
    if m:
        test_if_else_action()
        test_if_implicit_filter_condition()
        print("Tests passed")
