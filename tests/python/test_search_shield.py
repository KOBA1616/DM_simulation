import pytest
from dm_ai_module import GameState, Phase, Zone, ActionType, EffectActionType, CardData, CardDefinition, TriggerType, EffectDef, ActionDef, FilterDef, TargetScope, register_card_data, JsonLoader, PhaseManager, GenericCardSystem, ConditionDef
import json

# Define card IDs
SEARCH_SPELL_ID = 200
SHIELD_ADD_ID = 201
SHIELD_BURN_ID = 202

def setup_cards():
    # Register Search Spell: "Look at top 3 cards, add 1 Creature to hand, rest to bottom"
    # Note: Using SEARCH_DECK_BOTTOM which is implemented.
    # But for Phase 6 we need full SEARCH_DECK logic if possible, but let's test what we have.
    # The requirement says "Action: SEARCH_DECK (or LOOK_AT_DECK + SELECT_FROM_DECK)".
    # Current SEARCH_DECK_BOTTOM is "Look N, select 1 matching filter, rest bottom".
    search_action = ActionDef(
        EffectActionType.SEARCH_DECK_BOTTOM,
        TargetScope.NONE,
        FilterDef(types=["CREATURE"])
    )
    search_action.value1 = 3

    search_effects = [
        EffectDef(
            TriggerType.ON_PLAY,
            ConditionDef(), # No condition
            [search_action]
        )
    ]
    register_card_data(CardData(SEARCH_SPELL_ID, "SearchSpell", 2, "WATER", 0, "SPELL", [], search_effects))

    # Register Shield Add Creature: "When enters, add top deck to shield"
    shield_add_effects = [
        EffectDef(
            TriggerType.ON_PLAY,
            ConditionDef(),
            [
                ActionDef(
                    EffectActionType.NONE, # Placeholder until implemented
                    TargetScope.NONE,
                    FilterDef()
                )
            ]
        )
    ]
    # We will manually trigger the action via GenericCardSystem.resolve_action to test the logic even if JSON mapping isn't perfect yet
    register_card_data(CardData(SHIELD_ADD_ID, "ShieldAdder", 3, "LIGHT", 3000, "CREATURE", [], []))

    # Register Shield Burner
    register_card_data(CardData(SHIELD_BURN_ID, "ShieldBurner", 4, "FIRE", 4000, "CREATURE", [], []))

def test_shuffle_deck_logic():
    state = GameState(0)
    # Setup deck with sequential IDs to verify shuffle
    state.set_deck(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    original_order = [c.card_id for c in state.players[0].deck]

    # We need to invoke SHUFFLE_DECK. It's not in bindings yet as ActionType or EffectActionType in logic,
    # but I plan to implement it.
    # I will add EffectActionType.SHUFFLE_DECK to the implementation and bindings.
    action = ActionDef(
        EffectActionType.SHUFFLE_DECK,
        TargetScope.NONE,
        FilterDef()
    )

    GenericCardSystem.resolve_action(state, action, -1)

    new_order = [c.card_id for c in state.players[0].deck]
    # Verify order changed (probabilistic, but highly likely for 10 cards)
    # If it fails, it might be random chance, but very low.
    assert new_order != original_order

def test_search_deck_bottom():
    setup_cards()
    state = GameState(0)
    # Setup deck: Top is Spell (ID 200), then Creature (ID 201), then Creature (ID 202)
    # SEARCH_DECK_BOTTOM looks at top 3.
    # If we filter for "CREATURE", it should skip the Spell and pick the Creature?
    # Wait, SEARCH_DECK_BOTTOM implementation in GenericCardSystem:
    # "Auto-select first matching card for simplicity (MVP)"
    # So if top is Spell (no match), 2nd is Creature (match), it picks 2nd.
    # 1st and 3rd go to bottom.

    state.set_deck(0, [202, 201, 200]) # 200 is top (back), 201, 202
    # Deck: [202, 201, 200] (Top)

    # Execute effect
    action = ActionDef(
        EffectActionType.SEARCH_DECK_BOTTOM,
        TargetScope.NONE,
        FilterDef(types=["CREATURE"])
    )
    action.value1 = 3

    GenericCardSystem.resolve_action(state, action, -1)

    # Check Hand: Should have ID 201 (First creature found from top)
    # Top was 200 (Spell), skipped.
    # Next was 201 (Creature), picked.
    # Next was 202 (Creature), not picked (only 1).

    hand_ids = [c.card_id for c in state.players[0].hand]
    assert 201 in hand_ids
    assert len(hand_ids) == 1

    # Check Deck: Should have 200 and 202 at bottom (begin).
    deck_ids = [c.card_id for c in state.players[0].deck]
    assert len(deck_ids) == 2
    assert 200 in deck_ids
    assert 202 in deck_ids
    # Order at bottom depends on implementation (insert at begin).
    # Logic: loop looked, if chosen skip, else insert at begin.
    # order in 'looked' was [200, 201, 202].
    # chosen 201.
    # i=0 (200): insert at begin -> [200]
    # i=2 (202): insert at begin -> [202, 200]
    # So bottom is 202, then 200.
    assert deck_ids[0] == 202
    assert deck_ids[1] == 200

if __name__ == "__main__":
    setup_cards()
    test_search_deck_bottom()
