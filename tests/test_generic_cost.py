
import pytest
import dm_ai_module
from dm_ai_module import GameState, Player, CardDefinition, CardData, Civilization, CardType, Zone, EffectDef, ActionDef, EffectActionType, CostCalculator, PaymentType, ManaSystem

def create_dummy_card(id, cost, civ=Civilization.FIRE, hyper_energy=False, g_zero=False):
    keywords = dm_ai_module.CardKeywords()
    keywords.hyper_energy = hyper_energy
    keywords.g_zero = g_zero

    # We use CardDefinition constructor exposed in bindings
    # id, name, civ_enum, races, cost, power, keywords, effects
    card = CardDefinition(id, f"TestCard_{id}", civ, ["Human"], cost, 1000, keywords, [])
    return card

def setup_game():
    state = GameState(100)
    state.setup_test_duel()

    # We need a dummy DB
    card_db = {}
    return state, card_db

def test_standard_mana_cost():
    state, card_db = setup_game()
    card = create_dummy_card(1, 5, Civilization.FIRE)

    req = CostCalculator.calculate_requirement(state, state.players[0], card)
    assert req.base_mana_cost == 5
    assert req.final_mana_cost == 5
    assert req.uses_hyper_energy == False
    assert req.is_g_zero == False

def test_hyper_energy_cost_calculation():
    state, card_db = setup_game()
    card = create_dummy_card(2, 6, Civilization.WATER, hyper_energy=True)

    # Test with 0 creatures tapped
    req = CostCalculator.calculate_requirement(state, state.players[0], card, use_hyper_energy=True, hyper_energy_creature_count=0)
    assert req.final_mana_cost == 6
    assert req.uses_hyper_energy == True

    # Test with 1 creature tapped (reduces by 2)
    req = CostCalculator.calculate_requirement(state, state.players[0], card, use_hyper_energy=True, hyper_energy_creature_count=1)
    assert req.final_mana_cost == 4

    # Test with 3 creatures tapped (reduces by 6 -> 0)
    req = CostCalculator.calculate_requirement(state, state.players[0], card, use_hyper_energy=True, hyper_energy_creature_count=3)
    assert req.final_mana_cost == 0

def test_hyper_energy_floor():
    state, card_db = setup_game()
    card = create_dummy_card(3, 3, Civilization.NATURE, hyper_energy=True)

    # Tap 2 creatures (reduces by 4). Cost 3 - 4 = -1. Should be clamped to 0.
    req = CostCalculator.calculate_requirement(state, state.players[0], card, use_hyper_energy=True, hyper_energy_creature_count=2)
    assert req.final_mana_cost == 0

def test_g_zero_placeholder():
    # Since G-Zero condition logic wasn't fully implemented in the C++ block (just the structure),
    # we expect it to NOT trigger yet unless we flag it manually or mock it.
    # But let's verify the structure handles the flag if the keyword is present (implementation assumed TODO)
    state, card_db = setup_game()
    card = create_dummy_card(4, 8, Civilization.DARKNESS, g_zero=True)

    # Current implementation in C++ just checks the flag in calculate_requirement but
    # the TODO comment says "Evaluate G-Zero condition".
    # Since I didn't implement the condition evaluator, this test might just check standard cost
    # UNLESS I implemented the boolean check.
    # Looking at my C++ code:
    # if (card_def.keywords.g_zero) { // TODO: Evaluate... }
    # So it doesn't set is_g_zero = true yet.

    req = CostCalculator.calculate_requirement(state, state.players[0], card)
    assert req.final_mana_cost == 8 # Default behavior until implemented
    assert req.is_g_zero == False

def test_mana_system_integration():
    state, card_db = setup_game()
    card = create_dummy_card(5, 2, Civilization.LIGHT)

    # Should use the new logic
    cost = ManaSystem.get_adjusted_cost(state, state.players[0], card)
    assert cost == 2
