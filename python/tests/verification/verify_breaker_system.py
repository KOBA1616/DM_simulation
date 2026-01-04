
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../bin'))
# Add python directory to path (for other python modules if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import dm_ai_module as dm
from dm_ai_module import GameState, CardDefinition, CardType, Civilization, ActionType, EffectType, TriggerType, FilterDef, InstructionOp, CardKeywords, PassiveType, PassiveEffect, EffectResolver, CardDatabase

def test_breaker_logic():
    print("Initializing GameState...", flush=True)
    state = GameState(100) # Seed
    state.setup_test_duel()

    # Define Cards
    card_db = CardDatabase()

    def register(c):
        card_db[c.id] = c

    # 1. Register Cards
    c100 = CardDefinition()
    c100.id = 100
    c100.name = "Normal Guy"
    c100.type = CardType.CREATURE
    c100.civilization = dm.Civilization.FIRE
    c100.cost = 1
    c100.power = 1000
    register(c100)

    c101 = CardDefinition()
    c101.id = 101
    c101.name = "Double Guy"
    c101.type = CardType.CREATURE
    c101.civilization = dm.Civilization.FIRE
    c101.cost = 2
    c101.power = 2000
    kw101 = CardKeywords()
    kw101.double_breaker = True
    c101.keywords = kw101
    register(c101)

    c102 = CardDefinition()
    c102.id = 102
    c102.name = "Triple Guy"
    c102.type = CardType.CREATURE
    c102.civilization = dm.Civilization.FIRE
    c102.cost = 3
    c102.power = 3000
    kw102 = CardKeywords()
    kw102.triple_breaker = True
    c102.keywords = kw102
    register(c102)

    c103 = CardDefinition()
    c103.id = 103
    c103.name = "World Guy"
    c103.type = CardType.CREATURE
    c103.civilization = dm.Civilization.FIRE
    c103.cost = 10
    c103.power = 15000
    kw103 = CardKeywords()
    kw103.world_breaker = True
    c103.keywords = kw103
    register(c103)

    # Setup Board
    # Add instances to Battle Zone (Owner 0)

    state.add_test_card_to_battle(0, 100, 0, False, False)
    state.add_test_card_to_battle(0, 101, 1, False, False)
    state.add_test_card_to_battle(0, 102, 2, False, False)
    state.add_test_card_to_battle(0, 103, 3, False, False)

    # Retrieve instances AFTER setup to avoid iterator invalidation
    inst100 = state.get_card_instance(0)
    inst101 = state.get_card_instance(1)
    inst102 = state.get_card_instance(2)
    inst103 = state.get_card_instance(3)

    # Verify DB
    print(f"DB Keys: {[k for k in card_db]}", flush=True)
    print(f"100 in DB: {100 in card_db}", flush=True)

    # Verify Base Counts
    print("Verifying base counts...", flush=True)

    count100 = EffectResolver.get_breaker_count(state, inst100, card_db)
    print(f"Normal: {count100}", flush=True)
    assert count100 == 1

    count101 = EffectResolver.get_breaker_count(state, inst101, card_db)
    print(f"Double: {count101}", flush=True)
    assert count101 == 2

    count102 = EffectResolver.get_breaker_count(state, inst102, card_db)
    print(f"Triple: {count102}", flush=True)
    assert count102 == 3

    count103 = EffectResolver.get_breaker_count(state, inst103, card_db)
    print(f"World: {count103}", flush=True)
    assert count103 >= 999

    # Test Modifier: Grant Double Breaker to Normal Guy
    print("Testing modifier...", flush=True)
    pe = PassiveEffect()
    pe.type = PassiveType.KEYWORD_GRANT
    pe.str_value = "DOUBLE_BREAKER"
    pe.controller = 0
    pe.target_filter = FilterDef()
    # Filter targets Normal Guy (ID 100) or just "battle zone"
    pe.target_filter.zones = ["BATTLE_ZONE"]
    # pe.target_filter.card_types = [CardType.CREATURE] # Optional filter

    # Apply to state using new helper
    print(f"Passive Effects before: {state.get_passive_effect_count()}", flush=True)
    state.add_passive_effect(pe)
    print(f"Passive Effects after: {state.get_passive_effect_count()}", flush=True)

    # Verify Normal Guy now has 2
    count100_mod = EffectResolver.get_breaker_count(state, inst100, card_db)
    print(f"Normal (Modified): {count100_mod}", flush=True)
    assert count100_mod == 2

    # Modifier: Grant Triple Breaker
    pe2 = PassiveEffect()
    pe2.type = PassiveType.KEYWORD_GRANT
    pe2.str_value = "TRIPLE_BREAKER"
    pe2.controller = 0
    pe2.target_filter = FilterDef()
    pe2.target_filter.zones = ["BATTLE_ZONE"]

    state.add_passive_effect(pe2)

    # Should be 3 now (max of 2 and 3)
    count100_mod2 = EffectResolver.get_breaker_count(state, inst100, card_db)
    print(f"Normal (Modified 3): {count100_mod2}", flush=True)
    assert count100_mod2 == 3

    print("All tests passed.", flush=True)

if __name__ == "__main__":
    test_breaker_logic()
