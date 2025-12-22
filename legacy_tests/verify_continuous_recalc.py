
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module

    # Setup
    registry = dm_ai_module.CardRegistry.instance()
    game_state = dm_ai_module.GameState(100) # Seed

    # 1. Define a Card with Static Ability (Grant +1000 Power)
    # We need ModifierDef.
    # Check if we can instantiate ModifierDef directly?
    # dm_ai_module.ModifierDef(type, value, filter, condition, str_val)

    # Filter for "SELF" (though we want to test empty filter -> self)
    # But wait, ModifierDef filter is a FilterDef object.

    # Test 1: Empty Filter -> Should imply SELF (Battle Zone)
    mod_filter = dm_ai_module.FilterDef() # Empty

    # Create ModifierDef
    # Type: POWER_MODIFIER = 0 ? Need enum value or check binding.
    # Assuming ModifierType is exposed.
    # If not exposed, we might need to guess integer values or use string if binding supports it.
    # Usually enum is dm_ai_module.ModifierType.POWER_MODIFIER

    if hasattr(dm_ai_module, 'ModifierType'):
        mod_type = dm_ai_module.ModifierType.POWER_MODIFIER
    else:
        print("ModifierType not exposed, using int 0 (POWER_MODIFIER)")
        mod_type = 0 # Assuming 0 is power modifier based on typical enum

    mod_def = dm_ai_module.ModifierDef()
    mod_def.type = mod_type
    mod_def.value = 1000
    mod_def.filter = mod_filter
    # condition empty

    # Create CardData
    # CardData(id, name, cost, civ, power, type, races, effects)
    # And then set static_abilities
    card_id = 9001
    card_data = dm_ai_module.CardData(card_id, "Lord Creature", 5, "FIRE", 3000, "CREATURE", [], [])
    card_data.static_abilities = [mod_def]

    dm_ai_module.register_card_data(card_data)

    # 2. Add card to Battle Zone
    # We use GameState helpers or direct manipulation if exposed
    # GameState.add_test_card_to_battle(player_id, card_id, instance_id, tapped, sick)

    # Add the Lord
    game_state.add_test_card_to_battle(0, card_id, 0, False, False)

    # Add a vanilla creature (checking if it gets buffed? No, filter is empty -> SELF only)
    # So the Lord itself should be 3000 + 1000 = 4000

    # 3. Recalculate
    # We call PhaseManager.start_turn to trigger recalculation
    # We need a card_db map for start_turn.
    # PhaseManager.start_turn(state, card_db)

    # We can get card_db from registry?
    # GenericCardSystem uses registry.
    # We can construct a dict mapping
    card_db = { card_id: card_data } # CardData inherits/compatible with CardDefinition in binding usually?
    # Wait, CardData and CardDefinition might be different types in Python.
    # But C++ expects map<CardID, CardDefinition>.
    # If bindings convert dict, CardData needs to be convertible to CardDefinition.
    # Usually CardDefinition binding constructor takes CardData? Or they are same?
    # inspect_module showed CardData.

    # Let's try passing the dict.

    print("Starting turn to trigger recalculation...")
    dm_ai_module.PhaseManager.start_turn(game_state, card_db)

    # 4. Check Power
    # We check passive_effects on the state
    passives = game_state.passive_effects
    print(f"Passive Effects count: {len(passives)}")

    found = False
    for p in passives:
        if p.source_instance_id == 0 and p.value == 1000:
            print("Found expected passive effect!")
            # Check filter
            print(f"Filter Owner: {p.target_filter.owner}")
            print(f"Filter Zones: {p.target_filter.zones}")

            if p.target_filter.owner == "SELF" and "BATTLE_ZONE" in p.target_filter.zones:
                print("SUCCESS: Filter correctly defaulted to SELF/BATTLE_ZONE")
                found = True
            else:
                print("FAILURE: Filter not set correctly")

    if not found:
        print("FAILURE: Passive effect not found")
        sys.exit(1)

    # Check actual power?
    # dm_ai_module.get_creature_power is not exposed directly usually, it's in GameLogicSystem.
    # But we can check GameState.get_card_power(instance_id) if exposed?
    # Or just rely on passive_effect presence.

    print("Test Passed")

except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
