#!/usr/bin/env python
"""Test to verify which resolve_action is being called"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

# Check available resolve_action methods
print("=== Checking resolve_action bindings ===")

# Check GameInstance
game_inst_class = dm.GameInstance
print(f"dm.GameInstance: {game_inst_class}")
print(f"Has resolve_action: {hasattr(game_inst_class, 'resolve_action')}")

# Check if EffectResolver exists
if hasattr(dm, 'EffectResolver'):
    eff_res_class = dm.EffectResolver  
    print(f"\ndm.EffectResolver: {eff_res_class}")
    print(f"Has resolve_action: {hasattr(eff_res_class, 'resolve_action')}")

# Create instance and check
card_db = dm.JsonLoader.load_cards('data/cards.json')
game = dm.GameInstance(99, card_db)

print(f"\ngame instance type: {type(game)}")
print(f"game.resolve_action: {game.resolve_action}")
print(f"Method type: {type(game.resolve_action)}")

# Try calling with fake action
action = dm.Action()
action.type = dm.PlayerIntent.PASS  
action.slot_index = 0
action.source_instance_id = -1

print("\n=++ Calling game.resolve_action(PASS) ===")
game.resolve_action(action)
print("Completed")
