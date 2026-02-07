#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

gs = dm.GameState(42)
gs.setup_test_duel()

print("=== Player Attributes Check ===\n")
p = gs.players[0]

# List all attributes
attrs = [a for a in dir(p) if not a.startswith('_')]
print(f"Public attributes of Player: {attrs}\n")

# Check if id exists
if 'id' in attrs:
    print(f"Player 0 id: {p.id}")
    print(f"Player 1 id: {gs.players[1].id}")
else:
    print("ERROR: 'id' not found in Player attributes!")
    print("Expected to find: id, hand, mana_zone, battle_zone, etc.")
