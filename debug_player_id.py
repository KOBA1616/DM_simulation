#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

gs = dm.GameState(42)
gs.setup_test_duel()

print("=== Player ID Check ===\n")
print(f"Number of players: {len(gs.players)}")

for i, p in enumerate(gs.players):
    print(f"Player {i}:")
    print(f"  Object: {p}")
    print(f"  ID: {p.id if hasattr(p, 'id') else 'NO ID ATTR'}")

print(f"\nActive player ID: {gs.active_player_id}")
print(f"Opponent player index: {1 - gs.active_player_id}")
print(f"Opponent player object: {gs.players[1 - gs.active_player_id]}")

opponent = gs.players[1 - gs.active_player_id]
opp_id = opponent.id if hasattr(opponent, 'id') else 'NO ID ATTR'
print(f"Opponent ID: {opp_id}")
