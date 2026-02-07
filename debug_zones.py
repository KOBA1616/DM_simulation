#!/usr/bin/env python
"""
Debug: Check all zones contents and creature status
"""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession

class Logger:
    def log_message(self, msg):
        pass

session = GameSession(
    callback_log=Logger().log_message, 
    callback_update_ui=lambda: None, 
    callback_action_executed=None
)
session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

print("=== Zone Debug ===\n")

# Check initial state before any actions
gs = session.gs

print("=== INITIAL STATE ===")
for player_id in [0, 1]:
    p = gs.players[player_id]
    print(f"\nPlayer {player_id}:")
    print(f"  Deck: {len(p.deck)} cards")
    if len(p.deck) > 0:
        print(f"    Top card: id={p.deck[0].card_id}")
    
    print(f"  Hand: {len(p.hand)} cards")
    for i, c in enumerate(p.hand):
        print(f"    H[{i}]: card_id={c.card_id}")
    
    print(f"  Battle zone: {len(p.battle_zone)} creatures")
    print(f"  Graveyard: {len(p.graveyard)} cards")
    print(f"  Mana zone: {len(p.mana_zone)} cards")

print("\n=== STEPPING THROUGH 5 ACTIONS ===\n")

for step in range(5):
    gs = session.gs
    phase = int(gs.current_phase)
    phase_names = {0: "START", 1: "DRAW", 2: "MANA", 3: "MAIN", 4: "ATTACK", 5: "BLOCK", 6: "END"}
    phase_name = phase_names.get(phase, f"UNKNOWN({phase})")
    
    print(f"Step {step}: Turn={gs.turn_number} Phase={phase_name} Active={gs.active_player_id}")
    
    # Show zones
    for player_id in [0, 1]:
        p = gs.players[player_id]
        print(f"  P{player_id} Hand={len(p.hand)} BZ={len(p.battle_zone)} GY={len(p.graveyard)} MZ={len(p.mana_zone)}")
    
    try:
        session.step_game()
    except Exception as e:
        print(f"Error: {e}")
        break
    
    if gs.game_over:
        break

print("\nDone")
