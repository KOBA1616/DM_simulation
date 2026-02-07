#!/usr/bin/env python
"""
Debug: Check card properties from hand
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

# Get game state
gs = session.gs

print("=== Card Properties from Hand ===\n")

# Check P0's hand
p0 = gs.players[0]
print(f"Player 0 Hand ({len(p0.hand)} cards):")
for i, card in enumerate(p0.hand):
    print(f"  H[{i}]: card_id={card.card_id}")
    # Try to inspect card properties
    try:
        print(f"    name: {card.name}")
    except:
        print(f"    name: (no attribute)")
    try:
        print(f"    card_type: {card.card_type}")
    except:
        print(f"    card_type: (no attribute)")
    try:
        print(f"    power: {card.power}")
    except:
        print(f"    power: (no attribute)")
    try:
        print(f"    attributes: {dir(card)}")
    except:
        pass
    print()

print("\nStep 1 game action:")
try:
    session.step_game()
except Exception as e:
    print(f"Error: {e}")

print(f"\nAfter step: P0 Hand={len(p0.hand)} BZ={len(p0.battle_zone)}")

