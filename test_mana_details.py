#!/usr/bin/env python
"""
Test to check mana card properties and civilizations
"""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession
import json

class Logger:
    def log_message(self, msg):
        pass

# Load card data
with open("python/data/cards.json", "r", encoding="utf-8") as f:
    cards_data = json.load(f)

card_map = {c["id"]: c for c in cards_data}

session = GameSession(
    callback_log=Logger().log_message, 
    callback_update_ui=lambda: None, 
    callback_action_executed=None
)
session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

gs = session.gs

print("Initial hand contents:")
p0 = gs.players[0]
for i, c in enumerate(p0.hand):
    if c.card_id in card_map:
        card_meta = card_map[c.card_id]
        print(f"  H[{i}]: id={c.card_id} name={card_meta['name']}")
        print(f"         cost={card_meta['cost']} civilizations={card_meta['civilizations']}")
        print(f"         tapped={c.is_tapped} sick={c.summoning_sickness}")
    else:
        print(f"  H[{i}]: id={c.card_id} (not in card_map)")
print()

# Step once
session.step_game()

print("After Step 1:")
print(f"Mana Zone ({len(p0.mana_zone)} cards):")
for i, c in enumerate(p0.mana_zone):
    print(f"  M[{i}]: id={c.card_id} tapped={c.is_tapped} turns_played={c.turn_played}")
    if c.card_id in card_map:
        card_meta = card_map[c.card_id]
        print(f"         name={card_meta['name']} civilizations={card_meta['civilizations']}")
print()

print("Hand after Step 1:")
for i, c in enumerate(p0.hand):
    if c.card_id in card_map:
        card_meta = card_map[c.card_id]
        print(f"  H[{i}]: id={c.card_id} name={card_meta['name'][:30]}")
        print(f"         cost={card_meta['cost']} civilizations={card_meta['civilizations']}")
    else:
        print(f"  H[{i}]: id={c.card_id}")
