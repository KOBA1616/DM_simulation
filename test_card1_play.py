#!/usr/bin/env python3
"""Direct test of Card ID=1 effect"""
import sys
sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')

import dm_ai_module as m

print(f"[TEST] dm_ai_module loaded successfully")
print(f"[TEST] Available classes: {dir(m)[:5]}...")

# Create a game state from scenario
scenario = m.Scenario()
game = m.GameInstance(scenario)

print(f"[TEST] GameInstance created")

# Get current game state
state = game.get_game_state()
print(f"[TEST] GameState obtained")
print(f"  players count: {len(state.players)}")
print(f"  player 0 hand size: {len(state.players[0].hand)}")

# Load card database
import json
with open(r'C:\Users\ichirou\DM_simulation\data\cards.json', 'r', encoding='utf-8') as f:
    cards_data = json.load(f)
    
card_1 = [c for c in cards_data if c['id'] == 1][0]
print(f"[TEST] Card ID=1 found: {card_1['name']}")

# Check effect structure
if card_1.get('effects'):
    effect = card_1['effects'][0]
    print(f"[TEST] Effect trigger: {effect.get('trigger')}")
    print(f"[TEST] Commands count: {len(effect.get('commands', []))}")
    if effect.get('commands'):
        for i, cmd in enumerate(effect['commands']):
            print(f"  [{i}] type={cmd['type']}, optional={cmd.get('optional')}, input_value_key={cmd.get('input_value_key')}")

print(f"[TEST] Complete")

