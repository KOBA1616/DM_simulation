#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

print("=== Testing up_to text generation ===\n")

# Test 1: MOVE_CARD without up_to
action = {'type': 'MOVE_CARD', 'destination_zone': 'HAND', 'value1': 3, 'up_to': False}
result = CardTextGenerator._format_action(action, is_spell=False)
print(f'Test 1 - MOVE_CARD to HAND (up_to=False):\n  {result}\n')

# Test 2: MOVE_CARD with up_to
action = {'type': 'MOVE_CARD', 'destination_zone': 'HAND', 'value1': 3, 'up_to': True}
result = CardTextGenerator._format_action(action, is_spell=False)
print(f'Test 2 - MOVE_CARD to HAND (up_to=True):\n  {result}\n')

# Test 3: DISCARD with up_to
action = {'type': 'DISCARD', 'value1': 2, 'up_to': True}
result = CardTextGenerator._format_action(action, is_spell=False)
print(f'Test 3 - DISCARD (up_to=True):\n  {result}\n')

# Test 4: TRANSITION BATTLE->GRAVEYARD with up_to
action = {'type': 'TRANSITION', 'from_zone': 'BATTLE_ZONE', 'to_zone': 'GRAVEYARD', 'value1': 2, 'up_to': True}
result = CardTextGenerator._format_action(action, is_spell=False)
print(f'Test 4 - TRANSITION BATTLE->GRAVEYARD (up_to=True):\n  {result}\n')

# Test 5: REPLACE_CARD_MOVE with up_to
action = {'type': 'REPLACE_CARD_MOVE', 'destination_zone': 'BATTLE_ZONE', 'source_zone': 'GRAVEYARD', 'value1': 1, 'up_to': True}
result = CardTextGenerator._format_action(action, is_spell=False)
print(f'Test 5 - REPLACE_CARD_MOVE (up_to=True):\n  {result}\n')

print("=== All tests completed ===")
