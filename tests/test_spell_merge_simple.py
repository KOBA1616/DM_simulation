#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

# Test 1: CAST_SPELL only
cast_spell_cmd = {
    'type': 'CAST_SPELL',
}

# Test 2: REPLACE_CARD_MOVE only
replace_cmd = {
    'type': 'REPLACE_CARD_MOVE',
    'from_zone': 'GRAVEYARD',
    'to_zone': 'DECK_BOTTOM',
    'input_value_key': 'card_ref'
}

# Test 3: Sequence with both
commands = [cast_spell_cmd, replace_cmd]
texts = [CardTextGenerator._format_command(cmd) for cmd in commands]

print(f"Test results:")
print(f"1. CAST_SPELL text: {texts[0]}")
print(f"2. REPLACE_CARD_MOVE text: {texts[1]}")

# Test merged text
merged = CardTextGenerator._merge_action_texts(commands, texts)
print(f"3. Merged text: {merged}")
print()

# Expected: "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。"
expected = "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。"
print(f"Expected: {expected}")
print(f"Success: {merged == expected}")
