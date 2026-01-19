#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test CAST_SPELL + REPLACE_CARD_MOVE text generation merge."""

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

# Test 1: CAST_SPELL only
cast_spell_cmd = {
    'type': 'CAST_SPELL',
    'target_group': 'SELF',
    'target_filter': {}
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

print("CAST_SPELL text:", texts[0])
print("REPLACE_CARD_MOVE text:", texts[1])

# Test merged text
merged = CardTextGenerator._merge_action_texts(commands, texts)
print("Merged text:", merged)
print()

# Expected: "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。"
expected = "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。"
print("Expected text:", expected)
print("Match:", merged == expected)