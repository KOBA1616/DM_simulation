"""Simple script to print reaction mapping outputs for quick verification."""
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

cases = [
    {"type": "COUNTER_ATTACK", "cost": 2},
    {"type": "SHIELD_TRIGGER"},
    {"type": "RETURN_ATTACK", "cost": 1},
    {"type": "ON_DEFEND"},
]

for c in cases:
    print(c.get('type'), '->', CardTextGenerator._format_reaction(c))
