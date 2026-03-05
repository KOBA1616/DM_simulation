from dm_toolkit.gui.editor.text_generator import CardTextGenerator
import json
import sys
p='data/cards.json'
with open(p, encoding='utf-8') as f:
    cards = json.load(f)
card = next((c for c in cards if c.get('id')==1), None)
if not card:
    print('Card id=1 not found', file=sys.stderr); sys.exit(1)
print('--- JSON CARD (id=1) ---')
print(json.dumps(card, ensure_ascii=False, indent=2))
print('\n--- GENERATED TEXT ---')
print(CardTextGenerator.generate_text(card))
