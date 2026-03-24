import json
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

with open('data/cards.json','r',encoding='utf-8') as f:
    data = json.load(f)
if isinstance(data, list):
    cards = {c['id']:c for c in data}
else:
    cards = data
c = cards.get(17)
print('card id 17 static_abilities:', c.get('static_abilities'))
print('\nGenerated body:')
print('\n'.join(CardTextGenerator.generate_body_text_lines(c)))
