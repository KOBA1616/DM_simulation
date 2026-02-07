import json

# Load and check JSON structure
with open('data/cards.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)

print(f'Total cards in JSON: {len(cards)}')
print(f'\nFirst card structure:')
card1 = cards[0]
print(f'  id: {card1.get("id")}')
print(f'  name: {card1.get("name")}')
print(f'  type: {card1.get("type")} (type: {type(card1.get("type")).__name__})')
print(f'  cost: {card1.get("cost")} (type: {type(card1.get("cost")).__name__})')
print(f'  civilizations: {card1.get("civilizations")}')

# Check if there's a 'civilization' (singular) field
if 'civilization' in card1:
    print(f'  civilization (singular): {card1.get("civilization")} (type: {type(card1.get("civilization")).__name__})')

# Check several cards for type field
print(f'\nChecking first 5 cards for type field:')
for i, card in enumerate(cards[:5]):
    card_type = card.get('type')
    print(f'  Card {i+1} (ID {card.get("id")}): type="{card_type}" (Python type: {type(card_type).__name__})')
