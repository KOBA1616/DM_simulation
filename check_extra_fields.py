import json

with open('data/cards.json', encoding='utf-8') as f:
    cards = json.load(f)

print("Checking for 'extra_fields' in conditions...")
for i, card in enumerate(cards):
    for effect in card.get('effects', []):
        condition = effect.get('condition', {})
        if 'extra_fields' in condition:
            print(f"Card {i} (ID {card['id']}, {card['name']}): has extra_fields in condition")
            print(f"  Condition: {condition}")

print("\nChecking for 'extra_fields' in commands...")
for i, card in enumerate(cards):
    for effect in card.get('effects', []):
        for cmd in effect.get('commands', []):
            if 'extra_fields' in cmd:
                print(f"Card {i} (ID {card['id']}, {card['name']}): has extra_fields in command")
                print(f"  Command: {cmd['type']}")
