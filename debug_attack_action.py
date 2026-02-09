#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm
import json

print("=== Attack Action Debug ===\n")

# Load card DB
card_db = dm.JsonLoader.load_cards('data/cards.json')
print(f'Card DB loaded: {len(card_db)} cards')

# Create game state
gs = dm.GameState(42)
gs.setup_test_duel()
gs.current_phase = dm.Phase.ATTACK
gs.active_player_id = 0
gs.turn_number = 2

p0 = gs.players[0]
p1 = gs.players[1]

# Add attacker
test_card_id = 1
c_attacker = dm.CardInstance()
c_attacker.card_id = test_card_id
c_attacker.instance_id = 1001
c_attacker.is_tapped = False
c_attacker.summoning_sickness = False
c_attacker.turn_played = 1
p0.battle_zone.append(c_attacker)

# Add defender
c_enemy = dm.CardInstance()
c_enemy.card_id = 1
c_enemy.instance_id = 2001
c_enemy.is_tapped = True
c_enemy.summoning_sickness = False
c_enemy.turn_played = 1
p1.battle_zone.append(c_enemy)

print(f'Phase: ATTACK, Active Player: 0, Turn: {gs.turn_number}')
print(f'My battle zone: {len(p0.battle_zone)} creatures')
print(f'Enemy battle zone: {len(p1.battle_zone)} creatures')

# Check card in DB
print(f'\nCard {test_card_id} in card_db: {test_card_id in card_db}')
if test_card_id in card_db:
    card_def = card_db[test_card_id]
    print(f'  Card type: {card_def.type}')

# Generate actions
print('\n=== Generating Actions ===')
    from dm_toolkit import commands_v2 as commands
    actions = commands.generate_legal_commands(gs, card_db, strict=False)
print(f'Total actions: {len(actions)}')

attack_actions = []
pass_actions = []
other_actions = []

for a in actions:
    action_type = int(a.type)
    if action_type == 6:  # ATTACK_PLAYER
        attack_actions.append(('ATTACK_PLAYER', a))
    elif action_type == 7:  # ATTACK_CREATURE  
        attack_actions.append(('ATTACK_CREATURE', a))
    elif action_type == 0:  # PASS
        pass_actions.append(a)
    else:
        other_actions.append(('OTHER', action_type))

print(f'ATTACK_PLAYER: {sum(1 for t, a in attack_actions if t == "ATTACK_PLAYER")}')
print(f'ATTACK_CREATURE: {sum(1 for t, a in attack_actions if t == "ATTACK_CREATURE")}')
print(f'PASS: {len(pass_actions)}')
print(f'OTHER: {len(other_actions)}')

if not attack_actions and not pass_actions:
    print('\n[ERROR] No attack or pass actions generated!')
    print('Possible causes:')
    print('  1. card_db lookup failed')
    print('  2. can_attack_player/creature returned false')
    print('  3. Passive restriction applied')

# Check JSON
print('\n=== Checking JSON ===')
with open('data/cards.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)
    card = next((c for c in cards if c['id'] == 1), None)
    if card:
        print(f'JSON Card 1 found: type={card.get("type")}')
    else:
        print('Card 1 not found in JSON')
