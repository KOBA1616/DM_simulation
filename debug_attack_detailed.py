#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== Detailed Attack Generation Debug ===\n")

card_db = dm.JsonLoader.load_cards('data/cards.json')
gs = dm.GameState(42)
gs.setup_test_duel()
gs.current_phase = dm.Phase.ATTACK
gs.active_player_id = 0
gs.turn_number = 2

p0 = gs.players[0]
p1 = gs.players[1]

# Add attacker
c_attacker = dm.CardInstance()
c_attacker.card_id = 1
c_attacker.instance_id = 1001
c_attacker.is_tapped = False
c_attacker.summoning_sickness = False
c_attacker.turn_played = 1
p0.battle_zone.append(c_attacker)

# Add defender (tapped)
c_enemy = dm.CardInstance()
c_enemy.card_id = 1
c_enemy.instance_id = 2001
c_enemy.is_tapped = True
c_enemy.summoning_sickness = False
c_enemy.turn_played = 1
p1.battle_zone.append(c_enemy)

print(f'Configuration:')
print(f'  Phase: {gs.current_phase} (ATTACK={dm.Phase.ATTACK})')
print(f'  Active Player: {gs.active_player_id}')
print(f'  Turn: {gs.turn_number}')

print(f'\nAttacker (P0 Creature):')
print(f'  Card ID: {c_attacker.card_id}')
print(f'  Instance ID: {c_attacker.instance_id}')
print(f'  Is Tapped: {c_attacker.is_tapped}')
print(f'  Has Summoning Sickness: {c_attacker.summoning_sickness}')
print(f'  Turn Played: {c_attacker.turn_played}')

print(f'\nDefender (P1 Creature):')
print(f'  Card ID: {c_enemy.card_id}')
print(f'  Instance ID: {c_enemy.instance_id}')
print(f'  Is Tapped: {c_enemy.is_tapped}')
print(f'  Turn Played: {c_enemy.turn_played}')

from dm_toolkit import commands_v2 as commands
# Generate commands with details
print(f'\n=== Command Generation ===')
actions = commands.generate_legal_commands(gs, card_db, strict=False)
print(f'Total commands: {len(actions) if actions is not None else 0}')

for i, a in enumerate((actions or [])):
    atype = int(a.type) if str(getattr(a, 'type', '')).isdigit() else getattr(a, 'type', None)
    if atype == 0:
        print(f'  {i}: PASS')
    elif atype == 6:
        src_iid = getattr(a, 'source_instance_id', -1)
        tgt_player = getattr(a, 'target_player', -1)
        print(f'  {i}: ATTACK_PLAYER (source_iid={src_iid}, target_player={tgt_player})')
    elif atype == 7:
        src_iid = getattr(a, 'source_instance_id', -1)
        tgt_iid = getattr(a, 'target_instance_id', -1)
        print(f'  {i}: ATTACK_CREATURE (source_iid={src_iid}, target_iid={tgt_iid})')
    else:
        print(f'  {i}: OTHER(type={atype})')

# Manual check: can the creature attack?
print(f'\n=== Manual Attack Check ===')
card_def = card_db[1]
print(f'Card 1 type: {card_def.type}')
print(f'Card 1 keywords.evolution: {card_def.keywords.evolution}')
print(f'Card 1 keywords.speed_attacker: {card_def.keywords.speed_attacker}')

# Check if enemy has tapped creatures
print(f'\nEnemy has {len(p1.battle_zone)} creatures')
print(f'Enemy creatures tapped status:')
for c in p1.battle_zone:
    print(f'  Creature {c.instance_id}: tapped={c.is_tapped}')
