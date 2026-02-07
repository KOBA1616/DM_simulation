#!/usr/bin/env python
"""
Debug why ATTACK_CREATURE actions are not generated.
"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== ATTACK_CREATURE Generation Debug ===\n")

card_db = dm.JsonLoader.load_cards('data/cards.json')
gs = dm.GameState(42)
gs.setup_test_duel()

# Set up attack phase
gs.current_phase = dm.Phase.ATTACK
gs.active_player_id = 0
gs.turn_number = 2

p0 = gs.players[0]
p1 = gs.players[1]

# Add attacker (untapped, no summoning sickness)
attacker = dm.CardInstance()
attacker.card_id = 1
attacker.instance_id = 1001
attacker.is_tapped = False
attacker.summoning_sickness = False
attacker.turn_played = 1
p0.battle_zone.append(attacker)

# Add defender - TAPPED (required for creature attack)
defender = dm.CardInstance()
defender.card_id = 1
defender.instance_id = 2001
defender.is_tapped = True  # IMPORTANT: Must be tapped
defender.summoning_sickness = False  
defender.turn_played = 1
p1.battle_zone.append(defender)

print(f"Attacker: untapped={not attacker.is_tapped}, sick={attacker.summoning_sickness}")
print(f"Defender: tapped={defender.is_tapped}")

# Check can_attack conditions
print(f"\nCard 1 in card_db: {1 in card_db}")
if 1 in card_db:
    card_def = card_db[1]
    # Check if can_attack_creature would return true
    # (We can't call the function directly from Python, but we can check the card properties)
    print(f"Card type: {card_def.type}")
    print(f"Card has SPEED_ATTACKER: {getattr(card_def.keywords, 'speed_attacker', False)}")
    print(f"Card is EVOLUTION: {getattr(card_def.keywords, 'evolution', False)}")
    print(f"Card has MACH_FIGHTER: {getattr(card_def.keywords, 'mach_fighter', False)}")

# Generate actions
actions = dm.IntentGenerator.generate_legal_actions(gs, card_db)

print(f"\nGenerated {len(actions)} actions:")
for i, a in enumerate(actions):
    atype = int(a.type)
    if atype == 0:
        print(f"  {i}: PASS (type=0)")
    elif atype == 5:
        src = getattr(a, 'source_instance_id', -1)
        target_p = getattr(a, 'target_player', -1)
        print(f"  {i}: ATTACK_PLAYER (type=5) source={src} target_player={target_p}")
    elif atype == 6:
        src = getattr(a, 'source_instance_id', -1)
        tgt = getattr(a, 'target_instance_id', -1)
        print(f"  {i}: ATTACK_CREATURE (type=6) source={src} target={tgt}")
    else:
        print(f"  {i}: OTHER (type={atype})")

if not any(int(a.type) == 6 for a in actions):
    print("\n[ERROR] No ATTACK_CREATURE actions generated!")
    print("Possible causes:")
    print("  1. can_attack_creature() returned false")
    print("  2. No tapped enemy creatures found")
    print("  3. CANNOT_ATTACK passive restriction applied")
    print("  4. Enemy battle zone is empty")
    
    print(f"\nActual enemy battle zone size: {len(p1.battle_zone)}")
    if len(p1.battle_zone) > 0:
        for i, c in enumerate(p1.battle_zone):
            print(f"  Enemy creature {i}: is_tapped={c.is_tapped}, instance_id={c.instance_id}")
