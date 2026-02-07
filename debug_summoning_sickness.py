#!/usr/bin/env python
"""
Debug summoning sickness handling.
"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== Summoning Sickness Debug ===\n")

card_db = dm.JsonLoader.load_cards('data/cards.json')
gs = dm.GameState(42)
gs.setup_test_duel()

p0 = gs.players[0]

# Simulate: add a creature via effect in MAIN phase (should have summoning_sickness=true)
c_new = dm.CardInstance()
c_new.card_id = 1
c_new.instance_id = 3001
c_new.is_tapped = False
c_new.summoning_sickness = True  # New creatures should have this
c_new.turn_played = 1  # Played this turn
p0.battle_zone.append(c_new)

# Next turn starts
gs.active_player_id = 0  # Still player 0
gs.current_phase = dm.Phase.START_OF_TURN
gs.turn_number = 2

# Create another creature that was played last turn
c_old = dm.CardInstance()
c_old.card_id = 2
c_old.instance_id = 3002
c_old.is_tapped = False
c_old.summoning_sickness = True  # Incorrectly still has this
c_old.turn_played = 1  # Played turn 1
p0.battle_zone.clear()  # Clear for fresh start
p0.battle_zone.append(c_new)
p0.battle_zone.append(c_old)

print(f"Turn number: {gs.turn_number}")
print(f"Current phase: {gs.current_phase} (START_OF_TURN={dm.Phase.START_OF_TURN})")
print(f"\nBefore start_turn:")
print(f"  New creature (turn_played=1): summoning_sickness={c_new.summoning_sickness}")
print(f"  Old creature (turn_played=1): summoning_sickness={c_old.summoning_sickness}")

# Simulate start_turn
# NOTE: We can't directly call PhaseManager.start_turn from Python,
# but we can manually do what it should do
print(f"\nWhat should happen in start_turn:")
print(f"  1. Clear summoning_sickness for all creatures")
print(f"  2. Result: both creatures should have summoning_sickness=false")

# The ideal sequence would be to call PhaseManager.start_turn(gs, card_db)
# but that's not directly exposed via Python API

# Check can_attack_player for the new creature
print(f"\nCan new creature attack player?")
print(f"  Conditions checked by can_attack_player:")
print(f"    - is_tapped: {c_new.is_tapped} (should be False)")
print(f"    - summoning_sickness: {c_new.summoning_sickness} (should be False after start_turn)")
print(f"    - turn_played ({c_new.turn_played}) vs current_turn ({gs.turn_number})")
print(f"      - Same turn? {c_new.turn_played == gs.turn_number} (means has sickness)")
print(f"      - Needs SPEED_ATTACKER if summoning_sickness is true")
