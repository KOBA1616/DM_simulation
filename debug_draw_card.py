#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug test for DRAW_CARD optional functionality"""
import sys
sys.path.insert(0, '.')

import dm_ai_module as dm

# Load card database
card_db = dm.JsonLoader.load_cards('data/cards.json')

# Create game instance
game = dm.GameInstance(99, card_db)

# Setup scenario with Card ID=1 in hand
config = dm.ScenarioConfig()
config.my_hand_cards = [1]
config.my_mana_zone = [1, 1]  # 2 water mana
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN

print("=== Step 1: Play Card ID=1 ===")
print(f"Before PLAY_CARD, pending_effects: {len(game.state.pending_effects)}")

# Generate legal actions and play the card
from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(game.state, card_db, strict=False)
declare_play_actions = [a for a in actions if int(a.type) == 15]  # DECLARE_PLAY

if declare_play_actions:
    game.resolve_action(declare_play_actions[0])
    print(f"After PLAY_CARD, pending_effects: {len(game.state.pending_effects)}")
    
    if game.state.pending_effects:
        pe = game.state.pending_effects[0]
        pe_type = getattr(pe, 'type', pe.get('type', None) if isinstance(pe, dict) else None)
        src_id = getattr(pe, 'source_instance_id', pe.get('source_instance_id', None) if isinstance(pe, dict) else None)
        is_opt = getattr(pe, 'optional', pe.get('optional', None) if isinstance(pe, dict) else None)
        print(f"  Effect type: {pe_type}")
        print(f"  Source instance: {src_id}")
        print(f"  Optional: {is_opt}")

print("\n=== Step 2: Generate actions for pending effect ===")
actions = commands.generate_legal_commands(game.state, card_db, strict=False)
pass_actions = [a for a in actions if a.type == dm.PlayerIntent.PASS]
resolve_actions = [a for a in actions if a.type == dm.PlayerIntent.RESOLVE_EFFECT]

print(f"Total actions: {len(actions)}")
print(f"PASS actions: {len(pass_actions)}")
print(f"RESOLVE_EFFECT actions: {len(resolve_actions)}")

print("\n=== Step 3: Execute RESOLVE_EFFECT ===")
if resolve_actions:
    resolve_action = resolve_actions[0]
    print(f"Executing RESOLVE_EFFECT")
    print(f"  Action type (enum value): {resolve_action.type}")
    print(f"  Slot index: {resolve_action.slot_index}")
    print(f"  Source instance: {resolve_action.source_instance_id}")
    
    # CRITICAL DIAGNOSTIC: Test if C++ is being called
    import os
    test_marker = "C:\\temp\\BEFORE_CPP_CALL.txt"
    with open(test_marker, "w") as f:
        f.write("About to call game.resolve_action()\n")
    
    game.resolve_action(resolve_action)
    
    with open(test_marker, "a") as f:
        f.write("Returned from game.resolve_action()\n")
        
    print(f"After RESOLVE_EFFECT, pending_effects: {len(game.state.pending_effects)}")
    print(f"Game state - waiting_for_user_input: {game.state.waiting_for_user_input}")
    
    if game.state.pending_query:
        print(f"Pending query: {game.state.pending_query.query_type}")
        print(f"  Min: {game.state.pending_query.params.get('min', 'N/A')}")
        print(f"  Max: {game.state.pending_query.params.get('max', 'N/A')}")
    else:
        print("No pending query")

print("\n=== Step 4: Check game state ===")
player = game.state.players[0]
print(f"Hand: {len(player.hand)} cards")
print(f"Deck: {len(player.deck)} cards")
print(f"Battle zone: {len(player.battle_zone)} cards")
print(f"Mana zone: {len(player.mana_zone)} cards")
print(f"Graveyard: {len(player.graveyard)} cards")
