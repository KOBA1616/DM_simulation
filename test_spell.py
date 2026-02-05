#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""呪文をプレイした時の動作テスト"""
import sys
sys.path.insert(0, 'bin\\Release')

import dm_ai_module as dm

# Load card database
card_db = dm.JsonLoader.load_cards('data/cards.json')

# Find a spell card
spell_cards = [cid for cid, cdef in card_db.items() if cdef.type == dm.CardType.SPELL]
print(f"Found {len(spell_cards)} spell cards: {spell_cards[:10]}")

if not spell_cards:
    print("No spell cards found!")
    exit(1)

# Use first spell card for testing
test_spell_id = spell_cards[0]
spell_def = card_db[test_spell_id]
print(f"\nTesting spell: ID={test_spell_id}, Name={spell_def.name}")
print(f"Cost: {spell_def.cost}")
print(f"Effects: {len(spell_def.effects)}")

# Create game instance
game = dm.GameInstance(99, card_db)

# Setup scenario with spell in hand
config = dm.ScenarioConfig()
config.my_hand_cards = [test_spell_id]
config.my_mana_zone = [test_spell_id] * (spell_def.cost + 1)  # Enough mana
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN

print(f"\n=== Before playing spell ===")
print(f"pending_effects: {len(game.state.pending_effects)}")
print(f"Hand: {len(game.state.players[0].hand)} cards")
print(f"Mana: {len(game.state.players[0].mana_zone)} cards")

# Generate legal actions and play the spell
actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
declare_play_actions = [a for a in actions if int(a.type) == 15]  # DECLARE_PLAY

print(f"\nDECLARE_PLAY actions available: {len(declare_play_actions)}")

if declare_play_actions:
    print(f"Playing spell...")
    game.resolve_action(declare_play_actions[0])
    
    print(f"\n=== After playing spell ===")
    print(f"pending_effects: {len(game.state.pending_effects)}")
    
    if game.state.pending_effects:
        for i, pe in enumerate(game.state.pending_effects):
            print(f"\nPendingEffect {i}:")
            # Support both native object with attributes and dict-like bindings
            pe_type = getattr(pe, 'type', None)
            pe_resolve = getattr(pe, 'resolve_type', None)
            pe_optional = getattr(pe, 'optional', None)
            pe_source = getattr(pe, 'source_instance_id', None)
            if isinstance(pe, dict):
                pe_type = pe.get('type', pe_type)
                pe_resolve = pe.get('resolve_type', pe_resolve)
                pe_optional = pe.get('optional', pe_optional)
                pe_source = pe.get('source_instance_id', pe_source)

            print(f"  type: {pe_type} (TRIGGER_ABILITY={dm.EffectType.TRIGGER_ABILITY})")
            print(f"  resolve_type: {pe_resolve}")
            print(f"  optional: {pe_optional}")
            print(f"  source_instance_id: {pe_source}")
    
    # Generate actions for the pending effect
    print(f"\n=== Generating actions for pending effects ===")
    actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
    pass_count = sum(1 for a in actions if a.type == dm.PlayerIntent.PASS)
    resolve_count = sum(1 for a in actions if a.type == dm.PlayerIntent.RESOLVE_EFFECT)
    
    print(f"Total actions: {len(actions)}")
    print(f"PASS actions: {pass_count}")
    print(f"RESOLVE_EFFECT actions: {resolve_count}")
    
    if pass_count > 0:
        print("\n✓ SUCCESS: PASS action generated for spell effect")
    else:
        print("\n✗ FAILURE: No PASS action for spell effect")
        print("\nAvailable action types:")
        for a in actions[:10]:
            print(f"  - {a.type}")
else:
    print("No DECLARE_PLAY actions available - spell cannot be played")
