#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'bin\\Release')

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

print("Before PLAY_CARD, pending_effects:", len(game.state.pending_effects))

# Generate legal actions and play the card
actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
declare_play_actions = [a for a in actions if int(a.type) == 15]  # DECLARE_PLAY

if declare_play_actions:
    game.resolve_action(declare_play_actions[0])
    print("After PLAY_CARD, pending_effects:", len(game.state.pending_effects))
    
    if game.state.pending_effects:
        pe = game.state.pending_effects[0]
        # Support both native object with attributes and dict-like bindings
        pe_type = getattr(pe, 'type', None)
        if pe_type is None and isinstance(pe, dict):
            pe_type = pe.get('type')
        print(f"Type: {pe_type} == {dm.EffectType.TRIGGER_ABILITY} ? {pe_type == dm.EffectType.TRIGGER_ABILITY}")

# Generate actions for the pending effect
actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
pass_count = sum(1 for a in actions if a.type == dm.PlayerIntent.PASS)
resolve_count = sum(1 for a in actions if a.type == dm.PlayerIntent.RESOLVE_EFFECT)

print(f"Total actions: {len(actions)}")
print(f"PASS: {pass_count}")
print(f"RESOLVE_EFFECT: {resolve_count}")

if pass_count > 0:
    print("SUCCESS: PASS action generated!")
else:
    print("FAILURE: No PASS action")
