#!/usr/bin/env python3
"""Test card ID 1 effect resolution"""

import dm_ai_module
from dm_toolkit.action_to_command import map_action
from dm_toolkit import commands_v2

# Prefer command-first wrapper
generate_legal_commands = commands_v2.generate_legal_commands

# Initialize game
print("Initializing game...")
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
gs = dm_ai_module.GameState(seed=12345, p0_deck=[1]*40, p1_deck=[2]*40)

# Helper function
def step_game(msg=""):
    if msg:
        print(f"\n{'='*50}")
        print(msg)
        print('='*50)
    cmds = generate_legal_commands(gs, card_db)
    print(f"Phase={gs.current_phase.name} Active={gs.active_player_id} Commands={len(cmds)}")
    if cmds:
        # Show first few command types
        types = [cmd.to_dict().get('type') for cmd in cmds[:5]]
        print(f"  Types: {types}")
        # Select command
        selected = None
        # Priority: RESOLVE_EFFECT > DECLARE_PLAY/PLAY_CARD > MANA_CHARGE > PASS
        for cmd in cmds:
            d = cmd.to_dict()
            if d.get('type') == 'RESOLVE_EFFECT':
                selected = cmd
                print(f"  >> SELECTING: RESOLVE_EFFECT (effect_index={d.get('effect_index')})")
                break
        if not selected:
            for cmd in cmds:
                d = cmd.to_dict()
                if d.get('type') in ['DECLARE_PLAY', 'PLAY_CARD']:
                    selected = cmd
                    print(f"  >> SELECTING: {d.get('type')} (card_id={d.get('card_id',d.get('instance_id'))})")
                    break
        if not selected:
            for cmd in cmds:
                d = cmd.to_dict()
                if d.get('type') == 'MANA_CHARGE':
                    selected = cmd
                    print(f"  >> SELECTING: MANA_CHARGE")
                    break
        if not selected:
            selected = cmds[0]
            print(f"  >> SELECTING: {cmds[0].to_dict().get('type')}")
        
        # Execute
        selected.execute(gs, card_db)
        
        # Fast forward
        try:
            dm_ai_module.PhaseManager.fast_forward_until_decision(gs, card_db,  max_steps=10)
        except:
            pass
    else:
        print("  No commands")
    
    # Check pending effects
    if gs.pending_effects:
        print(f"  Pending effects: {len(gs.pending_effects)}")
        for i, eff in enumerate(list(gs.pending_effects)[:3]):
            print(f"    #{i} type={eff.type.name if hasattr(eff.type, 'name') else eff.type} src={eff.source_instance_id}")

# Play through game until card ID 1 goes to battle zone
max_steps = 100
for step_num in range(max_steps):
    # Check if card ID 1 is in battle zone
    if gs.players[0].battle_zone:
        for c in gs.players[0].battle_zone:
            if c.card_id == 1:
                print(f"\n{'='*60}")
                print(f"CARD ID 1 IN BATTLE ZONE! Step {step_num}")
                print(f"{'='*60}")
                # Pending effects should exist
                if gs.pending_effects:
                    print(f"✓ Pending effects exist: {len(gs.pending_effects)}")
                    step_game(f"STEP {step_num+1}: Resolve pending effect")
                    # Check if effect was resolved (pending queue should be empty now)
                    if not gs.pending_effects:
                        print("\n✓✓✓ SUCCESS: Pending effect resolved!")
                    else:
                        print(f"\n✗✗✗ FAIL: Pending effect still exists ({len(gs.pending_effects)} effects)")
                else:
                    print("✗ No pending effects (trigger didn't fire?)")
                break
        else:
            continue
        break
    
    step_game(f"STEP {step_num}")

print(f"\n{'='*60}")
print("Test completed")
print(f"{'='*60}")
