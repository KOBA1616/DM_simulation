"""Test if C++ generates PLAY actions in MAIN_PHASE."""
import sys
sys.path.insert(0, '.')
import dm_ai_module

# Create game
seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)
gs = game_instance.state
gs.setup_test_duel()

deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

# Start game
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f"Initial phase: {gs.current_phase}")
print(f"P0 hand: {[c.card_id for c in gs.players[0].hand]}")
print(f"P0 mana: {len(gs.players[0].mana_zone)}")

# Do MANA_CHARGE for first 3 turns
for turn in range(1, 4):
    print(f"\n=== Turn {turn} ===")
    from dm_toolkit import commands_v2 as commands
    actions = commands.generate_legal_commands(gs, card_db, strict=False)
    print(f"Phase: {gs.current_phase}, Actions: {len(actions)}")
    
    # Find MANA_CHARGE action
    mana_action = None
    for a in actions:
        if int(a.type) == 1:  # MANA_CHARGE
            mana_action = a
            break
    
    if mana_action:
        print(f"Executing MANA_CHARGE card_id={mana_action.card_id}")
        game_instance.resolve_action(mana_action)
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        
        print(f"After MANA_CHARGE: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
        
        # Check MAIN_PHASE actions
        if gs.current_phase == dm_ai_module.Phase.MAIN:
            main_actions = commands.generate_legal_commands(gs, card_db, strict=False)
            print(f"MAIN_PHASE: {len(main_actions) if main_actions is not None else 0} actions available")
            
            # Show first few actions
            for i, a in enumerate(main_actions[:5]):
                print(f"  Action {i}: type={a.type} card_id={a.card_id}")
            
            # Try to play a creature if available
            play_action = None
            for a in main_actions:
                if int(a.type) == 15:  # PLAY_CREATURE/SPELL
                    play_action = a
                    break
            
            if play_action:
                print(f"Playing card_id={play_action.card_id}")
                game_instance.resolve_action(play_action)
                dm_ai_module.PhaseManager.fast_forward(gs, card_db)
                print(f"After PLAY: battle_zone={len(gs.players[0].battle_zone)}")
    
    # Fast-forward to next turn
    dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f"\n=== Final State ===")
print(f"P0 mana: {len(gs.players[0].mana_zone)}")
print(f"P0 battle_zone: {len(gs.players[0].battle_zone)}")
print(f"P0 hand: {len(gs.players[0].hand)}")
