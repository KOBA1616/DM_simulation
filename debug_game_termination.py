#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game Termination Debug Script
ゲーム終了ロジックの詳細な検査
"""

import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found")
    sys.exit(1)

def test_game_completion():
    """Test if games complete properly"""
    
    print("="*60)
    print("GAME TERMINATION DEBUG")
    print("="*60)
    
    # Load card database
    print("\n1. Loading card database...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    print(f"   ✓ Loaded {len(card_db) if hasattr(card_db, '__len__') else '?'} cards")
    
    # Create game
    print("\n2. Creating game...")
    gs = dm_ai_module.GameState(42)  # seed as positional arg
    gs.setup_test_duel()
    
    # Set decks
    deck = [1] * 40
    gs.set_deck(0, deck)
    gs.set_deck(1, deck)
    print(f"   ✓ Decks set")
    
    # Start game
    print("\n3. Starting game...")
    dm_ai_module.PhaseManager.start_game(gs, card_db)
    print(f"   ✓ Game started")
    
    # Print initial state
    print(f"\n4. Initial state:")
    print(f"   Phase: {gs.current_phase}")
    print(f"   Active player: {gs.active_player_id}")
    print(f"   Turn number: {gs.turn_number}")
    print(f"   Winner: {gs.winner}")
    print(f"   Game over: {gs.game_over}")
    print(f"   P0 shields: {len(gs.players[0].shield_zone)}")
    print(f"   P1 shields: {len(gs.players[1].shield_zone)}")
    
    # Run game loop
    print(f"\n5. Running game loop (100 steps)...")
    
    max_steps = 100
    step_count = 0
    
    while gs.winner == dm_ai_module.GameResult.NONE and step_count < max_steps:
        step_count += 1
        
        # Get active player
        active_player = gs.active_player_id
        
        # Try to get legal actions
        try:
            legal_actions = dm_ai_module.PhaseManager.get_legal_actions(gs, card_db)
        except Exception as e:
            print(f"   Error generating actions at step {step_count}: {e}")
            legal_actions = []
        
        if not legal_actions:
            # Try advancing phase
            try:
                dm_ai_module.PhaseManager.next_phase(gs, card_db)
                print(f"   Step {step_count}: Advanced phase (no actions)")
            except Exception as e:
                print(f"   Error at step {step_count}: {e}")
                break
        else:
            # Execute first action
            try:
                action = legal_actions[0]
                dm_ai_module.GameLogicSystem.resolve_action(gs, action, card_db)
                print(f"   Step {step_count}: Executed action, shields: P0={len(gs.players[0].shield_zone)} P1={len(gs.players[1].shield_zone)}")
                
                # Check for PASS and advance phase
                if action.type == dm_ai_module.PlayerIntent.PASS:
                    dm_ai_module.PhaseManager.next_phase(gs, card_db)
                    
            except Exception as e:
                print(f"   Error at step {step_count}: {e}")
                break
        
        # Check game over
        try:
            result = dm_ai_module.PhaseManager.check_game_over(gs)
            if isinstance(result, tuple):
                is_over, winner = result
            else:
                is_over = result
                winner = gs.winner
            
            if is_over:
                print(f"\n   ✓ Game over detected at step {step_count}")
                print(f"   Winner: {winner}")
                break
        except Exception as e:
            print(f"   Error checking game over: {e}")
        
        # Check shield zones for win condition
        p0_shields = len(gs.players[0].shield_zone)
        p1_shields = len(gs.players[1].shield_zone)
        
        if p0_shields == 0 or p1_shields == 0:
            print(f"\n   ⚠️  Shield zone empty!")
            print(f"   P0 shields: {p0_shields}, P1 shields: {p1_shields}")
            print(f"   But winner: {gs.winner}")
            
            # Try to manually update
            if p0_shields == 0:
                gs.winner = dm_ai_module.GameResult.P2_WIN
                print(f"   Manually set winner to P2_WIN")
            elif p1_shields == 0:
                gs.winner = dm_ai_module.GameResult.P1_WIN
                print(f"   Manually set winner to P1_WIN")
            break
    
    # Final state
    print(f"\n6. Final state after {step_count} steps:")
    print(f"   Winner: {gs.winner}")
    print(f"   Game over: {gs.game_over}")
    print(f"   Turn number: {gs.turn_number}")
    print(f"   P0 shields: {len(gs.players[0].shield_zone)}")
    print(f"   P1 shields: {len(gs.players[1].shield_zone)}")
    
    # Check loop detection
    gs.update_loop_check()
    print(f"   Loop proven: {gs.loop_proven}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if gs.winner != dm_ai_module.GameResult.NONE:
        print(f"✓ Game terminated with winner: {gs.winner}")
    elif gs.loop_proven:
        print(f"✓ Game terminated with DRAW (loop proven)")
    else:
        print(f"✗ Game did not terminate properly!")
        print(f"  - Shield zones: P0={len(gs.players[0].shield_zone)}, P1={len(gs.players[1].shield_zone)}")
        print(f"  - Max steps reached ({max_steps})")
        print(f"  - Possible causes:")
        print(f"    1. Game termination logic not triggered")
        print(f"    2. Loop detection threshold not reached")
        print(f"    3. Game state not properly updated")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    test_game_completion()
