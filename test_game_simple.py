#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Game Completion Test
勝敗がちゃんと記録されるか確認
"""

import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found")
    sys.exit(1)

def main():
    print("="*60)
    print("SIMPLE GAME COMPLETION TEST")
    print("="*60)
    
    # Load card database
    print("\n0. Loading card database...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    print(f"   ✓ Loaded cards")
    
    # Create game instance
    print("\n1. Creating game instance...")
    instance = dm_ai_module.GameInstance(42, card_db)
    gs = instance.state
    print(f"   ✓ Created with GameResult enum:")
    print(f"   NONE={dm_ai_module.GameResult.NONE}")
    print(f"   P1_WIN={dm_ai_module.GameResult.P1_WIN}")
    print(f"   P2_WIN={dm_ai_module.GameResult.P2_WIN}")
    print(f"   DRAW={dm_ai_module.GameResult.DRAW}")
    
    # Test loop detection
    print("\n2. Testing loop detection...")
    print(f"   Initial winner: {gs.winner} (should be NONE)")
    
    # Trigger loop detection
    print(f"   Triggering loop detection...")
    dm_ai_module.DevTools.trigger_loop_detection(gs)
    print(f"   ✓ Loop detection triggered")
    
    # Check game over
    print("\n3. Checking game over after loop detection...")
    try:
        # check_game_over takes (GameState, GameResult) and returns bool
        # We need to pass a GameResult reference
        result = dm_ai_module.GameResult.DRAW
        is_over = dm_ai_module.PhaseManager.check_game_over(gs, result)
        print(f"   Is over: {is_over}")
        print(f"   GameState.winner: {gs.winner} (type: {type(gs.winner)})")
        print(f"   GameState.loop_proven: {gs.loop_proven}")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with DataCollector
    print(f"\n4. Testing DataCollector...")
    try:
        collector = dm_ai_module.DataCollector()
        print(f"   ✓ DataCollector created")
        
        batch = collector.collect_data_batch_heuristic(1, True, False)
        print(f"   ✓ Collected 1 episode")
        print(f"   Token states: {len(batch.token_states)}")
        print(f"   Policies: {len(batch.policies)}")
        print(f"   Values: {batch.values}")
        
        # Check values
        if batch.values:
            print(f"   Value results (should be -1, 0, or 1):")
            for i, v in enumerate(batch.values[:5]):
                print(f"     Sample {i}: {v}")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()
