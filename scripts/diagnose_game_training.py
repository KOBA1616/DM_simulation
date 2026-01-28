#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game Completion and Training Diagnostic
ゲーム終局・訓練診断スクリプト
"""

import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module
except ImportError as e:
    print(f"❌ dm_ai_module not found: {e}")
    sys.exit(1)

def check_game_termination():
    """Check if games terminate properly"""
    print("=" * 80)
    print("1. ゲーム終局テスト")
    print("=" * 80)
    
    try:
        # Load cards
        print("Loading cards...")
        dm_ai_module.JsonLoader.load_cards("data/cards.json")
        
        # Create game state
        gs = dm_ai_module.GameState(42)
        gs.setup_test_duel()
        
        # Set magic.json deck
        import json
        with open("data/decks/magic.json", 'r') as f:
            deck = json.load(f)
        gs.set_deck(0, deck)
        gs.set_deck(1, deck)
        
        # Start game
        native_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        if hasattr(dm_ai_module, 'PhaseManager'):
            dm_ai_module.PhaseManager.start_game(gs, native_db)
        
        print(f"  初期状態:")
        print(f"    - Turn: {gs.turn_number}")
        print(f"    - Active Player: {gs.active_player_id}")
        print(f"    - Game Over: {gs.game_over}")
        print(f"    - P0 Shields: {len(gs.players[0].shield_zone)}")
        print(f"    - P1 Shields: {len(gs.players[1].shield_zone)}")
        
        # Simulate game
        max_turns = 50
        turn = 0
        
        for turn in range(max_turns):
            if gs.game_over:
                print(f"\n  ✓ ゲーム終局: ターン {turn + 1}")
                print(f"    - Winner: {gs.winner}")
                print(f"    - Turn Number: {gs.turn_number}")
                return True
            
            # Try to advance game
            try:
                dm_ai_module.PhaseManager.next_phase(gs, native_db)
            except Exception as e:
                print(f"  ⚠ Phase advancement error at turn {turn}: {e}")
                break
            
            if turn % 10 == 0 and turn > 0:
                print(f"  ... turn {turn}: Status={gs.status}, Winner={gs.winner}")
        
        print(f"\n  ⚠ ゲーム未終局: {max_turns} ターン経過後も game_over=False")
        print(f"    - Status: {gs.status}")
        print(f"    - Winner: {gs.winner}")
        print(f"    - Turn Number: {gs.turn_number}")
        return False
        
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_collector():
    """Check if DataCollector works properly"""
    print("\n" + "=" * 80)
    print("2. DataCollector テスト")
    print("=" * 80)
    
    try:
        # Load cards
        print("Loading cards...")
        dm_ai_module.JsonLoader.load_cards("data/cards.json")
        
        print("Creating DataCollector...")
        collector = dm_ai_module.DataCollector()
        
        print("Collecting 1 episode...")
        batch = collector.collect_data_batch_heuristic(1, True, False)
        
        print(f"  ✓ DataCollector working:")
        print(f"    - Samples collected: {len(batch.token_states)}")
        print(f"    - Token states shape: {len(batch.token_states[0]) if batch.token_states else 'empty'}")
        print(f"    - Policies shape: {len(batch.policies[0]) if batch.policies else 'empty'}")
        print(f"    - Values shape: {len(batch.values) if batch.values else 'empty'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_training_setup():
    """Check training setup"""
    print("\n" + "=" * 80)
    print("3. 訓練設定チェック")
    print("=" * 80)
    
    try:
        import yaml
        
        # Load training config
        with open("config/train_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("訓練設定:")
        print(f"  - Batch Size: {config['training']['batch_size']}")
        print(f"  - Learning Rate: {config['training']['learning_rate']}")
        print(f"  - Epochs: {config['training']['epochs']}")
        print(f"  - Games per Iteration: {config['training']['games_per_iteration']}")
        print(f"  - Iterations: {config['training']['iterations']}")
        
        # Check data file
        import os
        data_path = "data/transformer_training_data.npz"
        if os.path.exists(data_path):
            size_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"\n  ✓ 訓練データ存在: {data_path} ({size_mb:.2f} MB)")
        else:
            print(f"\n  ⚠ 訓練データ未生成: {data_path}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ 設定読み込みエラー: {e}")
        return False

def main():
    print("\n")
    print("=" * 80)
    print("  GAME COMPLETION & TRAINING DIAGNOSTIC")
    print("=" * 80)
    
    results = []
    
    # Run diagnostics
    results.append(("ゲーム終局判定", check_game_termination()))
    results.append(("DataCollector", check_data_collector()))
    results.append(("訓練設定", check_training_setup()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 診断結果サマリー")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ OK" if result else "❌ NG"
        print(f"  {status}: {name}")
    
    all_pass = all(r for _, r in results)
    
    if all_pass:
        print("\n✅ すべての診断に合格しました")
        print("\n推奨アクション:")
        print("  1. python training/generate_training_data.py --samples 1000")
        print("  2. python training/train_transformer_phase4.py --epochs 5")
    else:
        print("\n❌ いくつかの診断に失敗しました")
        print("\n修正が必要な項目:")
        for name, result in results:
            if not result:
                print(f"  - {name}")

if __name__ == "__main__":
    main()
