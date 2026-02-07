# Phase 1: AI選択ロジック統一 - テストスクリプト

import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module
    print("✓ dm_ai_module インポート成功")
    
    # GameInstance作成
    db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    gi = dm_ai_module.GameInstance(42, db)
    print("✓ GameInstance作成成功")
    
    # ゲーム初期化
    gi.state.setup_test_duel()
    deck = [1,2,3,4,5,6,7,8,9,10]*4
    gi.state.set_deck(0, deck)
    gi.state.set_deck(1, deck)
    print("✓ デッキ設定完了")
    
    # ゲーム開始
    dm_ai_module.PhaseManager.start_game(gi.state, db)
    dm_ai_module.PhaseManager.fast_forward(gi.state, db)
    print(f"✓ ゲーム開始完了 (Phase: {gi.state.current_phase}, Turn: {gi.state.turn_number})")
    
    # step()実行テスト（SimpleAI使用）
    print("\n=== SimpleAI動作テスト ===")
    for i in range(5):
        success = gi.step()
        if not success:
            print(f"  Step {i+1}: 失敗またはゲーム終了")
            break
        print(f"  Step {i+1}: 成功 (Phase: {gi.state.current_phase}, Turn: {gi.state.turn_number})")
    
    print("\n✅ Phase 1実装テスト完了")
    
except Exception as e:
    print(f"\n❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
