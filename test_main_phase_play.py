"""メインフェイズのカードプレイテスト"""
import dm_ai_module as dm

# ゲーム作成（デフォルトのカードセット使用）
print("Initializing game...")
game = dm.GameInstance(12345)

# ゲーム開始
state = game.state
print(f"Game initialized. Turn: {state.turn_number}, Phase: {int(state.current_phase)}")

# プレイヤー0のターン、メインフェイズまで進める
for step in range(20):  # 最大20ステップ
    actions = game.get_legal_actions_shallow()
    if not actions:
        print("No actions available, game over")
        break
    
    state = game.state
    print(f"\nStep {step}: Phase={int(state.current_phase)}, Actions={len(actions)}")
    
    # メインフェイズに到達したか確認
    if state.current_phase == dm.Phase.MAIN:
        print("\n=== Reached MAIN phase! ===")
        break
        
    # PASSアクションまたは最初のアクションを実行
    action = actions[0]
    game.step(action)

# メインフェイズでのアクション確認
state = game.state
if state.current_phase == dm.Phase.MAIN:
    actions = game.get_legal_actions_shallow()
    print(f"\nMAIN phase actions: {len(actions)}")
    
    # PLAY_CARDアクションを探す
    play_actions = [a for a in actions if a.type == dm.PlayerIntent.PLAY_CARD]
    print(f"PLAY_CARD actions available: {len(play_actions)}")
    
    if play_actions:
        print("\n=== Attempting to play a card ===")
        play_action = play_actions[0]
        print(f"Playing card instance ID: {play_action.source_instance_id}")
        
        # プレイ前の状態
        before_state = game.state
        player_id = before_state.active_player_id
        player_before = before_state.players[player_id]
        print(f"\nBefore PLAY_CARD:")
        print(f"  Hand: {len(player_before.hand)}")
        print(f"  Battle zone: {len(player_before.battle_zone)}")
        print(f"  Graveyard: {len(player_before.graveyard)}")
        print(f"  Stack: {len(player_before.stack)}")
        
        # カードをプレイ
        try:
            game.step(play_action)
            
            # 結果確認
            after_state = game.state
            player_after = after_state.players[player_id]
            print(f"\nAfter PLAY_CARD:")
            print(f"  Hand: {len(player_after.hand)}")
            print(f"  Battle zone: {len(player_after.battle_zone)}")
            print(f"  Graveyard: {len(player_after.graveyard)}")
            print(f"  Stack: {len(player_after.stack)}")
            
            # 成功判定
            hand_decreased = len(player_after.hand) < len(player_before.hand)
            battle_increased = len(player_after.battle_zone) > len(player_before.battle_zone)
            graveyard_increased = len(player_after.graveyard) > len(player_before.graveyard)
            
            if hand_decreased and (battle_increased or graveyard_increased):
                print("\n✓ SUCCESS: Card was played correctly!")
                if battle_increased:
                    print("  → Creature/permanent entered battle zone")
                if graveyard_increased:
                    print("  → Spell was cast and sent to graveyard")
            elif len(player_after.stack) > 0:
                print("\n⚠ WARNING: Card is on stack (may need additional actions)")
            else:
                print("\n✗ PARTIAL: Card state changed but not as expected")
                
        except Exception as e:
            print(f"\n✗ ERROR playing card: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No PLAY_CARD actions available (might need mana/valid cards)")
else:
    print(f"\n✗ ERROR: Could not reach MAIN phase. Current phase: {int(state.current_phase)}")

print("\n=== Test Complete ===")
