"""メインフェイズ後のターン終了テスト"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== Turn Ending Test ===\n")

# ゲーム作成
card_db = dm.JsonLoader.load_cards('data/cards.json')
game = dm.GameInstance(12345, card_db)

# シナリオ設定
config = dm.ScenarioConfig()
config.my_hand_cards = [1, 1, 1]  # cost=2クリーチャー
config.my_mana_zone = [1, 1]  # マナ2枚
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN
game.state.turn_number = 1

print("初期状態:")
print(f"  Turn: {game.state.turn_number}")
print(f"  Phase: {game.state.current_phase}")
print(f"  Active Player: {game.state.active_player_id}")

# P0のカードをプレイ
p0 = game.state.players[0]
print(f"\nP0状態:")
print(f"  Hand: {len(p0.hand)}")
print(f"  Mana: {len(p0.mana_zone)} (untapped: {sum(1 for c in p0.mana_zone if not c.is_tapped)})")
print(f"  Battle: {len(p0.battle_zone)}")

# アクション生成
actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
print(f"\n利用可能なアクション: {len(actions)}")

# DECLARE_PLAYアクションを探す
declare_play_actions = [a for a in actions if int(a.type) == 15]
pass_actions = [a for a in actions if int(a.type) == 0]

print(f"  DECLARE_PLAY: {len(declare_play_actions)}")
print(f"  PASS: {len(pass_actions)}")

# カードをプレイ
if declare_play_actions:
    print("\n=== DECLARE_PLAYアクション実行 ===")
    action = declare_play_actions[0]
    
    print(f"プレイ前フェーズ: {game.state.current_phase}")
    game.resolve_action(action)
    print(f"プレイ後フェーズ: {game.state.current_phase}")
    
    print(f"\nプレイ後P0状態:")
    print(f"  Hand: {len(p0.hand)}")
    print(f"  Mana: {len(p0.mana_zone)} (untapped: {sum(1 for c in p0.mana_zone if not c.is_tapped)})")
    print(f"  Battle: {len(p0.battle_zone)}")
    
    # PASSアクションを確認
    actions_after_play = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
    pass_actions_after = [a for a in actions_after_play if int(a.type) == 0]
    declare_actions_after = [a for a in actions_after_play if int(a.type) == 15]
    
    print(f"\nカードプレイ後の利用可能なアクション: {len(actions_after_play)}")
    print(f"  DECLARE_PLAY: {len(declare_actions_after)}")
    print(f"  PASS: {len(pass_actions_after)}")
    
    # 全アクションの詳細を表示
    for i, action in enumerate(actions_after_play):
        print(f"  Action[{i}]: type={int(action.type)}")
    
    # RESOLVE_EFFECT/PASSアクションを繰り返し実行して効果を完全に解決
    max_iterations = 10
    iteration = 0
    while iteration < max_iterations:
        actions_current = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
        resolve_actions = [a for a in actions_current if int(a.type) == 10]  # RESOLVE_EFFECT
        pass_actions_current = [a for a in actions_current if int(a.type) == 0]  # PASS
        
        print(f"\nIteration {iteration+1}: Actions={len(actions_current)}, RESOLVE_EFFECT={len(resolve_actions)}, PASS={len(pass_actions_current)}")
        
        if pass_actions_current:
            # PASSアクションが見つかった - メインフェイズのPASSかチェック
            # slot_indexが設定されているPASSはpending_effect用
            # slot_indexが未設定のPASSはメインフェイズ用
            main_phase_pass = [a for a in pass_actions_current if a.slot_index < 0]
            if main_phase_pass:
                print(f"メインフェイズのPASSアクション発見！")
                pass_actions_after = main_phase_pass
                break
            else:
                # pending_effect用のPASSを実行
                print(f"  pending_effect用PASSを実行")
                game.resolve_action(pass_actions_current[0])
        elif resolve_actions:
            # RESOLVE_EFFECTを実行
            print(f"  RESOLVE_EFFECTを実行")
            game.resolve_action(resolve_actions[0])
        else:
            # 他のアクションがある場合
            print(f"  他のアクション: {[int(a.type) for a in actions_current]}")
            if len(actions_current) == 1 and int(actions_current[0].type) == 0:
                # PASSのみ
                pass_actions_after = actions_current
                break
            break
        
        iteration += 1
    else:
        print(f"\nMax iterations達成（効果が完全に解決されなかった可能性）")
        # 強制的にpending_effectsをクリアしてターン終了をテスト
        print(f"pending_effects数: {len(game.state.pending_effects)}")
        print(f"pending_effectsを強制クリアしてターン終了機能をテスト")
        game.state.pending_effects.clear()
        
        actions_after_clear = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
        pass_actions_after = [a for a in actions_after_clear if int(a.type) == 0]
        print(f"クリア後のアクション: {len(actions_after_clear)}, PASS: {len(pass_actions_after)}")
        if actions_after_clear:
            print(f"  Action types: {[int(a.type) for a in actions_after_clear]}")
            print(f"  Current phase: {game.state.current_phase}")
    
    print(f"\n最終状態: PASS actions = {len(pass_actions_after) if 'pass_actions_after' in locals() else 0}")
    # PASSアクションでターン終了
    if pass_actions_after:
        print("\n=== PASSアクション実行（ターン終了） ===")
        pass_action = pass_actions_after[0]
        
        print(f"PASS前:")
        print(f"  Turn: {game.state.turn_number}")
        print(f"  Phase: {game.state.current_phase}")
        print(f"  Active Player: {game.state.active_player_id}")
        
        game.resolve_action(pass_action)
        
        print(f"\nPASS後:")
        print(f"  Turn: {game.state.turn_number}")
        print(f"  Phase: {game.state.current_phase}")
        print(f"  Active Player: {game.state.active_player_id}")
        
        # 検証
        print(f"\n=== 検証結果 ===")
        phase_changed = game.state.current_phase != dm.Phase.MAIN
        turn_advanced = game.state.turn_number > 1 or game.state.active_player_id != 0
        
        print(f"[{'OK' if phase_changed else 'NG'}] フェーズが変わった (MAIN以外)")
        print(f"[{'OK' if turn_advanced else 'NG'}] ターンが進行した (ターン番号またはアクティブプレイヤー変更)")
        
        if phase_changed and turn_advanced:
            print("\n[OK] ターン終了は正常に動作しています")
        else:
            print("\nターン終了に問題がある可能性があります")
    else:
        print("\nPASSアクションが生成されていません")
else:
    print("\nDECLARE_PLAYアクションが生成されていません")

print("\n=== Test Complete ===")
