"""DECLARE_PLAY修正の検証テスト"""
import dm_ai_module as dm

# ゲーム初期化（2プレイヤー）
state = dm.GameState(2)
state.initialize_two_player_game()

# デッキ設定
for pid in [0, 1]:
    player = state.get_player(pid)
    player.deck.clear()
    for _ in range(40):
        # cost=2のクリーチャー (id=1: 月光電人オボロカゲロウ)
        player.deck.append(dm.CardInstance(1, state.generate_instance_id(), pid))

# ゲーム開始
state.deploy_decks()
state.start_game()

print(f"ゲーム開始後 - Phase: {state.current_phase}, Active Player: {state.active_player_id}")

# P0の初期手札とマナを確認
p0 = state.get_player(0)
print(f"\nP0初期状態:")
print(f"  Hand: {len(p0.hand)}")
print(f"  Mana: {len(p0.mana)}")
print(f"  Battle: {len(p0.battle)}")

# 2ターン分マナチャージ
for turn in range(2):
    # P0のターン
    state.active_player_id = 0
    state.current_phase = dm.Phase.MAIN
    
    if len(p0.hand) > 0:
        card = p0.hand[0]
        action = dm.Action()
        action.type = dm.PlayerIntent.MANA_CHARGE
        action.source_instance_id = card.instance_id
        
        cmd = dm.ManaChargeCommand(card.instance_id)
        state.execute_command(cmd)
        
        print(f"\nTurn {turn+1}: P0 MANA_CHARGE (hand: {len(p0.hand)}, mana: {len(p0.mana)})")
    
    # フェーズを進める（簡易版）
    state.active_player_id = 1

print(f"\nマナチャージ後:")
print(f"  Hand: {len(p0.hand)}")
print(f"  Mana: {len(p0.mana)}")
print(f"  Mana untapped: {sum(1 for c in p0.mana if not c.is_tapped)}")

# クリーチャープレイ（DECLARE_PLAY）を試行
if len(p0.mana) >= 2 and len(p0.hand) > 0:
    print(f"\n=== DECLARE_PLAYアクション実行 ===")
    
    state.active_player_id = 0
    state.current_phase = dm.Phase.MAIN
    
    # 手札の最初のカードをプレイ
    card = p0.hand[0]
    
    print(f"プレイ前:")
    print(f"  Hand: {len(p0.hand)}")
    print(f"  Battle: {len(p0.battle)}")
    print(f"  Stack: {len(state.get_stack())}")
    print(f"  Mana untapped: {sum(1 for c in p0.mana if not c.is_tapped)}")
    print(f"  Phase: {state.current_phase}")
    
    # DECLARE_PLAYアクションを作成
    action = dm.Action()
    action.type = dm.PlayerIntent.DECLARE_PLAY
    action.source_instance_id = card.instance_id
    action.card_id = card.card_id
    
    # GameInstanceを使ってアクションを実行
    game = dm.GameInstance()
    game.state = state
    game.card_db = dm.CardRegistry.get_all_definitions()
    
    try:
        game.resolve_action(action)
        
        print(f"\nプレイ後:")
        print(f"  Hand: {len(p0.hand)}")
        print(f"  Battle: {len(p0.battle)}")
        print(f"  Stack: {len(state.get_stack())}")
        print(f"  Mana untapped: {sum(1 for c in p0.mana if not c.is_tapped)}")
        print(f"  Phase: {state.current_phase}")
        
        # 検証
        print(f"\n=== 検証結果 ===")
        hand_decreased = len(p0.hand) < 3  # 初期5 - 2回マナチャージ = 3
        battle_increased = len(p0.battle) == 1
        stack_empty = len(state.get_stack()) == 0
        mana_tapped = sum(1 for c in p0.mana if not c.is_tapped) == 0  # 2マナ全てタップ
        
        print(f"[{'OK' if hand_decreased else 'NG'}] 手札が減った")
        print(f"[{'OK' if battle_increased else 'NG'}] クリーチャーがBattle Zoneに出た")
        print(f"[{'OK' if stack_empty else 'NG'}] Stackが空（カードが残っていない）")
        print(f"[{'OK' if mana_tapped else 'NG'}] マナがタップされた（コスト支払い済み）")
        
        if hand_decreased and battle_increased and stack_empty and mana_tapped:
            print("\n✓ DECLARE_PLAY修正は正常に動作しています")
        else:
            print("\n✗ 問題が残っています")
            
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nテスト条件不足（マナまたは手札が足りない）")
