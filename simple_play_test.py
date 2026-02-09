"""メインフェイズのカードプレイ簡易テスト"""
import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module as dm
    print("✓ dm_ai_module imported successfully")
    
    # カードデータベース読み込み
    print("\n1. Loading card database...")
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    print("✓ Card database loaded")
    
    # ゲーム作成
    print("\n2. Creating game instance...")
    game = dm.GameInstance(12345, card_db)
    print("✓ Game instance created")
    
    # 状態取得
    print("\n3. Getting game state...")
    state = game.state
    print(f"✓ State retrieved: Turn {state.turn_number}, Phase {int(state.current_phase)}")
    
    # デッキ設定（カードID 1を40枚）
    print("\n4. Setting up decks...")
    deck = [1] * 40
    state.set_deck(0, deck)
    state.set_deck(1, deck)
    print("✓ Decks configured")
    
    # ゲーム開始
    print("\n5. Starting game...")
    game.start_game()
    state = game.state
    print(f"✓ Game started: Turn {state.turn_number}, Phase {int(state.current_phase)}")
    
    # 手札にカード追加 (テスト用にカードID 1を5枚追加)
    print("\n6. Adding cards to hand for testing...")
    player = state.players[0]
    for _ in range(5):
        card_inst = state.create_new_card(1, 0)  # card_id=1, player=0
        player.hand.append(card_inst)
    print(f"✓ Hand size: {len(player.hand)}")
    
    # MANAフェーズにマナチャージ (テスト用に5マナ用意)
    print("\n7. Setting up mana...")
    for i in range(5):
        if i < len(player.hand):
            card = player.hand[i]
            player.mana_zone.append(card)
    # 手札から削除
    player.hand = [c for c in player.hand if c not in player.mana_zone]
    print(f"✓ Mana zone: {len(player.mana_zone)}, Hand: {len(player.hand)}")
    
    # MAINフェイズに移行
    print("\n8. Moving to MAIN phase...")
    state.current_phase = dm.Phase.MAIN
    print(f"✓ Phase: {int(state.current_phase)} (MAIN=3)")
    
    # IntentGeneratorでアクション生成
    print("\n9. Generating actions...")
    from dm_toolkit import commands_v2 as commands
    actions = commands.generate_legal_commands(state, card_db, strict=False)
    # IntentGeneratorでアクション生成
    print("\n9. Generating actions...")
    actions = commands.generate_legal_commands(state, card_db, strict=False)
    print(f"✓ {len(actions)} actions available")
    
    # PLAY_CARDアクション確認
    play_actions = [a for a in actions if int(a.type) == 1]  # ActionType::PLAY_CARD = 1
    print(f"  PLAY_CARD actions: {len(play_actions)}")
    
    if play_actions:
        print("\n10. Testing PLAY_CARD...")
        
        # プレイ前の状態
        player = state.players[state.active_player_id]
        before_hand = len(player.hand)
        before_battle = len(player.battle_zone)
        before_grave = len(player.graveyard)
        before_stack = len(player.stack)
        print(f"  Before: Hand={before_hand}, Battle={before_battle}, Grave={before_grave}, Stack={before_stack}")
        
        # カードプレイ
        action = play_actions[0]
        game.resolve_action(action)
        
        # プレイ後の状態
        state = game.state
        player = state.players[state.active_player_id]
        after_hand = len(player.hand)
        after_battle = len(player.battle_zone)
        after_grave = len(player.graveyard)
        after_stack = len(player.stack)
        print(f"  After:  Hand={after_hand}, Battle={after_battle}, Grave={after_grave}, Stack={after_stack}")
        
        # 結果判定
        if after_hand < before_hand and (after_battle > before_battle or after_grave > before_grave):
            print("\n✓✓ SUCCESS: Card was played correctly! ✓✓")
            if after_battle > before_battle:
                print("  → Creature entered battle zone")
            if after_grave > before_grave:
                print("  → Spell was cast to graveyard")
        elif after_stack > before_stack:
            print("\n⚠ PARTIAL: Card is on stack (old behavior - needs manual PAY_COST/RESOLVE)")
        else:
            print("\n⚠ WARNING: Unexpected state change")
            print(f"  Hand: {before_hand} → {after_hand}")
            print(f"  Battle: {before_battle} → {after_battle}")
            print(f"  Grave: {before_grave} → {after_grave}")
            print(f"  Stack: {before_stack} → {after_stack}")
    else:
        print("\n⚠ No PLAY_CARD actions in MAIN phase")
    
except Exception as e:
    print(f"\n✗✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
