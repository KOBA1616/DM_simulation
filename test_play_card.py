"""メインフェイズのPLAY_CARD動作テスト"""
import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module as dm
    from dm_toolkit import commands_v2 as commands
    print("✓ dm_ai_module imported")
    
    # カードDB読み込み
    print("\n1. Loading card database...")
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    print("✓ Card database loaded")
    
    # ゲーム作成
    print("\n2. Creating game...")
    game = dm.GameInstance(12345, card_db)
    print("✓ Game created")
    
    # デッキ設定
    print("\n3. Setting up decks...")
    deck = [1] * 40
    game.state.set_deck(0, deck)
    game.state.set_deck(1, deck)
    print("✓ Decks configured")
    
    # ゲーム開始
    print("\n4. Starting game...")
    game.start_game()
    print(f"✓ Game started: Phase {int(game.state.current_phase)}")
    
    # テスト用にカードを手札に追加
    print("\n5. Adding cards to hand...")
    player = game.state.players[0]
    for i in range(3):
        card_inst = dm.CardInstance()
        card_inst.card_id = 1
        card_inst.instance_id = 100 + i
        card_inst.owner = 0
        player.hand.append(card_inst)
    print(f"✓ Hand size: {len(player.hand)}")
    
    # マナゾーンにカード追加
    print("\n6. Adding cards to mana zone...")
    for i in range(5):
        card_inst = dm.CardInstance()
        card_inst.card_id = 1
        card_inst.instance_id = 200 + i
        card_inst.owner = 0
        player.mana_zone.append(card_inst)
    print(f"✓ Mana zone: {len(player.mana_zone)}")
    
    # MAINフェーズに移行
    print("\n7. Moving to MAIN phase...")
    game.state.current_phase = dm.Phase.MAIN
    print(f"✓ Phase: {int(game.state.current_phase)} (MAIN=3)")
    
    # アクション生成
    print("\n8. Generating actions...")
        actions = commands.generate_legal_commands(game.state, card_db, strict=False)
    play_actions = [a for a in actions if int(a.type) == 1]  # ActionType.PLAY_CARD = 1
    print(f"✓ Total actions: {len(actions)}, PLAY_CARD: {len(play_actions)}")
    
    if play_actions:
        print("\n9. Testing PLAY_CARD...")
        
        # プレイ前の状態
        before_hand = len(player.hand)
        before_battle = len(player.battle_zone)
        before_grave = len(player.graveyard)
        before_stack = len(player.stack)
        print(f"  Before: Hand={before_hand}, Battle={before_battle}, Grave={before_grave}, Stack={before_stack}")
        
        # カードプレイ
        action = play_actions[0]
        game.resolve_action(action)
        
        # プレイ後の状態
        player = game.state.players[0]
        after_hand = len(player.hand)
        after_battle = len(player.battle_zone)
        after_grave = len(player.graveyard)
        after_stack = len(player.stack)
        print(f"  After:  Hand={after_hand}, Battle={after_battle}, Grave={after_grave}, Stack={after_stack}")
        
        # 結果検証
        print("\n10. Validation:")
        success = True
        
        if after_stack == 0:
            print("  ✓ Card not stuck on stack (old bug fixed!)")
        else:
            print(f"  ✗ Card stuck on stack: {after_stack}")
            success = False
            
        if after_hand < before_hand:
            print("  ✓ Card left hand")
        else:
            print("  ✗ Card still in hand")
            success = False
            
        if after_battle > before_battle or after_grave > before_grave:
            print("  ✓ Card reached final zone")
            if after_battle > before_battle:
                print("    → Battle zone (creature)")
            if after_grave > before_grave:
                print("    → Graveyard (spell)")
        else:
            print("  ✗ Card didn't reach final zone")
            success = False
        
        if success:
            print("\n✓✓ PLAY_CARD WORKS CORRECTLY! ✓✓")
        else:
            print("\n✗✗ PLAY_CARD HAS ISSUES ✗✗")
    else:
        print("\n✗ No PLAY_CARD actions available")

except Exception as e:
    print(f"\n✗✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
