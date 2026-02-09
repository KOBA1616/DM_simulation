"""
ツインパクトカード処理修正のテストスクリプト
"""
import sys
sys.path.insert(0, '.')

import dm_ai_module as dm

def test_twinpact_action_generation():
    """ツインパクトカードから2つのDECLARE_PLAYアクションが生成されるか確認"""
    print("=== Test: Twinpact Action Generation ===")
    
    gs = dm.GameState(42)
    gs.setup_test_duel()
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    
    # ツインパクトカード (ID: 4) を手札に追加
    twinpact_card = dm.CardInstance()
    twinpact_card.card_id = 4
    twinpact_card.instance_id = 100
    twinpact_card.owner = 0
    
    # マナを4枚追加（クリーチャー側をプレイできる量）
    for i in range(4):
        mana = dm.CardInstance()
        mana.card_id = 1
        mana.instance_id = 200 + i
        mana.owner = 0
        gs.players[0].mana_zone.append(mana)
    
    # 手札に追加
    gs.players[0].hand.append(twinpact_card)
    
    # ゲーム開始してメインフェイズまで進める
    dm.PhaseManager.start_game(gs, card_db)
    dm.PhaseManager.fast_forward(gs, card_db)
    
    print(f"Current Phase: {gs.current_phase}")
    print(f"Active Player: {gs.active_player_id}")
    print(f"P0 Hand: {len(gs.players[0].hand)} cards")
    print(f"P0 Mana: {len(gs.players[0].mana_zone)} cards")
    
    # コマンド優先で生成
    from dm_toolkit import commands_v2 as commands
    actions = commands.generate_legal_commands(gs, card_db, strict=False)
    
    # DECLARE_PLAYアクションを抽出
    # Map command-like objects to a similar interface if needed, fallback to filtering by type string
    declare_play_actions = [a for a in (actions or []) if getattr(a, 'type', None) == getattr(dm.PlayerIntent, 'DECLARE_PLAY', None) or str(getattr(a, 'type', '')).upper().find('DECLARE_PLAY') != -1]
    
    print(f"\nTotal DECLARE_PLAY actions: {len(declare_play_actions)}")
    
    # ツインパクトカード用のアクションを確認
    twinpact_actions = [a for a in declare_play_actions if a.source_instance_id == 100]
    
    print(f"Twinpact card actions: {len(twinpact_actions)}")
    
    for i, action in enumerate(twinpact_actions):
        print(f"  Action {i+1}:")
        print(f"    card_id: {action.card_id}")
        print(f"    source_instance_id: {action.source_instance_id}")
        print(f"    is_spell_side: {action.is_spell_side}")
    
    # 検証
    assert len(twinpact_actions) == 2, f"Expected 2 actions, got {len(twinpact_actions)}"
    
    has_creature_side = any(not a.is_spell_side for a in twinpact_actions)
    has_spell_side = any(a.is_spell_side for a in twinpact_actions)
    
    assert has_creature_side, "Creature side action not found"
    assert has_spell_side, "Spell side action not found"
    
    print("\n✅ Test PASSED: Both creature and spell side actions generated correctly")
    return True


def test_twinpact_spell_execution():
    """呪文側プレイが正しいコストで実行されるか確認"""
    print("\n=== Test: Twinpact Spell Side Execution ===")
    
    gs = dm.GameState(42)
    gs.setup_test_duel()
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    gi = dm.GameInstance(gs)
    
    # ツインパクトカード (ID: 4, クリーチャーコスト4, 呪文コスト3) を手札に追加
    twinpact_card = dm.CardInstance()
    twinpact_card.card_id = 4
    twinpact_card.instance_id = 100
    twinpact_card.owner = 0
    gs.players[0].hand.append(twinpact_card)
    
    # マナを3枚だけ追加（呪文側はプレイ可能、クリーチャー側は不可）
    for i in range(3):
        mana = dm.CardInstance()
        mana.card_id = 1
        mana.instance_id = 200 + i
        mana.owner = 0
        mana.tapped = False
        gs.players[0].mana_zone.append(mana)
    
    # ゲーム開始してメインフェイズまで進める
    dm.PhaseManager.start_game(gs, card_db)
    dm.PhaseManager.fast_forward(gs, card_db)
    
    print(f"Before play:")
    print(f"  Hand: {len(gs.players[0].hand)} cards")
    print(f"  Mana: {len(gs.players[0].mana_zone)} cards")
    print(f"  Graveyard: {len(gs.players[0].graveyard)} cards")
    
    # 呪文側プレイアクションを作成
    spell_action = dm.Action()
    spell_action.type = dm.PlayerIntent.DECLARE_PLAY
    spell_action.source_instance_id = 100
    spell_action.card_id = 4
    spell_action.is_spell_side = True
    
    # 実行
    try:
        gi.resolve_action(spell_action)
        gs = gi.state  # 状態を再取得
        
        print(f"\nAfter spell side play:")
        print(f"  Hand: {len(gs.players[0].hand)} cards")
        print(f"  Mana: {len(gs.players[0].mana_zone)} cards")
        print(f"  Graveyard: {len(gs.players[0].graveyard)} cards")
        print(f"  Battle Zone: {len(gs.players[0].battle_zone)} cards")
        
        # 呪文はバトルゾーンに出ず、墓地に行くはず
        tapped_mana = sum(1 for m in gs.players[0].mana_zone if m.tapped)
        print(f"  Tapped Mana: {tapped_mana}")
        
        # 検証
        assert len(gs.players[0].hand) == 0, "Hand should be empty"
        assert len(gs.players[0].battle_zone) == 0, "Spell should not go to battle zone"
        assert tapped_mana == 3, f"Expected 3 tapped mana, got {tapped_mana}"
        
        print("\n✅ Test PASSED: Spell side executed with correct cost (3)")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_twinpact_creature_execution():
    """クリーチャー側プレイが正しいコストで実行されるか確認"""
    print("\n=== Test: Twinpact Creature Side Execution ===")
    
    gs = dm.GameState(42)
    gs.setup_test_duel()
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    gi = dm.GameInstance(gs)
    
    # ツインパクトカード (ID: 4) を手札に追加
    twinpact_card = dm.CardInstance()
    twinpact_card.card_id = 4
    twinpact_card.instance_id = 100
    twinpact_card.owner = 0
    gs.players[0].hand.append(twinpact_card)
    
    # マナを4枚追加（クリーチャー側プレイ可能）
    for i in range(4):
        mana = dm.CardInstance()
        mana.card_id = 1
        mana.instance_id = 200 + i
        mana.owner = 0
        mana.tapped = False
        gs.players[0].mana_zone.append(mana)
    
    # ゲーム開始してメインフェイズまで進める
    dm.PhaseManager.start_game(gs, card_db)
    dm.PhaseManager.fast_forward(gs, card_db)
    
    print(f"Before play:")
    print(f"  Hand: {len(gs.players[0].hand)} cards")
    print(f"  Mana: {len(gs.players[0].mana_zone)} cards")
    print(f"  Battle Zone: {len(gs.players[0].battle_zone)} cards")
    
    # クリーチャー側プレイアクション
    creature_action = dm.Action()
    creature_action.type = dm.PlayerIntent.DECLARE_PLAY
    creature_action.source_instance_id = 100
    creature_action.card_id = 4
    creature_action.is_spell_side = False
    
    # 実行
    try:
        gi.resolve_action(creature_action)
        gs = gi.state
        
        print(f"\nAfter creature side play:")
        print(f"  Hand: {len(gs.players[0].hand)} cards")
        print(f"  Mana: {len(gs.players[0].mana_zone)} cards")
        print(f"  Battle Zone: {len(gs.players[0].battle_zone)} cards")
        print(f"  Graveyard: {len(gs.players[0].graveyard)} cards")
        
        tapped_mana = sum(1 for m in gs.players[0].mana_zone if m.tapped)
        print(f"  Tapped Mana: {tapped_mana}")
        
        # 検証
        assert len(gs.players[0].hand) == 0, "Hand should be empty"
        assert len(gs.players[0].battle_zone) == 1, "Creature should be in battle zone"
        assert tapped_mana == 4, f"Expected 4 tapped mana, got {tapped_mana}"
        
        print("\n✅ Test PASSED: Creature side executed with correct cost (4)")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Testing Twinpact Card Fix\n")
    
    results = []
    
    try:
        results.append(("Action Generation", test_twinpact_action_generation()))
    except Exception as e:
        print(f"Action Generation test failed: {e}")
        results.append(("Action Generation", False))
    
    try:
        results.append(("Spell Side Execution", test_twinpact_spell_execution()))
    except Exception as e:
        print(f"Spell Side test failed: {e}")
        results.append(("Spell Side Execution", False))
    
    try:
        results.append(("Creature Side Execution", test_twinpact_creature_execution()))
    except Exception as e:
        print(f"Creature Side test failed: {e}")
        results.append(("Creature Side Execution", False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
