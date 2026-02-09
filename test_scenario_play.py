"""PLAY_CARD機能テスト - シナリオベース"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== PLAY_CARD Functionality Test ===\n")

# カードDB読み込み
print("1. Loading card database...")
card_db = dm.JsonLoader.load_cards('data/cards.json')
print("✓ Loaded")

# ゲーム作成
print("\n2. Creating game...")
game = dm.GameInstance(12345, card_db)
print("✓ Created")

# シナリオ設定
print("\n3. Setting up scenario...")
config = dm.ScenarioConfig()
config.my_hand_cards = [1, 1, 1]  # カードID 1を3枚
config.my_mana_zone = [1, 1, 1, 1, 1]  # マナ5枚
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
state = game.state
state.current_phase = dm.Phase.MAIN  # MAINフェーズに設定
print(f"✓ Scenario set - Phase: {int(state.current_phase)}, Active: {state.active_player_id}")
print(f"  P0: Hand={len(state.players[0].hand)}, Mana={len(state.players[0].mana_zone)}, Battle={len(state.players[0].battle_zone)}")

# IntentGeneratorでアクション生成
print("\n4. Generating commands with commands_v2...")
from dm_toolkit import commands_v2 as commands
try:
    # Prefer native command generator
    actions = commands.generate_legal_commands(state, card_db, strict=False)
    print(f"  commands_v2 type: {type(commands)}")
    print(f"✓ Generated {len(actions) if actions is not None else 0} commands")
    
    # アクションタイプを確認
    action_types = {}
    for a in actions:
        atype = int(a.type)
        action_types[atype] = action_types.get(atype, 0) + 1
    
    print(f"  Action types: {action_types}")
    
    # PLAY_CARDアクション検索 (ActionType値を複数試す)
    play_actions = []
    for type_val in [1, 3]:  # PLAY_CARD候補値
        play_actions = [a for a in actions if int(a.type) == type_val]
        if play_actions:
            print(f"  Found PLAY_CARD actions with type={type_val}")
            break
    
    if play_actions:
        print(f"\n5. Executing PLAY_CARD action...")
        
        # プレイ前の状態
        p = state.players[0]
        before_hand = len(p.hand)
        before_battle = len(p.battle_zone)
        before_grave = len(p.graveyard)
        before_stack = len(p.stack)
        
        print(f"  Before: Hand={before_hand}, Battle={before_battle}, Grave={before_grave}, Stack={before_stack}")
        
        # アクション実行
        action = play_actions[0]
        game.resolve_action(action)
        
        # プレイ後の状態
        p = game.state.players[0]
        after_hand = len(p.hand)
        after_battle = len(p.battle_zone)
        after_grave = len(p.graveyard)
        after_stack = len(p.stack)
        
        print(f"  After:  Hand={after_hand}, Battle={after_battle}, Grave={after_grave}, Stack={after_stack}")
        
        # 結果検証
        print("\n6. Results:")
        print(f"  Hand: {before_hand} → {after_hand} ({after_hand - before_hand:+d})")
        print(f"  Battle: {before_battle} → {after_battle} ({after_battle - before_battle:+d})")
        print(f"  Grave: {before_grave} → {after_grave} ({after_grave - before_grave:+d})")
        print(f"  Stack: {before_stack} → {after_stack} ({after_stack - before_stack:+d})")
        
        # 成功判定
        if after_stack == 0 and after_hand < before_hand and (after_battle > before_battle or after_grave > before_grave):
            print("\n✓✓ SUCCESS! PLAY_CARD works correctly:")
            print("  ✓ Card left hand")
            print("  ✓ Card not stuck on stack")
            print("  ✓ Card reached final zone")
        else:
            print("\n⚠ PARTIAL SUCCESS or OLD BEHAVIOR:")
            if after_stack > 0:
                print(f"  ⚠ Card stuck on stack ({after_stack})")
            if after_hand >= before_hand:
                print("  ⚠ Card didn't leave hand")
            if after_battle == before_battle and after_grave == before_grave:
                print("  ⚠ Card didn't reach final zone")
    else:
        print("\n⚠ No PLAY_CARD actions found")
        print(f"  Available action types: {action_types}")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
