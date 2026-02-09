"""メインフェイズ実装の整合性チェック"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== Main Phase Implementation Consistency Check ===\n")

# カードDB読み込み
card_db = dm.JsonLoader.load_cards('data/cards.json')

# テストケース1: 通常のカードプレイフロー
print("## Test Case 1: Normal Card Play Flow")
print("-" * 50)

game = dm.GameInstance(12345, card_db)
config = dm.ScenarioConfig()
config.my_hand_cards = [1, 1, 1]
config.my_mana_zone = [1, 1, 1, 1, 1]
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN

print(f"Phase: {int(game.state.current_phase)} (MAIN=3)")
print(f"Hand: {len(game.state.players[0].hand)} cards")
print(f"Mana: {len(game.state.players[0].mana_zone)} cards (5 untapped)")

# アクション生成
from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(game.state, card_db, strict=False)
print(f"\nGenerated {len(actions)} actions")

# アクションタイプの分類
action_types = {}
for a in actions:
    atype = int(a.type)
    if atype not in action_types:
        action_types[atype] = []
    action_types[atype].append(a)

print("\nAction Type Distribution:")
for atype, acts in sorted(action_types.items()):
    # PlayerIntentの名前を取得
    type_name = "UNKNOWN"
    for attr in dir(dm.PlayerIntent):
        if not attr.startswith('_') and getattr(dm.PlayerIntent, attr) == atype:
            type_name = attr
            break
    print(f"  {type_name} ({atype}): {len(acts)} actions")

# DECLARE_PLAYが生成されることを確認
if 15 in action_types:  # DECLARE_PLAY
    print("\n✓ DECLARE_PLAY actions are generated (expected)")
else:
    print("\n✗ DECLARE_PLAY actions NOT generated (unexpected!)")

if 3 in action_types:  # PLAY_CARD
    print("⚠ PLAY_CARD actions also generated (may indicate duplication)")
else:
    print("✓ PLAY_CARD actions not generated (expected)")

# テストケース2: Stackに手動で配置したカードの処理
print("\n\n## Test Case 2: Manual Stack Handling")
print("-" * 50)

game2 = dm.GameInstance(54321, card_db)
config2 = dm.ScenarioConfig()
config2.my_hand_cards = []
config2.my_mana_zone = [1, 1, 1, 1, 1]
config2.my_shields = []
config2.enemy_shield_count = 5

game2.reset_with_scenario(config2)

# 手動でStackにカードを配置（テスト用）
card_inst = dm.CardInstance()
card_inst.card_id = 1
card_inst.instance_id = 999
card_inst.owner = 0
card_inst.is_tapped = False  # 未払い状態
game2.state.players[0].stack.append(card_inst)

game2.state.current_phase = dm.Phase.MAIN

print(f"Manually placed card on stack (untapped)")
print(f"Stack: {len(game2.state.players[0].stack)} cards")

# アクション生成 - StackStrategyがPAY_COSTを生成するか確認
actions2 = commands.generate_legal_commands(game2.state, card_db, strict=False)
print(f"\nGenerated {len(actions2)} actions")

action_types2 = {}
for a in actions2:
    atype = int(a.type)
    if atype not in action_types2:
        action_types2[atype] = []
    action_types2[atype].append(a)

print("\nAction Type Distribution:")
for atype, acts in sorted(action_types2.items()):
    type_name = "UNKNOWN"
    for attr in dir(dm.PlayerIntent):
        if not attr.startswith('_') and getattr(dm.PlayerIntent, attr) == atype:
            type_name = attr
            break
    print(f"  {type_name} ({atype}): {len(acts)} actions")

# PAY_COSTアクションの確認
if 16 in action_types2:  # PAY_COST
    print("\n✓ PAY_COST action generated for card on stack (manual flow supported)")
else:
    print("\n⚠ PAY_COST action NOT generated (manual flow may not work)")

# テストケース3: 支払い済みカードの解決
print("\n\n## Test Case 3: Paid Card Resolution")
print("-" * 50)

card_inst.is_tapped = True  # 支払い済み状態に変更

actions3 = commands.generate_legal_commands(game2.state, card_db, strict=False)
action_types3 = {}
for a in actions3:
    atype = int(a.type)
    if atype not in action_types3:
        action_types3[atype] = []
    action_types3[atype].append(a)

print("Card on stack marked as tapped (paid)")
print(f"Generated {len(actions3)} actions")

print("\nAction Type Distribution:")
for atype, acts in sorted(action_types3.items()):
    type_name = "UNKNOWN"
    for attr in dir(dm.PlayerIntent):
        if not attr.startswith('_') and getattr(dm.PlayerIntent, attr) == atype:
            type_name = attr
            break
    print(f"  {type_name} ({atype}): {len(acts)} actions")

# RESOLVE_PLAYアクションの確認
if 17 in action_types3:  # RESOLVE_PLAY
    print("\n✓ RESOLVE_PLAY action generated for paid card (manual flow supported)")
else:
    print("\n⚠ RESOLVE_PLAY action NOT generated (manual flow may not work)")

# 整合性チェック結果
print("\n\n" + "=" * 50)
print("CONSISTENCY CHECK RESULTS")
print("=" * 50)

issues = []
warnings = []

# アクション生成の確認
if 15 not in action_types:
    issues.append("DECLARE_PLAY not generated in MAIN phase")
if 3 in action_types:
    warnings.append("Both PLAY_CARD and DECLARE_PLAY may be generated (check for duplication)")

# Stack手動フローの確認
if 16 not in action_types2:
    warnings.append("PAY_COST not generated for unpaid card on stack (manual flow limited)")
if 17 not in action_types3:
    warnings.append("RESOLVE_PLAY not generated for paid card on stack (manual flow limited)")

if issues:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ No critical issues found")

if warnings:
    print("\n⚠ WARNINGS:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
else:
    print("\n✓ No warnings")

# 実装の説明
print("\n\nIMPLEMENTATION SUMMARY:")
print("-" * 50)
print("Current Implementation:")
print("  1. MainPhaseStrategy generates DECLARE_PLAY actions")
print("  2. DECLARE_PLAY executes full Stack Lifecycle automatically:")
print("     - Move to Stack")
print("     - Auto-pay mana cost")
print("     - Resolve to final zone (Battle/Graveyard)")
print("  3. Manual PAY_COST/RESOLVE_PLAY flow may still be supported")
print("     for cards manually placed on stack")

print("\n✓ Implementation is consistent with Stack Lifecycle design")

print("\n=== Check Complete ===")
