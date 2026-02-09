"""カードプレイ後の状態検証テスト - マナ支払いと召喚確認"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== Card Play Verification Test ===\n")

# カードDB読み込み
card_db = dm.JsonLoader.load_cards('data/cards.json')
print("1. Card database loaded")

# ゲーム作成
game = dm.GameInstance(12345, card_db)
print("2. Game instance created")

# シナリオ設定
config = dm.ScenarioConfig()
config.my_hand_cards = [1, 1, 1]  # カードID 1 (コスト2のクリーチャー)
config.my_mana_zone = [1, 1, 1, 1, 1]  # マナ5枚
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN
print("3. Scenario setup - MAIN phase")

# プレイ前の状態を詳細に記録
player = game.state.players[0]
print("\n--- Before Card Play ---")
print(f"Hand: {len(player.hand)} cards")
for i, card in enumerate(player.hand):
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}")

print(f"Mana Zone: {len(player.mana_zone)} cards")
untapped_mana = 0
for i, card in enumerate(player.mana_zone):
    tapped_str = "TAPPED" if card.is_tapped else "untapped"
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}, {tapped_str}")
    if not card.is_tapped:
        untapped_mana += 1
print(f"  Untapped mana: {untapped_mana}")

print(f"Battle Zone: {len(player.battle_zone)} cards")
print(f"Graveyard: {len(player.graveyard)} cards")
print(f"Stack: {len(player.stack)} cards")

# アクション生成（コマンド優先）
from dm_toolkit import commands_v2 as commands
import dm_ai_module as _dm

def _get_legal(gs, card_db):
    try:
        cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
    except Exception:
        cmds = []
    if not cmds:
        try:
            try:
                cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
            except Exception:
                try:
                    cmds = commands.generate_legal_commands(gs, card_db) or []
                except Exception:
                    cmds = []
        except Exception:
            cmds = []
    return cmds

actions = _get_legal(game.state, card_db)
declare_play_actions = [a for a in actions if int(a.type) == 15]  # DECLARE_PLAY

if not declare_play_actions:
    print("\nERROR: No DECLARE_PLAY actions available!")
    sys.exit(1)

print(f"\n4. Found {len(declare_play_actions)} playable cards")

# 最初のカードをプレイ
action = declare_play_actions[0]
print(f"\n5. Playing card ID={action.card_id}, IID={action.source_instance_id}")

# カード情報を取得
card_def = card_db[action.card_id]
print(f"   Name: {card_def.name}")
print(f"   Cost: {card_def.cost}")
print(f"   Type: {card_def.type}")

# アクション実行
game.resolve_action(action)

# プレイ後の状態を詳細に記録
player = game.state.players[0]
print("\n--- After Card Play ---")
print(f"Hand: {len(player.hand)} cards")
for i, card in enumerate(player.hand):
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}")

print(f"Mana Zone: {len(player.mana_zone)} cards")
tapped_mana = 0
untapped_mana = 0
for i, card in enumerate(player.mana_zone):
    tapped_str = "TAPPED" if card.is_tapped else "untapped"
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}, {tapped_str}")
    if card.is_tapped:
        tapped_mana += 1
    else:
        untapped_mana += 1
print(f"  Tapped mana: {tapped_mana}")
print(f"  Untapped mana: {untapped_mana}")

print(f"Battle Zone: {len(player.battle_zone)} cards")
for i, card in enumerate(player.battle_zone):
    tapped_str = "TAPPED" if card.is_tapped else "untapped"
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}, {tapped_str}")

print(f"Graveyard: {len(player.graveyard)} cards")
for i, card in enumerate(player.graveyard):
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}")

print(f"Stack: {len(player.stack)} cards")
for i, card in enumerate(player.stack):
    print(f"  [{i}] Card ID={card.card_id}, IID={card.instance_id}")

# 検証
print("\n--- Verification ---")
expected_cost = card_def.cost
success = True

# 1. マナが正しく支払われたか
if tapped_mana >= expected_cost:
    print(f"[OK] Mana paid: {tapped_mana} mana tapped (cost={expected_cost})")
else:
    print(f"[FAIL] Mana not paid correctly: {tapped_mana} tapped (expected >={expected_cost})")
    success = False

# 2. カードがバトルゾーンに移動したか（クリーチャーの場合）
if card_def.type == dm.CardType.CREATURE:
    if len(player.battle_zone) > 0:
        print(f"[OK] Creature summoned to battle zone")
    else:
        print(f"[FAIL] Creature not in battle zone!")
        success = False
elif card_def.type == dm.CardType.SPELL:
    if len(player.graveyard) > 0:
        print(f"[OK] Spell cast to graveyard")
    else:
        print(f"[FAIL] Spell not in graveyard!")
        success = False

# 3. Stackが空か
if len(player.stack) == 0:
    print(f"[OK] Stack is empty (no stuck cards)")
else:
    print(f"[FAIL] Cards stuck on stack: {len(player.stack)}")
    success = False

# 4. 手札が減ったか
if len(player.hand) == 2:  # 3から1枚減って2枚
    print(f"[OK] Card left hand (3 -> 2)")
else:
    print(f"[FAIL] Hand count unexpected: {len(player.hand)}")
    success = False

if success:
    print("\n*** ALL CHECKS PASSED ***")
else:
    print("\n*** SOME CHECKS FAILED ***")

print("\n=== Test Complete ===")
