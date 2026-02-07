#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
カードID=1の「optional + up_to」フラグ処理のテスト
"""
import sys
sys.path.insert(0, '.')

import dm_ai_module as dm

# カードデータベース読み込み
card_db = dm.load_card_database("data/cards.json")

# カードID=1の定義を確認
card_def = card_db.get(1)
if card_def:
    print(f"=== カードID=1: {card_def.name} ===\n")
    
    for i, eff in enumerate(card_def.effects):
        print(f"Effect {i}: trigger={eff.trigger}")
        for j, act in enumerate(eff.actions):
            print(f"  Action {j}: type={act.type}")
            print(f"    optional={act.optional}")
            print(f"    up_to={act.up_to}")
            print(f"    input_value_key='{act.input_value_key}'")
            print(f"    output_value_key='{act.output_value_key}'")
            print()

# ゲーム状態を作成してテスト
print("=== ゲームシミュレーション開始 ===\n")

# デッキ構築（カードID=1を含む）
deck_p0 = [1] * 10 + [2, 3, 4, 5] * 10  # カードID=1を10枚
deck_p1 = [2, 3, 4, 5] * 10

# ゲーム初期化
game = dm.GameInstance()
game.initialize_game(deck_p0, deck_p1, card_db, seed=12345)

# カードID=1をプレイする状況を作成
state = game.get_current_state()
print(f"初期ターン: {state.turn_number}")
print(f"アクティブプレイヤー: {state.active_player_id}")
print(f"フェーズ: {state.phase}")
print(f"P0手札枚数: {len(state.players[0].hand)}")

# 数ターン進めてカードID=1がプレイできる状況を作る
for turn in range(5):
    actions = game.get_legal_actions()
    if actions:
        # マナチャージまたはパスを選択
        action = actions[0]  # 最初のアクション
        game.apply_action(action)
    else:
        break

print(f"\n5ターン後の状態:")
state = game.get_current_state()
print(f"ターン: {state.turn_number}")
print(f"P0手札: {[c.card_id for c in state.players[0].hand[:5]]}")
print(f"P0マナ: {len(state.players[0].mana_zone)}")

# pending_effectsを確認
if state.pending_effects:
    print(f"\nPending Effects: {len(state.pending_effects)}")
    for i, pe in enumerate(state.pending_effects):
        print(f"  [{i}] type={pe.type} optional={pe.optional}")

print("\n=== テスト完了 ===")
print("GUIで実際にカードID=1をプレイして、枚数選択ダイアログが表示されるか確認してください。")
