#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
カードID=1の「optional + up_to」フラグ処理のテスト
"""
import sys
sys.path.insert(0, '.')

import dm_ai_module as dm
from dm_toolkit.data.card_registry import CardRegistry

# カードデータベース読み込み
registry = CardRegistry()
registry.load_from_file("data/cards.json")
card_db = registry.get_all_cards()

# カードID=1の定義を確認
if 1 in card_db:
    card_def = card_db[1]
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
else:
    print("カードID=1が見つかりません")

print("\n=== テスト完了 ===")
print("up_to フィールドが正しく読み込まれていることを確認してください。")
print("次のステップ: GUIでカードID=1をプレイして、枚数選択ダイアログが表示されるか確認")
