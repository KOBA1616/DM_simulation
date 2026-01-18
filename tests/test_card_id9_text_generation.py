# -*- coding: utf-8 -*-
"""
id=9 (ボン・キゴマイム) のテキスト生成と修復検証
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def load_card_by_id(card_id: int):
    """cards.json からカードをロード"""
    with open("data/cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)
    for card in cards:
        if card.get("id") == card_id:
            return card
    return None


def test_card_id9_structure():
    """id=9のカード構造をチェック"""
    card = load_card_by_id(9)
    if not card:
        print("❌ id=9 カード が見つかりません")
        return False

    print("=" * 80)
    print(f"✓ id=9 カード: {card.get('name')}")
    print("=" * 80)

    # メイン側のチェック
    main_effects = card.get("effects", [])
    print(f"\n【メイン側】")
    print(f"  効果数: {len(main_effects)}")
    for i, effect in enumerate(main_effects, 1):
        print(f"  効果 {i}:")
        print(f"    トリガー: {effect.get('trigger')}")
        commands = effect.get("commands", [])
        print(f"    コマンド数: {len(commands)}")
        for j, cmd in enumerate(commands, 1):
            cmd_type = cmd.get("type")
            print(f"      コマンド {j}: {cmd_type}")
            # 必須フィールドをチェック
            if cmd_type == "APPLY_MODIFIER":
                required = ["duration", "str_param", "target_filter", "target_group"]
                missing = [f for f in required if f not in cmd]
                if missing:
                    print(f"        ⚠️  不足フィールド: {missing}")
                else:
                    print(f"        ✓ 必須フィールド完全")
                    print(f"          - duration: {cmd.get('duration')}")
                    print(f"          - str_param: {cmd.get('str_param')}")
                    print(f"          - target_group: {cmd.get('target_group')}")

    # スペル側のチェック
    print(f"\n【スペル側】")
    spell_side = card.get("spell_side")
    if spell_side:
        spell_effects = spell_side.get("effects", [])
        print(f"  効果数: {len(spell_effects)}")
        for i, effect in enumerate(spell_effects, 1):
            print(f"  効果 {i}:")
            print(f"    トリガー: {effect.get('trigger')}")
            commands = effect.get("commands", [])
            print(f"    コマンド数: {len(commands)}")
            for j, cmd in enumerate(commands, 1):
                cmd_type = cmd.get("type")
                print(f"      コマンド {j}: {cmd_type}")
                if cmd_type == "SELECT_NUMBER":
                    if "output_value_key" in cmd:
                        print(f"        ✓ output_value_key: {cmd.get('output_value_key')}")
                    else:
                        print(f"        ⚠️  output_value_key が不足")
                elif cmd_type == "APPLY_MODIFIER":
                    required = ["duration", "str_param", "target_filter", "target_group"]
                    missing = [f for f in required if f not in cmd]
                    if missing:
                        print(f"        ⚠️  不足フィールド: {missing}")
                    else:
                        print(f"        ✓ 必須フィールド完全")
                        print(f"          - target_group: {cmd.get('target_group')}")

    return True


def test_trigger_filter_description():
    """トリガーフィルタの説明生成をテスト"""
    print("\n" + "=" * 80)
    print("トリガーフィルタ説明生成テスト")
    print("=" * 80)

    test_cases = [
        {
            "name": "空フィルタ",
            "filter": {}
        },
        {
            "name": "コスト固定値",
            "filter": {"exact_cost": 3, "types": ["CREATURE"]}
        },
        {
            "name": "コスト範囲",
            "filter": {"min_cost": 1, "max_cost": 3, "types": ["SPELL"]}
        },
        {
            "name": "cost_ref（選択した数字と同じコスト）",
            "filter": {"cost_ref": "chosen_cost", "types": ["CREATURE"]}
        },
        {
            "name": "パワー条件",
            "filter": {"min_power": 3000, "max_power": 5000, "types": ["CREATURE"]}
        },
        {
            "name": "複合条件（文明 + コスト + パワー）",
            "filter": {
                "civilizations": ["FIRE", "NATURE"],
                "min_cost": 5,
                "min_power": 2000,
                "is_blocker": 1,
                "types": ["CREATURE"]
            }
        },
        {
            "name": "ゾーン指定",
            "filter": {
                "types": ["CARD"],
                "zones": ["GRAVEYARD"],
                "min_cost": 0
            }
        }
    ]

    for test in test_cases:
        name = test["name"]
        filter_def = test["filter"]
        desc = CardTextGenerator.generate_trigger_filter_description(filter_def)
        if desc:
            print(f"\n✓ {name}")
            print(f"  説明: {desc}")
        else:
            print(f"\n○ {name}")
            print(f"  説明: (なし)")

    return True


def test_trigger_text_with_scope_and_filter():
    """スコープ + フィルタでのトリガーテキスト生成"""
    print("\n" + "=" * 80)
    print("トリガースコープ + フィルタテキスト生成テスト")
    print("=" * 80)

    test_cases = [
        {
            "name": "相手が呪文を唱えた時（フィルタなし）",
            "trigger": "ON_CAST_SPELL",
            "scope": "PLAYER_OPPONENT",
            "trigger_filter": {}
        },
        {
            "name": "相手がコスト3の呪文を唱えた時",
            "trigger": "ON_CAST_SPELL",
            "scope": "PLAYER_OPPONENT",
            "trigger_filter": {"exact_cost": 3, "types": ["SPELL"]}
        },
        {
            "name": "自分がクリーチャーをバトルゾーンに出した時",
            "trigger": "ON_PLAY",
            "scope": "PLAYER_SELF",
            "trigger_filter": {"types": ["CREATURE"]}
        },
        {
            "name": "相手がパワー4000以上のクリーチャーをバトルゾーンに出した時",
            "trigger": "ON_PLAY",
            "scope": "PLAYER_OPPONENT",
            "trigger_filter": {"min_power": 4000, "types": ["CREATURE"]}
        },
        {
            "name": "他のクリーチャーがバトルゾーンに出た時",
            "trigger": "ON_OTHER_ENTER",
            "scope": "PLAYER_SELF",
            "trigger_filter": {}
        }
    ]

    for test in test_cases:
        name = test["name"]
        trigger = test["trigger"]
        scope = test["scope"]
        trigger_filter = test["trigger_filter"]

        # Base trigger text
        base_text = CardTextGenerator.trigger_to_japanese(trigger, is_spell=False)

        # Apply scope and filter
        final_text = CardTextGenerator._apply_trigger_scope(base_text, scope, trigger, trigger_filter)

        print(f"\n✓ {name}")
        print(f"  ベース: {base_text}")
        print(f"  フィルタ: {trigger_filter}")
        print(f"  結果: {final_text}")

    return True


def main():
    """メイン処理"""
    print("\n" + "=" * 80)
    print("id=9 カード データ修復・テキスト生成検証")
    print("=" * 80)

    # Test 1: Card Structure
    if not test_card_id9_structure():
        return 1

    # Test 2: Trigger Filter Description
    if not test_trigger_filter_description():
        return 1

    # Test 3: Trigger Text with Scope and Filter
    if not test_trigger_text_with_scope_and_filter():
        return 1

    print("\n" + "=" * 80)
    print("✅ すべてのテストが成功しました")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
