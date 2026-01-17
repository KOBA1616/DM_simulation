# -*- coding: utf-8 -*-
"""
条件タイプの日本語化テスト
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_condition_type_labels():
    """条件タイプのラベル日本語化テスト"""
    
    print("=" * 80)
    print("条件タイプの日本語化テスト")
    print("=" * 80)
    
    # テスト対象の条件タイプ
    condition_types = [
        "NONE",
        "MANA_ARMED",
        "SHIELD_COUNT",
        "CIVILIZATION_MATCH",
        "OPPONENT_PLAYED_WITHOUT_MANA",
        "OPPONENT_DRAW_COUNT",
        "DURING_YOUR_TURN",
        "DURING_OPPONENT_TURN",
        "FIRST_ATTACK",
        "EVENT_FILTER_MATCH",
        "COMPARE_STAT",
        "COMPARE_INPUT",
        "CARDS_MATCHING_FILTER",
        "DECK_EMPTY",
        "MANA_CIVILIZATION_COUNT",
        "HAND_COUNT",
        "BATTLE_ZONE_COUNT",
        "GRAVEYARD_COUNT",
        "CUSTOM"
    ]
    
    print("\n条件タイプ → 日本語ラベル:")
    print("-" * 80)
    
    for ctype in condition_types:
        label = CardTextResources.get_condition_type_label(ctype)
        status = "✅" if label != ctype else "⚠️"
        print(f"{status} {ctype:35s} → {label}")
    
    # 統計
    print("\n" + "=" * 80)
    total = len(condition_types)
    translated = sum(1 for ct in condition_types if CardTextResources.get_condition_type_label(ct) != ct)
    
    print(f"合計: {total}個")
    print(f"翻訳済み: {translated}個")
    print(f"未翻訳: {total - translated}個")
    print(f"翻訳率: {translated/total*100:.1f}%")
    
    # IF/IF_ELSEで使用される条件タイプのテキスト生成確認
    print("\n" + "=" * 80)
    print("IF判定で使用される条件タイプ（テキスト生成用）")
    print("=" * 80)
    
    if_condition_types = [
        ("OPPONENT_DRAW_COUNT", 2),
        ("COMPARE_STAT", "MY_SHIELD_COUNT >= 3"),
        ("SHIELD_COUNT", "<=2"),
        ("CIVILIZATION_MATCH", "WATER"),
        ("MANA_CIVILIZATION_COUNT", ">=3")
    ]
    
    for ctype, example in if_condition_types:
        label = CardTextResources.get_condition_type_label(ctype)
        print(f"\n{ctype}:")
        print(f"  ラベル: {label}")
        print(f"  例: {example}")


if __name__ == '__main__':
    test_condition_type_labels()
