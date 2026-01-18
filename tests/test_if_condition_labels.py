# -*- coding: utf-8 -*-
"""
IF判定アクション条件タイプの日本語化統合テスト
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_all_labels():
    """全ての日本語化ラベルを確認"""
    
    print("=" * 80)
    print("IF判定アクション 条件タイプ日本語化 統合テスト")
    print("=" * 80)
    
    # 1. 条件タイプラベル
    print("\n【1. 条件タイプラベル】")
    print("-" * 80)
    
    test_conditions = [
        "OPPONENT_DRAW_COUNT",
        "COMPARE_STAT",
        "SHIELD_COUNT",
        "CIVILIZATION_MATCH",
        "MANA_CIVILIZATION_COUNT",
        "CARDS_MATCHING_FILTER",
        "DECK_EMPTY"
    ]
    
    for ctype in test_conditions:
        label = CardTextResources.get_condition_type_label(ctype)
        print(f"  {ctype:30s} → {label}")
    
    # 2. 統計キーラベル
    print("\n【2. 統計キーラベル（COMPARE_STAT用）】")
    print("-" * 80)
    
    test_stat_keys = [
        "MANA_COUNT",
        "SHIELD_COUNT",
        "HAND_COUNT",
        "OPPONENT_SHIELD_COUNT",
        "MANA_CIVILIZATION_COUNT",
        "BATTLE_ZONE_COUNT"
    ]
    
    for stat_key in test_stat_keys:
        label = CardTextResources.get_stat_key_label(stat_key)
        print(f"  {stat_key:30s} → {label}")
    
    # 3. 実際のIF判定でのテキスト生成例
    print("\n【3. IF判定テキスト生成例】")
    print("-" * 80)
    
    from dm_toolkit.gui.editor.text_generator import CardTextGenerator
    
    # Example 1: OPPONENT_DRAW_COUNT
    if1 = {
        "type": "IF",
        "target_filter": {
            "type": "OPPONENT_DRAW_COUNT",
            "value": 2
        },
        "if_true": [
            {
                "type": "DRAW_CARD",
                "amount": 1,
                "optional": True
            }
        ]
    }
    
    text1 = CardTextGenerator._format_command(if1, is_spell=False)
    print(f"\n  条件: OPPONENT_DRAW_COUNT >= 2")
    print(f"  生成: {text1}")
    
    # Example 2: SHIELD_COUNT (using text_generator IF processing)
    # Note: This needs proper action structure with condition in action dict
    
    print("\n" + "=" * 80)
    print("✅ 全テスト完了")
    print("=" * 80)


if __name__ == '__main__':
    test_all_labels()
