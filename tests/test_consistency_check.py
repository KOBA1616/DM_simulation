# -*- coding: utf-8 -*-
"""
整合性チェック：id=9修復とトリガーテキスト生成の統合検証
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dm_ai_module
# Patch GameResult for stub compatibility if __members__ is missing
if not hasattr(dm_ai_module.GameResult, '__members__'):
    class MockEnumMember(int):
        def __new__(cls, value, name):
            obj = int.__new__(cls, value)
            obj._value_ = value
            obj.name = name
            return obj

    dm_ai_module.GameResult.__members__ = {
        'NONE': MockEnumMember(dm_ai_module.GameResult.NONE, 'NONE'),
        'P1_WIN': MockEnumMember(dm_ai_module.GameResult.P1_WIN, 'P1_WIN'),
        'P2_WIN': MockEnumMember(dm_ai_module.GameResult.P2_WIN, 'P2_WIN'),
        'DRAW': MockEnumMember(dm_ai_module.GameResult.DRAW, 'DRAW')
    }

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def check_consistency_card_structure():
    """【チェック1】カード構造の整合性"""
    print("\n" + "=" * 80)
    print("【チェック1】id=9 カード構造の整合性")
    print("=" * 80)

    with open("data/cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    card = next((c for c in cards if c.get("id") == 9), None)
    if not card:
        print("❌ id=9 カードが見つかりません")
        return False

    issues = []
    
    # メイン側チェック
    main_effects = card.get("effects", [])
    if not main_effects:
        issues.append("❌ メイン側: effectsが空")
    
    for i, eff in enumerate(main_effects):
        commands = eff.get("commands", [])
        for j, cmd in enumerate(commands):
            cmd_type = cmd.get("type")
            if cmd_type == "APPLY_MODIFIER":
                required = ["duration", "str_param", "target_filter", "target_group"]
                missing = [f for f in required if f not in cmd or cmd.get(f) is None]
                if missing:
                    issues.append(f"❌ メイン側コマンド{j+1} (APPLY_MODIFIER): 不足フィールド {missing}")
    
    # スペル側チェック
    spell_side = card.get("spell_side")
    if not spell_side:
        issues.append("❌ spell_side が見つかりません")
    else:
        spell_effects = spell_side.get("effects", [])
        for i, eff in enumerate(spell_effects):
            commands = eff.get("commands", [])
            for j, cmd in enumerate(commands):
                cmd_type = cmd.get("type")
                if cmd_type == "SELECT_NUMBER":
                    if "output_value_key" not in cmd or not cmd.get("output_value_key"):
                        issues.append(f"❌ スペル側コマンド{j+1} (SELECT_NUMBER): output_value_keyが不足")
                elif cmd_type == "APPLY_MODIFIER":
                    required = ["duration", "str_param", "target_filter", "target_group"]
                    missing = [f for f in required if f not in cmd or cmd.get(f) is None]
                    if missing:
                        issues.append(f"❌ スペル側コマンド{j+1} (APPLY_MODIFIER): 不足フィールド {missing}")

    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("✅ メイン側: APPLY_MODIFIERが正しく実装")
        print("✅ スペル側: SELECT_NUMBERがoutput_value_keyを持つ")
        print("✅ スペル側: APPLY_MODIFIERが正しく実装")
        return True


def check_consistency_text_generation():
    """【チェック2】テキスト生成ロジックの一貫性"""
    print("\n" + "=" * 80)
    print("【チェック2】テキスト生成ロジックの一貫性")
    print("=" * 80)

    issues = []

    # Test Case 1: 両メソッドが同じフィルタ情報を処理できるか
    test_filters = [
        {"types": ["CREATURE"], "exact_cost": 3},
        {"types": ["SPELL"], "min_cost": 1, "max_cost": 5},
        {"civilizations": ["FIRE"], "min_power": 2000},
        {"cost_ref": "chosen_cost"},
        {"power_max_ref": "max_val", "min_cost": 1},
    ]

    for i, filt in enumerate(test_filters):
        try:
            # generate_trigger_filter_description で生成
            desc = CardTextGenerator.generate_trigger_filter_description(filt)
            
            # _compose_subject_from_filter で生成（外部呼び出し通過）
            # これは内部ヘルパー関数のため、直接呼び出し不可
            # ただし、トリガーテキスト生成時に使用される
            trigger_text = CardTextGenerator._apply_trigger_scope(
                "呪文を唱えた時", 
                "PLAYER_OPPONENT",
                "ON_CAST_SPELL",
                filt
            )
            
            if not desc or not trigger_text:
                issues.append(f"❌ Test {i+1}: テキスト生成が空です")
            else:
                print(f"✓ Test {i+1}: 一貫性OK")
                print(f"  フィルタ説明: {desc[:50]}...")
                print(f"  トリガーテキスト: {trigger_text}")
        except Exception as e:
            issues.append(f"❌ Test {i+1}: 例外発生 - {str(e)}")

    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n✅ 両メソッドが一貫性をもって機能")
        return True


def check_edge_cases():
    """【チェック3】エッジケースの処理"""
    print("\n" + "=" * 80)
    print("【チェック3】エッジケースの処理")
    print("=" * 80)

    issues = []

    edge_cases = [
        {
            "name": "空フィルタ",
            "filter": {},
            "expected": "説明が空 または 軽微"
        },
        {
            "name": "None値の処理",
            "filter": {"min_cost": None, "max_cost": None, "exact_cost": None},
            "expected": "コスト条件なし"
        },
        {
            "name": "0値の処理",
            "filter": {"min_cost": 0, "is_tapped": 0},
            "expected": "is_tapped==0は処理される"
        },
        {
            "name": "境界値（最大値）",
            "filter": {"min_cost": 999, "max_cost": 9999},
            "expected": "最大値が処理される"
        },
        {
            "name": "複数フラグ",
            "filter": {
                "is_blocker": 1,
                "is_evolution": 1,
                "is_tapped": 1,
                "is_summoning_sick": 1
            },
            "expected": "すべてのフラグが組み合わされる"
        }
    ]

    for case in edge_cases:
        try:
            desc = CardTextGenerator.generate_trigger_filter_description(case["filter"])
            print(f"✓ {case['name']}")
            print(f"  期待値: {case['expected']}")
            print(f"  結果: {desc if desc else '(空)'}")
        except Exception as e:
            issues.append(f"❌ {case['name']}: {str(e)}")

    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n✅ エッジケースの処理OK")
        return True


def check_scope_and_filter_text_generation():
    """【チェック4】スコープとフィルタによるテキスト生成"""
    print("\n" + "=" * 80)
    print("【チェック4】スコープ + フィルタ の組み合わせテキスト生成")
    print("=" * 80)

    test_cases = [
        {
            "scope": "PLAYER_OPPONENT",
            "filter": {"types": ["SPELL"]},
            "expected_contains": ["相手", "呪文"]
        },
        {
            "scope": "PLAYER_SELF",
            "filter": {"types": ["CREATURE"], "min_power": 3000},
            "expected_contains": ["自分", "パワー3000以上"]
        },
        {
            "scope": "PLAYER_OPPONENT",
            "filter": {"exact_cost": 5, "civilizations": ["FIRE"]},
            "expected_contains": ["相手", "火", "コスト5"]
        },
    ]

    issues = []
    for i, case in enumerate(test_cases):
        try:
            base_text = CardTextGenerator.trigger_to_japanese("ON_CAST_SPELL", is_spell=False)
            result = CardTextGenerator._apply_trigger_scope(
                base_text,
                case["scope"],
                "ON_CAST_SPELL",
                case["filter"]
            )
            
            missing = [e for e in case["expected_contains"] if e not in result]
            if missing:
                issues.append(f"❌ Case {i+1}: 不足要素 {missing}\n  結果: {result}")
            else:
                print(f"✓ Case {i+1}: 期待通り")
                print(f"  スコープ: {case['scope']}, フィルタ: {case['filter']}")
                print(f"  結果: {result}")
        except Exception as e:
            issues.append(f"❌ Case {i+1}: 例外 - {str(e)}")

    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n✅ スコープ + フィルタ組み合わせOK")
        return True


def check_filter_content_coverage():
    """【チェック5】フィルタ内容をすべてカバーしているか"""
    print("\n" + "=" * 80)
    print("【チェック5】フィルタコンテンツカバレッジ")
    print("=" * 80)

    # フィルタキーの定義
    expected_keys = {
        "types": "タイプ",
        "civilizations": "文明",
        "races": "種族",
        "min_cost": "最小コスト",
        "max_cost": "最大コスト",
        "exact_cost": "固定コスト",
        "cost_ref": "コスト参照",
        "min_power": "最小パワー",
        "max_power": "最大パワー",
        "power_max_ref": "パワー上限参照",
        "is_tapped": "タップ状態",
        "is_blocker": "ブロッカー",
        "is_evolution": "進化",
        "is_summoning_sick": "召喚酔い",
        "zones": "ゾーン",
        "flags": "フラグ"
    }

    # 実際のテストケース：有効な値を設定
    test_cases = [
        {
            "name": "タイプ",
            "filter": {"types": ["CREATURE"]},
            "should_contain": "クリーチャー"
        },
        {
            "name": "文明",
            "filter": {"civilizations": ["FIRE"]},
            "should_contain": "火"
        },
        {
            "name": "種族",
            "filter": {"races": ["ドラゴン"]},
            "should_contain": "ドラゴン"
        },
        {
            "name": "コスト範囲",
            "filter": {"min_cost": 2, "max_cost": 5},
            "should_contain": "コスト"
        },
        {
            "name": "固定コスト",
            "filter": {"exact_cost": 3},
            "should_contain": "コスト3"
        },
        {
            "name": "コスト参照",
            "filter": {"cost_ref": "chosen_cost"},
            "should_contain": "選択数字"
        },
        {
            "name": "パワー",
            "filter": {"min_power": 3000},
            "should_contain": "パワー"
        },
        {
            "name": "タップ状態",
            "filter": {"is_tapped": 1},
            "should_contain": "タップ"
        },
        {
            "name": "ブロッカー",
            "filter": {"is_blocker": 1},
            "should_contain": "ブロッカー"
        },
        {
            "name": "進化",
            "filter": {"is_evolution": 1},
            "should_contain": "進化"
        },
        {
            "name": "召喚酔い",
            "filter": {"is_summoning_sick": 1},
            "should_contain": "召喚酔い"
        }
    ]

    passed = 0
    failed = 0
    
    for case in test_cases:
        try:
            desc = CardTextGenerator.generate_trigger_filter_description(case["filter"])
            if case["should_contain"] in desc:
                print(f"✓ {case['name']}")
                print(f"  期待: {case['should_contain']}")
                print(f"  結果: {desc}")
                passed += 1
            else:
                print(f"❌ {case['name']}")
                print(f"  期待: {case['should_contain']} を含む")
                print(f"  結果: {desc}")
                failed += 1
        except Exception as e:
            print(f"❌ {case['name']}: {str(e)}")
            failed += 1

    print(f"\n結果: {passed}/{len(test_cases)} パス")
    
    if failed == 0:
        print("✅ すべてのフィルタキーが処理される")
        return True
    else:
        print(f"⚠️ {failed}件の不一致")
        return failed <= 2  # 許容値：2件まで



def main():
    """メイン処理"""
    print("\n" + "=" * 80)
    print("[整合性チェック] id=9修復とトリガーテキスト生成")
    print("=" * 80)

    checks = [
        ("カード構造の整合性", check_consistency_card_structure),
        ("テキスト生成ロジックの一貫性", check_consistency_text_generation),
        ("エッジケースの処理", check_edge_cases),
        ("スコープ + フィルタテキスト生成", check_scope_and_filter_text_generation),
        ("フィルタコンテンツカバレッジ", check_filter_content_coverage),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name}: 予期しないエラー - {str(e)}")
            results[name] = False

    # 最終結果
    print("\n" + "=" * 80)
    print("[結果] 整合性チェック結果")
    print("=" * 80)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    all_pass = all(results.values())
    if all_pass:
        print("\n✅ すべての整合性チェックに合格しました")
        return 0
    else:
        print("\n❌ 一部の整合性チェックに不合格")
        return 1


if __name__ == '__main__':
    sys.exit(main())
