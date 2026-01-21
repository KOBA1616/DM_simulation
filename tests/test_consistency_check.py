# -*- coding: utf-8 -*-
import sys
import json
import unittest
from pathlib import Path

# Setup paths
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

class TestConsistencyCheck(unittest.TestCase):

    def test_consistency_card_structure(self):
        """【チェック1】id=9 カード構造の整合性"""

        cards_path = project_root / "data" / "cards.json"
        with open(cards_path, "r", encoding="utf-8") as f:
            cards = json.load(f)

        card = next((c for c in cards if c.get("id") == 9), None)
        self.assertIsNotNone(card, "id=9 カードが見つかりません")

        # Main side check
        main_effects = card.get("effects", [])
        self.assertTrue(len(main_effects) > 0, "メイン側: effectsが空")

        for i, eff in enumerate(main_effects):
            commands = eff.get("commands", [])
            for j, cmd in enumerate(commands):
                cmd_type = cmd.get("type")
                if cmd_type == "APPLY_MODIFIER":
                    required = ["duration", "str_param", "target_filter", "target_group"]
                    for f in required:
                        self.assertIn(f, cmd, f"Main Cmd {j+1}: Missing {f}")
                        self.assertIsNotNone(cmd.get(f), f"Main Cmd {j+1}: None {f}")

        # Spell side check
        spell_side = card.get("spell_side")
        self.assertIsNotNone(spell_side, "spell_side が見つかりません")

        spell_effects = spell_side.get("effects", [])
        for i, eff in enumerate(spell_effects):
            commands = eff.get("commands", [])
            for j, cmd in enumerate(commands):
                cmd_type = cmd.get("type")
                if cmd_type == "SELECT_NUMBER":
                    self.assertTrue(cmd.get("output_value_key"), "Spell Cmd {j+1}: Missing output_value_key")
                elif cmd_type == "APPLY_MODIFIER":
                    required = ["duration", "str_param", "target_filter", "target_group"]
                    for f in required:
                        self.assertIn(f, cmd, f"Spell Cmd {j+1}: Missing {f}")

    def test_consistency_text_generation(self):
        """【チェック2】テキスト生成ロジックの一貫性"""
        test_filters = [
            {"types": ["CREATURE"], "exact_cost": 3},
            {"types": ["SPELL"], "min_cost": 1, "max_cost": 5},
            {"civilizations": ["FIRE"], "min_power": 2000},
            {"cost_ref": "chosen_cost"},
            {"power_max_ref": "max_val", "min_cost": 1},
        ]

        for i, filt in enumerate(test_filters):
            desc = CardTextGenerator.generate_trigger_filter_description(filt)
            trigger_text = CardTextGenerator._apply_trigger_scope(
                "呪文を唱えた時", 
                "PLAYER_OPPONENT",
                "ON_CAST_SPELL",
                filt
            )
            self.assertTrue(desc, f"Test {i+1}: Generated desc is empty")
            self.assertTrue(trigger_text, f"Test {i+1}: Generated trigger text is empty")

    def test_edge_cases(self):
        """【チェック3】エッジケースの処理"""
        edge_cases = [
            {
                "name": "空フィルタ",
                "filter": {},
            },
            {
                "name": "None値の処理",
                "filter": {"min_cost": None, "max_cost": None, "exact_cost": None},
            },
            {
                "name": "0値の処理",
                "filter": {"min_cost": 0, "is_tapped": 0},
            },
            {
                "name": "境界値（最大値）",
                "filter": {"min_cost": 999, "max_cost": 9999},
            },
            {
                "name": "複数フラグ",
                "filter": {
                    "is_blocker": 1,
                    "is_evolution": 1,
                    "is_tapped": 1,
                    "is_summoning_sick": 1
                },
            }
        ]

        for case in edge_cases:
            desc = CardTextGenerator.generate_trigger_filter_description(case["filter"])
            # We don't assert content strictly as long as it doesn't crash,
            # except ensuring it returns a string (even empty is fine for empty filter).
            self.assertIsInstance(desc, str, f"{case['name']} did not return string")

    def test_scope_and_filter_text_generation(self):
        """【チェック4】スコープとフィルタによるテキスト生成"""
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

        for i, case in enumerate(test_cases):
            base_text = CardTextGenerator.trigger_to_japanese("ON_CAST_SPELL", is_spell=False)
            result = CardTextGenerator._apply_trigger_scope(
                base_text,
                case["scope"],
                "ON_CAST_SPELL",
                case["filter"]
            )
            
            for item in case["expected_contains"]:
                self.assertIn(item, result, f"Case {i+1}: Expected '{item}' in result")

    def test_filter_content_coverage(self):
        """【チェック5】フィルタ内容をすべてカバーしているか"""
        test_cases = [
            {"name": "タイプ", "filter": {"types": ["CREATURE"]}, "should_contain": "クリーチャー"},
            {"name": "文明", "filter": {"civilizations": ["FIRE"]}, "should_contain": "火"},
            {"name": "種族", "filter": {"races": ["ドラゴン"]}, "should_contain": "ドラゴン"},
            {"name": "コスト範囲", "filter": {"min_cost": 2, "max_cost": 5}, "should_contain": "コスト"},
            {"name": "固定コスト", "filter": {"exact_cost": 3}, "should_contain": "コスト3"},
            {"name": "コスト参照", "filter": {"cost_ref": "chosen_cost"}, "should_contain": "選択数字"},
            {"name": "パワー", "filter": {"min_power": 3000}, "should_contain": "パワー"},
            {"name": "タップ状態", "filter": {"is_tapped": 1}, "should_contain": "タップ"},
            {"name": "ブロッカー", "filter": {"is_blocker": 1}, "should_contain": "ブロッカー"},
            {"name": "進化", "filter": {"is_evolution": 1}, "should_contain": "進化"},
            {"name": "召喚酔い", "filter": {"is_summoning_sick": 1}, "should_contain": "召喚酔い"}
        ]

        failed = 0
        for case in test_cases:
            desc = CardTextGenerator.generate_trigger_filter_description(case["filter"])
            if case["should_contain"] not in desc:
                failed += 1
                print(f"WARN: {case['name']} coverage check failed. Desc: {desc}")

        # Allow up to 2 failures as per original script tolerance
        self.assertLessEqual(failed, 2, f"Too many coverage failures: {failed}")

if __name__ == '__main__':
    unittest.main()
