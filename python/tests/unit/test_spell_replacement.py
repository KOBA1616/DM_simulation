# -*- coding: utf-8 -*-
"""
呪文の置換移動テスト

呪文を唱えた後、墓地に置かれる代わりに山札の下に置く処理を検証します。
"""

import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


class TestSpellReplacementMove:
    """呪文の置換移動に関するテスト"""
    
    def test_spell_effect_definition(self):
        """呪文効果の定義が正しく記述できることを確認"""
        
        # 呪文効果の定義
        spell_effects = [
            {
                "type": "DRAW_CARD",
                "amount": 2,
                "target_group": "PLAYER_SELF"
            },
            {
                "type": "REPLACE_CARD_MOVE",
                "from_zone": "GRAVEYARD",
                "to_zone": "DECK_BOTTOM",
                "target_group": "SELF"
            }
        ]
        
        # 各効果のテキスト生成を確認
        text1 = CardTextGenerator._format_command(spell_effects[0])
        assert "カードを2枚引く" in text1
        
        text2 = CardTextGenerator._format_command(spell_effects[1])
        assert "墓地に置くかわりに" in text2
        assert "山札の下に置く" in text2
    
    def test_replacement_move_from_stack(self):
        """スタックから墓地への移動が置換されることを確認"""
        
        cmd = {
            "type": "REPLACE_CARD_MOVE",
            "from_zone": "GRAVEYARD",  # 本来の移動先
            "to_zone": "DECK_BOTTOM",   # 実際の移動先
            "target_group": "SELF",
            "comment": "呪文を墓地に置く代わりに山札の下に置く"
        }
        
        # コマンドの構造を確認
        assert cmd["type"] == "REPLACE_CARD_MOVE"
        assert cmd["from_zone"] == "GRAVEYARD"
        assert cmd["to_zone"] == "DECK_BOTTOM"
        
        # テキスト生成
        generated = CardTextGenerator._format_command(cmd)
        assert "このカード" in generated or "そのカード" in generated
        assert "墓地" in generated
        assert "山札の下" in generated
    
    def test_spell_with_self_reference(self):
        """呪文自身を参照する置換移動のテキスト生成"""
        
        cmd = {
            "type": "REPLACE_CARD_MOVE",
            "from_zone": "GRAVEYARD",
            "to_zone": "DECK_BOTTOM",
            "target_group": "SELF",
            "input_value_key": None  # 自己参照（入力なし）
        }
        
        generated = CardTextGenerator._format_command(cmd)
        
        # 自己参照の場合は "このカード" または "そのカード" が使われる
        assert ("このカード" in generated or "そのカード" in generated)
        assert "墓地に置くかわりに" in generated
        assert "山札の下に置く" in generated
    
    def test_various_replacement_destinations(self):
        """様々な置換先ゾーンのテスト"""
        
        test_cases = [
            ("GRAVEYARD", "DECK_BOTTOM", "墓地", "山札の下"),
            ("GRAVEYARD", "DECK", "墓地", "デッキ"),  # DECKは「デッキ」または「山札」
            ("GRAVEYARD", "HAND", "墓地", "手札"),
            ("GRAVEYARD", "MANA_ZONE", "墓地", "マナゾーン"),
        ]
        
        for from_zone, to_zone, expected_from, expected_to in test_cases:
            cmd = {
                "type": "REPLACE_CARD_MOVE",
                "from_zone": from_zone,
                "to_zone": to_zone,
                "target_group": "SELF"
            }
            
            generated = CardTextGenerator._format_command(cmd)
            
            # from_zone のローカライズを確認
            assert expected_from in generated, \
                f"Expected '{expected_from}' in generated text for {from_zone}"
            
            # to_zone のローカライズを確認（「デッキ」または「山札」を許容）
            if to_zone == "DECK":
                assert ("デッキ" in generated or "山札" in generated), \
                    f"Expected 'デッキ' or '山札' in generated text for {to_zone}"
            else:
                assert expected_to in generated, \
                    f"Expected '{expected_to}' in generated text for {to_zone}"
    
    def test_spell_card_full_definition(self):
        """完全な呪文カード定義の例"""
        
        card_def = {
            "card_id": 9001,
            "name": "時空の霧",
            "type": "SPELL",
            "civilization": "WATER",
            "cost": 4,
            "effects": [
                {
                    "type": "DRAW_CARD",
                    "amount": 2,
                    "target_group": "PLAYER_SELF"
                },
                {
                    "type": "REPLACE_CARD_MOVE",
                    "from_zone": "GRAVEYARD",
                    "to_zone": "DECK_BOTTOM",
                    "target_group": "SELF",
                    "comment": "この呪文を墓地に置く代わりに山札の下に置く"
                }
            ]
        }
        
        # カード定義の構造を確認
        assert card_def["type"] == "SPELL"
        assert len(card_def["effects"]) == 2
        assert card_def["effects"][0]["type"] == "DRAW_CARD"
        assert card_def["effects"][1]["type"] == "REPLACE_CARD_MOVE"
        
        # 各効果のテキスト生成
        effect_texts = []
        for effect in card_def["effects"]:
            text = CardTextGenerator._format_command(effect)
            effect_texts.append(text)
        
        # 期待されるテキストパーツが含まれているか確認
        combined_text = "\n".join(effect_texts)
        assert "カードを2枚引く" in combined_text
        assert "墓地に置くかわりに" in combined_text
        assert "山札の下" in combined_text


class TestSpellZoneFlow:
    """呪文のゾーン遷移フローに関するテスト"""
    
    def test_normal_spell_flow_description(self):
        """通常の呪文フロー（置換なし）の説明"""
        
        # 通常フロー: HAND → STACK → GRAVEYARD
        flow = [
            ("HAND", "手札"),
            ("STACK", "スタック"),
            ("GRAVEYARD", "墓地")
        ]
        
        # 各ゾーンが認識されることを確認
        for zone_key, zone_name_jp in flow:
            assert zone_key in ["HAND", "STACK", "GRAVEYARD", "DECK", 
                               "BATTLE_ZONE", "MANA_ZONE", "SHIELD_ZONE"]
    
    def test_replacement_spell_flow_description(self):
        """置換効果ありの呪文フローの説明"""
        
        # 置換フロー: HAND → STACK → DECK_BOTTOM (GRAVEYARD をスキップ)
        flow_with_replacement = [
            ("HAND", "手札", "開始ゾーン"),
            ("STACK", "スタック", "解決中"),
            ("DECK_BOTTOM", "山札の下", "最終ゾーン（置換後）")
        ]
        
        # GRAVEYARDが含まれないことを確認
        zones = [zone for zone, _, _ in flow_with_replacement]
        assert "GRAVEYARD" not in zones, "置換フローでは墓地を経由しない"
        assert "DECK_BOTTOM" in zones, "最終的に山札の下に配置される"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
