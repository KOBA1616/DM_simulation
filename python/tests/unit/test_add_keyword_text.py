# -*- coding: utf-8 -*-
"""
ADD_KEYWORD テキスト生成テスト
再発防止:
  - amount=0 はすべてに適用（選択文なし）
  - amount>0 は「N体選び」テキストを生成
  - 制限系キーワード（CANNOT_ATTACK等）も同様に分岐
  - explicit_self は「このカード」固定
  - skip_selection (入力リンク) は「そのクリーチャーに」固定
"""
import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def _make_add_keyword_cmd(keyword: str, amount: int = 0, duration: str = "",
                           explicit_self: bool = False,
                           target_group: str = "PLAYER_SELF",
                           zones: list = None) -> dict:
    """ADD_KEYWORD コマンドの dict を組み立てるヘルパー"""
    cmd: dict = {
        "type": "ADD_KEYWORD",
        "str_val": keyword,
        "amount": amount,
        "target_group": target_group,
    }
    if duration:
        cmd["duration"] = duration
    if explicit_self:
        cmd["explicit_self"] = True
    if zones:
        cmd["target_filter"] = {"zones": zones}
    return cmd


class TestAddKeywordAmountAll:
    """amount=0（すべてに適用）のテキスト生成"""

    def test_all_creatures_get_blocker(self):
        cmd = _make_add_keyword_cmd("BLOCKER", amount=0)
        text = CardTextGenerator.format_command(cmd)
        assert "体選び" not in text
        assert "「ブロッカー」" in text

    def test_all_with_duration(self):
        cmd = _make_add_keyword_cmd("SPEED_ATTACKER", amount=0, duration="THIS_TURN")
        text = CardTextGenerator.format_command(cmd)
        assert "体選び" not in text
        assert "スピードアタッカー" in text  # CardTextResources の表記に合わせる
        # 期間テキストが含まれる
        assert "このターン" in text

    def test_restriction_all_without_selection(self):
        """制限キーワード + amount=0 → 選択文なし"""
        cmd = _make_add_keyword_cmd("CANNOT_ATTACK", amount=0)
        text = CardTextGenerator.format_command(cmd)
        assert "体選び" not in text
        # "{target}は攻撃できない。" の形式
        assert "攻撃できない" in text


class TestAddKeywordAmountN:
    """amount>0（N体選び）のテキスト生成"""

    def test_select_1_creature_gets_blocker(self):
        cmd = _make_add_keyword_cmd("BLOCKER", amount=1)
        text = CardTextGenerator.format_command(cmd)
        assert "1体選び" in text
        assert "「ブロッカー」" in text

    def test_select_2_creatures_get_speed_attacker(self):
        cmd = _make_add_keyword_cmd("SPEED_ATTACKER", amount=2)
        text = CardTextGenerator.format_command(cmd)
        assert "2体選び" in text
        assert "スピードアタッカー" in text  # CardTextResources の表記に合わせる

    def test_select_1_with_duration(self):
        cmd = _make_add_keyword_cmd("BLOCKER", amount=1, duration="THIS_TURN")
        text = CardTextGenerator.format_command(cmd)
        assert "1体選び" in text
        assert "このターン" in text
        assert "「ブロッカー」" in text

    def test_restriction_select_1(self):
        """制限キーワード + amount=1 → 1体選び → そのクリーチャーは…"""
        cmd = _make_add_keyword_cmd("CANNOT_ATTACK", amount=1)
        text = CardTextGenerator.format_command(cmd)
        assert "1体選び" in text
        assert "攻撃できない" in text

    def test_restriction_select_2(self):
        cmd = _make_add_keyword_cmd("CANNOT_BLOCK", amount=2)
        text = CardTextGenerator.format_command(cmd)
        assert "2体選び" in text
        assert "ブロック" in text or "できない" in text

    def test_target_details_include_owner_zone_and_type(self):
        cmd = _make_add_keyword_cmd("BLOCKER", amount=2, target_group="PLAYER_OPPONENT")
        cmd["target_filter"] = {"zones": ["BATTLE_ZONE"], "types": ["CREATURE"]}
        text = CardTextGenerator.format_command(cmd)
        assert "相手のバトルゾーンのクリーチャー" in text
        assert "2体選び" in text
        assert "ブロッカー" in text


class TestAddKeywordExplicitSelf:
    """explicit_self=True のとき「このカード」固定"""

    def test_this_card_gets_keyword(self):
        cmd = _make_add_keyword_cmd("SPEED_ATTACKER", amount=0, explicit_self=True)
        text = CardTextGenerator.format_command(cmd)
        assert "このカード" in text
        assert "スピードアタッカー" in text  # CardTextResources の表記に合わせる

    def test_this_card_with_duration(self):
        cmd = _make_add_keyword_cmd("BLOCKER", amount=0, duration="THIS_TURN", explicit_self=True)
        text = CardTextGenerator.format_command(cmd)
        assert "このカード" in text
        assert "このターン" in text


class TestAddKeywordFormatKeywordGrantText:
    """_format_keyword_grant_text ヘルパーの直接テスト"""

    def test_normal_keyword_amount_0(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "BLOCKER", "ブロッカー", "", amount=0)
        assert text == "クリーチャーに「ブロッカー」を与える。"

    def test_normal_keyword_amount_1(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "BLOCKER", "ブロッカー", "", amount=1)
        assert text == "クリーチャーを1体選び、「ブロッカー」を与える。"

    def test_normal_keyword_amount_3(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "SPEED_ATTACKER", "スピード・アタッカー", "", amount=3)
        assert text == "クリーチャーを3体選び、「スピード・アタッカー」を与える。"

    def test_normal_keyword_with_duration_amount_0(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "BLOCKER", "ブロッカー", "このターン、", amount=0)
        assert text == "このターン、クリーチャーに「ブロッカー」を与える。"

    def test_normal_keyword_with_duration_amount_2(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "BLOCKER", "ブロッカー", "このターン、", amount=2)
        assert text == "このターン、クリーチャーを2体選び、「ブロッカー」を与える。"

    def test_restriction_amount_0(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "CANNOT_ATTACK", "攻撃できない", "")
        assert text == "クリーチャーは攻撃できない。"

    def test_restriction_amount_1(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "CANNOT_ATTACK", "攻撃できない", "", amount=1)
        assert text == "クリーチャーを1体選び、そのクリーチャーは攻撃できない。"

    def test_restriction_with_duration_amount_0(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "CANNOT_ATTACK", "攻撃できない", "このターン、", amount=0)
        assert text == "このターン、クリーチャーは攻撃できない。"

    def test_restriction_with_duration_amount_2(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "CANNOT_BLOCK", "ブロックできない", "このターン、", amount=2)
        assert text == "クリーチャーを2体選び、このターン、そのクリーチャーはブロックできない。"

    def test_skip_selection_normal_keyword(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "BLOCKER", "ブロッカー", "", skip_selection=True)
        assert text == "そのクリーチャーに「ブロッカー」を与える。"

    def test_skip_selection_restriction(self):
        text = CardTextGenerator._format_keyword_grant_text(
            "クリーチャー", "CANNOT_ATTACK", "攻撃できない", "このターン、", skip_selection=True)
        assert text == "このターン、そのクリーチャーは攻撃できない。"
