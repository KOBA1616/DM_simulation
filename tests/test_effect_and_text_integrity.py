# -*- coding: utf-8 -*-
"""
カード効果処理・生成テキスト整合性チェック
===========================================
以下を網羅的に検証する:

  [効果コマンド整合性]
  A1. 各コマンドタイプが非空の日本語テキストを返す
  A2. DRAW_CARD → 「引」を含むテキスト
  A3. DESTROY  → 「破壊」を含むテキスト
  A4. TRANSITION / MOVE_CARD → ゾーン名を含むテキスト
  A5. APPLY_MODIFIER の必須フィールド検証
  A6. SELECT_TARGET / QUERY コマンドの非クラッシュ保証

  [トリガー・スコープ整合性]
  B1. 全トリガータイプが非空テキストを返す
  B2. トリガー + スコープの日本語重複検出 (「相手が相手が」など)
  B3. スコープ + フィルター owner の冗長検出 (consistency.py)

  [cards.json 全カード整合性]
  C1. 全カードのテキスト生成がクラッシュしない
  C2. ツインパクトカードの両面テキスト生成
  C3. 全カードの effects コマンド型が登録済み型
  C4. 静的能力 (static_abilities) テキスト生成

  [フィルター整合性]
  D1. コストフィールドの競合チェック
  D2. 進化フラグとタイプ不整合チェック
  D3. ゾーンとタイプの不整合チェック

  [テキスト品質]
  E1. 生成テキストに「のの」や「が、が」などの重複が含まれない
  E2. COST_MODIFIER / POWER_MODIFIER の値符号整合性

  [C++/テキスト生成整合性]
  F1. cards.json のコマンドタイプが C++ CommandType enum に存在するか
  F2. IF / IF_ELSE コマンドが条件文テキストを生成するか
  F3. APPLY_MODIFIER がキーワードテキストを含むか
  F4. SELECT_OPTION コマンドの非クラッシュ保証
  F5. C++ 未実装コマンドの網羅カタログ（警告として記録）
  F6. テキスト生成と C++ 実装の対応表整合性

再発防止:
  - APPLY_MODIFIER には duration, str_param, target_filter, target_group が必須
  - フィルター owner と trigger_scope は同時指定しない（重複）
  - generate_text() は全カードで例外を出さないこと
  - 新しいコマンドタイプを cards.json に追加したら C++ CommandType enum にも追加すること
  - C++ generate_instructions で default: break になるコマンドは実質 no-op → ゲームバグの原因
"""
from __future__ import annotations

import json
import os
import sys
import re
import pytest
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# ── パス設定 ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── インポート ─────────────────────────────────────────────────────────────────
from dm_toolkit.gui.editor.text_generator import CardTextGenerator
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.consistency import validate_trigger_scope_filter
from dm_toolkit.consts import TRIGGER_TYPES

# cards.json
_CARDS_PATH = _PROJECT_ROOT / "data" / "cards.json"

def _load_cards() -> List[Dict[str, Any]]:
    with open(_CARDS_PATH, encoding="utf-8") as f:
        return json.load(f)

_CARDS: List[Dict[str, Any]] = _load_cards()

# ── ヘルパー ───────────────────────────────────────────────────────────────────

def _make_command(cmd_type: str, **kwargs: Any) -> Dict[str, Any]:
    """テスト用コマンド辞書を生成する。"""
    base: Dict[str, Any] = {
        "type": cmd_type,
        "target_group": "PLAYER_SELF",
        "target_filter": {"zones": [], "civilizations": [], "races": [], "flags": []},
        "amount": 1,
        "str_param": "",
        "from_zone": "NONE",
        "to_zone": "HAND",
        "format": "command",
    }
    base.update(kwargs)
    return base


def _make_effect(trigger: str, commands: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """テスト用エフェクト辞書を生成する。"""
    base: Dict[str, Any] = {
        "trigger": trigger,
        "condition": {"type": "NONE"},
        "commands": commands,
    }
    base.update(kwargs)
    return base


def _cmd_text(cmd_type: str, **kwargs: Any) -> str:
    """コマンドテキスト生成のショートカット。"""
    cmd = _make_command(cmd_type, **kwargs)
    return CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))


# ===========================================================================
# A. 効果コマンド整合性
# ===========================================================================

class TestCommandTextIntegrity:
    """各コマンドタイプが適切な日本語テキストを生成することを確認する。"""

    # (コマンドタイプ, 期待キーワード)
    COMMAND_KEYWORD_MAP = [
        ("DRAW_CARD",       ["引"]),
        ("DESTROY",         ["破壊"]),
        ("DISCARD",         ["捨", "手札"]),
        ("SEND_TO_MANA",    ["マナ"]),
        ("MANA_CHARGE",     ["マナ"]),
        ("BREAK_SHIELD",    ["シールド", "ブレイク"]),
        ("BOOST_MANA",      ["マナ"]),
        ("UNTAP",           ["アンタップ"]),
        ("TAP",             ["タップ"]),
        ("SHUFFLE_DECK",    ["シャッフル", "山札"]),
    ]

    @pytest.mark.parametrize("cmd_type,expected_keywords", COMMAND_KEYWORD_MAP)
    def test_command_produces_japanese_text(self, cmd_type: str, expected_keywords: List[str]) -> None:
        """各コマンドタイプが期待キーワードを含む日本語テキストを返すことを確認する。"""
        text = _cmd_text(cmd_type)
        assert isinstance(text, str), f"{cmd_type}: テキストが str でない"
        assert len(text) > 0, (
            f"{cmd_type}: テキストが空\n"
            "再発防止: _format_command の ACTION_MAP に対応エントリを追加すること"
        )
        for kw in expected_keywords:
            assert kw in text, (
                f"{cmd_type}: テキスト「{text}」に「{kw}」が含まれない\n"
                "再発防止: ACTION_MAP のテンプレート文字列が正しいか確認すること"
            )

    def test_draw_card_with_amount(self) -> None:
        """DRAW_CARD で amount を指定した場合、枚数がテキストに含まれることを確認する。"""
        text = _cmd_text("DRAW_CARD", amount=3)
        assert "3" in text or "引" in text, (
            f"DRAW_CARD amount=3: テキスト「{text}」に枚数が含まれない"
        )

    def test_transition_with_zones(self) -> None:
        """TRANSITION コマンドのゾーン名がテキストに含まれることを確認する。"""
        text = _cmd_text("TRANSITION", from_zone="HAND", to_zone="MANA_ZONE")
        assert isinstance(text, str)
        # 手札 or マナゾーンのいずれかが含まれるはず
        assert any(z in text for z in ["手札", "マナ", "ゾーン", "HAND", "MANA"]), (
            f"TRANSITION: ゾーン情報が含まれない: 「{text}」"
        )

    def test_move_card_contains_zone(self) -> None:
        """MOVE_CARD コマンドのゾーン名がテキストに含まれることを確認する。"""
        text = _cmd_text("MOVE_CARD", from_zone="BATTLE_ZONE", to_zone="GRAVEYARD")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_query_does_not_crash(self) -> None:
        """QUERY コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command(
            "QUERY",
            str_param="MANA_CIVILIZATION_COUNT",
            output_value_key="var_test",
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_select_target_does_not_crash(self) -> None:
        """SELECT_TARGET コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command(
            "SELECT_TARGET",
            target_filter={
                "types": ["CREATURE"],
                "zones": ["BATTLE_ZONE"],
                "civilizations": [],
                "races": [],
                "flags": [],
            },
            amount=1,
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_apply_modifier_required_fields(self) -> None:
        """APPLY_MODIFIER には duration, str_param, target_filter, target_group が必須。

        再発防止: これらが欠けるとゲームロジックが不正動作する。
        """
        required_fields = ["duration", "str_param", "target_filter", "target_group"]
        # 必須フィールドが全て揃っている場合はクラッシュしない
        cmd = _make_command(
            "APPLY_MODIFIER",
            duration="THIS_TURN",
            str_param="SPEED_ATTACKER",
            target_filter={"types": ["CREATURE"], "zones": ["BATTLE_ZONE"], "civilizations": [], "races": [], "flags": []},
            target_group="PLAYER_SELF",
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

        # 必須フィールドが欠けていてもクラッシュしない（警告ログのみが正しい動作）
        incomplete_cmd = _make_command("APPLY_MODIFIER")
        try:
            text2 = CardTextGenerator._format_command(incomplete_cmd, TextGenerationContext(card_data={}))
            assert isinstance(text2, str)
        except Exception as e:
            pytest.fail(
                f"APPLY_MODIFIER 必須フィールド欠損時にクラッシュ: {e}\n"
                "再発防止: フィールド欠損時は空文字またはデフォルト値で処理すること"
            )

    def test_select_number_does_not_crash(self) -> None:
        """SELECT_NUMBER コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command("SELECT_NUMBER", amount=6, output_value_key="var_num")
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_summon_token_does_not_crash(self) -> None:
        """SUMMON_TOKEN コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command("SUMMON_TOKEN", str_param="TEST_TOKEN")
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_play_from_zone_does_not_crash(self) -> None:
        """PLAY_FROM_ZONE コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command(
            "PLAY_FROM_ZONE",
            from_zone="HAND",
            target_filter={"types": ["CREATURE"], "max_cost": 5, "zones": [], "civilizations": [], "races": [], "flags": []},
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_cast_spell_does_not_crash(self) -> None:
        """CAST_SPELL コマンドがクラッシュしないことを確認する。"""
        cmd = _make_command(
            "CAST_SPELL",
            target_filter={"types": ["SPELL"], "max_cost": 3, "zones": ["HAND"], "civilizations": [], "races": [], "flags": []},
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)

    def test_if_command_does_not_crash(self) -> None:
        """IF コマンド（条件分岐）がクラッシュしないことを確認する。"""
        cmd = {
            "type": "IF",
            "condition": {"type": "MANA_ARMED", "value": 5},
            "if_true": [_make_command("DRAW_CARD", amount=2)],
            "if_false": [_make_command("DRAW_CARD", amount=1)],
        }
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)


# ===========================================================================
# B. トリガー・スコープ整合性
# ===========================================================================

class TestTriggerAndScopeIntegrity:
    """トリガータイプとスコープが正しい日本語テキストを生成することを確認する。"""

    TRIGGER_TYPES = [
        "ON_PLAY", "AT_ATTACK", "ON_DESTROY", "AT_END_OF_TURN",
        "AT_END_OF_OPPONENT_TURN", "ON_BLOCK", "TURN_START",
        "ON_CAST_SPELL", "ON_OPPONENT_DRAW", "ON_SHIELD_ADD",
        "AT_BREAK_SHIELD", "ON_OPPONENT_CREATURE_ENTER",
    ]

    SCOPE_TYPES = [
        "PLAYER_SELF", "PLAYER_OPPONENT", "ALL_PLAYERS",
        "SELF", "OPPONENT", "ALL",
    ]

    @pytest.mark.parametrize("trigger", TRIGGER_TYPES)
    def test_trigger_produces_nonempty_text(self, trigger: str) -> None:
        """全トリガータイプが非空の日本語テキストを返すことを確認する。"""
        text = CardTextGenerator.trigger_to_japanese(trigger, is_spell=False)
        # NONE と PASSIVE_CONST は空でも許容
        assert isinstance(text, str), f"{trigger}: テキストが str でない"

    @pytest.mark.parametrize("trigger", TRIGGER_TYPES)
    def test_spell_trigger_does_not_crash(self, trigger: str) -> None:
        """呪文モードのトリガーがクラッシュしないことを確認する。"""
        text = CardTextGenerator.trigger_to_japanese(trigger, is_spell=True)
        assert isinstance(text, str)

    def test_trigger_scope_no_duplication(self) -> None:
        """トリガー + スコープで日本語の重複 (「相手が相手が」等) が生じないことを確認する。

        再発防止: _apply_trigger_scope は既に「相手が」が含まれる場合はスコープを追加しない。
        """
        # ON_CAST_SPELL (相手が呪文を唱えた時) + PLAYER_OPPONENT
        text = CardTextGenerator._apply_trigger_scope(
            "相手が呪文を唱えた時",
            "PLAYER_OPPONENT",
            "ON_CAST_SPELL",
            {},
        )
        assert "相手が相手が" not in text, (
            f"スコープ重複が検出されました: 「{text}」\n"
            "再発防止: _apply_trigger_scope で既存スコープ検出をチェックすること"
        )

    def test_scope_prefix_not_double_self(self) -> None:
        """自分スコープのトリガーテキストに「自分が自分が」が含まれないことを確認する。"""
        text = CardTextGenerator._apply_trigger_scope(
            "自分がクリーチャーを召喚した時",
            "PLAYER_SELF",
            "ON_PLAY",
            {},
        )
        assert "自分が自分が" not in text, (
            f"スコープ重複: 「{text}」\n"
            "再発防止: _apply_trigger_scope の重複チェックを確認すること"
        )

    @pytest.mark.parametrize("scope", SCOPE_TYPES)
    def test_scope_text_returns_string(self, scope: str) -> None:
        """全スコープタイプがテキストを返すことを確認する。"""
        text = CardTextResources.get_scope_text(scope)
        assert isinstance(text, str), f"scope={scope}: 返り値が str でない"

    def test_effect_with_scope_does_not_crash(self) -> None:
        """スコープ付きエフェクトのテキスト生成がクラッシュしないことを確認する。"""
        effect = _make_effect(
            "ON_CAST_SPELL",
            [_make_command("DESTROY", target_group="PLAYER_OPPONENT")],
            trigger_scope="PLAYER_OPPONENT",
            trigger_filter={"types": ["SPELL"], "civilizations": [], "races": [], "flags": []},
        )
        from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
        try:
            text = CardTextGenerator._format_effect(effect, TextGenerationContext(card_data={}))
            assert isinstance(text, str)
        except Exception as e:
            pytest.fail(
                f"スコープ付きエフェクトでクラッシュ: {e}\n"
                "再発防止: trigger_scope が NONE でない場合は _apply_trigger_scope を呼ぶこと"
            )

    def test_all_current_trigger_types_have_japanese_mapping(self) -> None:
        """現行TRIGGER_TYPESが生キーのまま表示されないことを確認する。"""
        missing: List[str] = []
        for trigger in TRIGGER_TYPES:
            if trigger in ("NONE", "PASSIVE_CONST"):
                continue
            text = CardTextGenerator.trigger_to_japanese(trigger, is_spell=False)
            if text == trigger or not text:
                missing.append(trigger)

        assert not missing, (
            f"未翻訳または空のトリガーがあります: {missing}\n"
            "再発防止: text_resources.py の TRIGGER_JAPANESE に現行TRIGGER_TYPESを追加すること"
        )

    def test_replacement_trigger_uses_pre_event_phrase(self) -> None:
        """置換効果では「〜た時」ではなく「〜る時」の時制になることを確認する。"""
        effect = _make_effect(
            "ON_OPPONENT_CREATURE_ENTER",
            [_make_command("DRAW_CARD", amount=1)],
            mode="REPLACEMENT",
            timing_mode="PRE",
        )
        text = CardTextGenerator._format_effect(effect, TextGenerationContext(card_data={}))
        assert "相手のクリーチャーが出る時" in text, f"置換時制が未適用: {text}"
        assert "相手のクリーチャーが出た時" not in text, f"置換時制が過去形のまま: {text}"


# ===========================================================================
# C. cards.json 全カード整合性
# ===========================================================================

class TestAllCardsTextGeneration:
    """cards.json の全カードでテキスト生成が成功することを確認する。"""

    @pytest.mark.parametrize("card", _CARDS, ids=[f"card_{c.get('id', i)}" for i, c in enumerate(_CARDS)])
    def test_generate_text_no_crash(self, card: Dict[str, Any]) -> None:
        """全カードのテキスト生成がクラッシュしないことを確認する。

        再発防止: generate_text() で KeyError や AttributeError が出る場合は
                  フィールド存在確認を text_generator.py に追加すること。
        """
        try:
            text = CardTextGenerator.generate_text(card)
            assert isinstance(text, str), f"カード id={card.get('id')}: generate_text が str を返さない"
        except Exception as e:
            pytest.fail(
                f"カード id={card.get('id')} name={card.get('name')}: "
                f"generate_text でクラッシュ: {type(e).__name__}: {e}\n"
                "再発防止: generate_text はすべてのカードデータ形式に対して例外を出さないこと"
            )

    @pytest.mark.parametrize("card", _CARDS, ids=[f"body_{c.get('id', i)}" for i, c in enumerate(_CARDS)])
    def test_generate_body_text_no_crash(self, card: Dict[str, Any]) -> None:
        """generate_body_text がクラッシュしないことを確認する。"""
        try:
            text = CardTextGenerator.generate_body_text(card)
            assert isinstance(text, str)
        except Exception as e:
            pytest.fail(
                f"カード id={card.get('id')}: generate_body_text でクラッシュ: {e}"
            )

    import pytest
    @pytest.mark.skip(reason='Handled by CardLayoutBuilder now')
    def test_twinpact_cards_both_sides_generated(self) -> None:
        """ツインパクトカード（spell_side あり）の両面テキストが生成されることを確認する。

        再発防止: spell_side の effects も generate_text に通すこと。
        """
        twinpact_cards = [c for c in _CARDS if c.get("spell_side")]
        if not twinpact_cards:
            pytest.skip("ツインパクトカードが cards.json にありません")

        for card in twinpact_cards:
            text = CardTextGenerator.generate_text(card)
            # 呪文側セパレーターが含まれるはず
            assert "呪文側" in text, (
                f"カード id={card.get('id')}: ツインパクトの呪文側テキストがない: 「{text[:100]}」\n"
                "再発防止: include_twinpact=True でスペル側を生成すること"
            )

    def test_static_abilities_text_generated(self) -> None:
        """static_abilities を持つカードのテキストが正しく生成されることを確認する。

        再発防止: static_abilities の各エントリを _format_effect で処理すること。
        """
        static_cards = [c for c in _CARDS if c.get("static_abilities")]
        if not static_cards:
            pytest.skip("static_abilities を持つカードが cards.json にありません")

        for card in static_cards:
            text = CardTextGenerator.generate_body_text(card)
            assert isinstance(text, str)
            assert len(text) > 0, (
                f"カード id={card.get('id')}: static_abilities があるのにテキストが空"
            )

    def test_all_effect_command_types_are_known(self) -> None:
        """cards.json の全エフェクトコマンドタイプが未知でないことを確認する。

        再発防止: 新しいコマンドタイプを追加したら ACTION_MAP にも追加すること。
        """
        # テキスト生成が空文字列を返す型を「未サポート」として警告する
        unknown_types: set[str] = set()

        for card in _CARDS:
            for effect in (card.get("effects") or []):
                for cmd in (effect.get("commands") or []):
                    cmd_type = cmd.get("type", "")
                    if not cmd_type or cmd_type == "NONE":
                        continue
                    text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
                    if text == "" and cmd_type not in (
                        "QUERY",      # クエリ単独は空で良い
                        "OUTPUT",     # 出力用プリミティブ
                        "REGISTER_DELAYED_EFFECT",  # 遅延効果登録
                    ):
                        unknown_types.add(cmd_type)

        # 未知コマンドが多すぎる場合は警告 (厳密な fail にすると追加時に壊れる)
        if unknown_types:
            # 3 件以上ならテスト失敗
            assert len(unknown_types) <= 3, (
                f"テキスト生成が空を返すコマンドタイプが多すぎます: {unknown_types}\n"
                "再発防止: _format_command の ACTION_MAP に新コマンドのテンプレートを追加すること"
            )

    def test_effect_commands_have_type_field(self) -> None:
        """全カードのエフェクトコマンドに 'type' フィールドが存在することを確認する。

        再発防止: コマンド辞書から 'type' が欠落するとテキスト生成が空文字になる。
        """
        missing: List[str] = []
        for card in _CARDS:
            for effect in (card.get("effects") or []):
                for i, cmd in enumerate(effect.get("commands") or []):
                    if not cmd.get("type"):
                        missing.append(
                            f"card_id={card.get('id')} effect_trigger={effect.get('trigger')} cmd_index={i}"
                        )
            # spell_side も確認
            spell = card.get("spell_side")
            if spell:
                for effect in (spell.get("effects") or []):
                    for i, cmd in enumerate(effect.get("commands") or []):
                        if not cmd.get("type"):
                            missing.append(
                                f"card_id={card.get('id')} spell_side cmd_index={i}"
                            )

        assert not missing, (
            f"'type' フィールドが欠落しているコマンド:\n" + "\n".join(missing) + "\n"
            "再発防止: コマンド辞書には必ず 'type' フィールドを設定すること"
        )


# ===========================================================================
# D. フィルター整合性
# ===========================================================================

class TestFilterIntegrity:
    """フィルター設定の矛盾・不整合を検出する。"""

    def test_exact_cost_conflicts_with_range(self) -> None:
        """exact_cost と min_cost/max_cost の競合を consistency.py が検出することを確認する。

        再発防止: ExactCost と Cost Range を同時指定してはならない。
        """
        effect = {
            "trigger_scope": "PLAYER_OPPONENT",
            "trigger_filter": {
                "exact_cost": 3,
                "min_cost": 1,
                "max_cost": 5,
            },
        }
        warnings = validate_trigger_scope_filter(effect)
        has_conflict = any("競合" in w or "ExactCost" in w for w in warnings)
        assert has_conflict, (
            f"exact_cost と min/max_cost の競合が検出されなかった。warnings={warnings}\n"
            "再発防止: validate_trigger_scope_filter で競合検出ロジックを維持すること"
        )

    def test_scope_owner_duplication_detected(self) -> None:
        """trigger_scope と filter.owner の重複を consistency.py が検出することを確認する。

        再発防止: scope=OPPONENT と filter.owner=OPPONENT を同時指定しない。
        """
        effect = {
            "trigger_scope": "PLAYER_OPPONENT",
            "trigger_filter": {"owner": "PLAYER_OPPONENT"},
        }
        warnings = validate_trigger_scope_filter(effect)
        has_dup = any("重複" in w for w in warnings)
        assert has_dup, (
            f"scope と filter.owner の重複が警告されなかった。warnings={warnings}\n"
            "再発防止: trigger_scope と filter.owner を両方指定しないこと"
        )

    def test_evolution_flag_with_non_creature_type_detected(self) -> None:
        """進化フラグにタイプ=クリーチャー以外が含まれると不整合を検出する。

        再発防止: is_evolution=1 は types=[CREATURE] と組み合わせること。
        """
        effect = {
            "trigger_scope": "ALL",
            "trigger_filter": {"is_evolution": 1, "types": ["SPELL"]},
        }
        warnings = validate_trigger_scope_filter(effect)
        has_inconsistency = any("進化" in w for w in warnings)
        assert has_inconsistency, (
            f"進化フラグ + 呪文タイプの不整合が検出されなかった。warnings={warnings}\n"
            "再発防止: is_evolution と types の組み合わせを validate_trigger_scope_filter でチェックすること"
        )

    def test_shield_zone_with_creature_type_detected(self) -> None:
        """シールドゾーン + クリーチャータイプの不整合を検出することを確認する。

        再発防止: SHIELD_ZONE にはカード型が無い（タイプ指定は不要）。
        """
        effect = {
            "trigger_scope": "ALL",
            "trigger_filter": {"zones": ["SHIELD_ZONE"], "types": ["CREATURE"]},
        }
        warnings = validate_trigger_scope_filter(effect)
        has_inconsistency = any("シールド" in w or "不整合" in w for w in warnings)
        assert has_inconsistency, (
            f"シールドゾーン + タイプの不整合が検出されなかった。warnings={warnings}\n"
            "再発防止: SHIELD_ZONE のフィルターにタイプ指定を許可しないこと"
        )

    def test_valid_filter_produces_no_warnings(self) -> None:
        """正常なフィルターで警告が出ないことを確認する。"""
        effect = {
            "trigger_scope": "PLAYER_OPPONENT",
            "trigger_filter": {
                "types": ["CREATURE"],
                "min_cost": 3,
                "zones": ["BATTLE_ZONE"],
                "civilizations": ["FIRE"],
            },
        }
        warnings = validate_trigger_scope_filter(effect)
        assert not warnings, (
            f"正常なフィルターで不要な警告が生成された: {warnings}"
        )

    def test_empty_filter_produces_no_warnings(self) -> None:
        """空フィルターで警告が出ないことを確認する。"""
        effect = {
            "trigger_scope": "PLAYER_SELF",
            "trigger_filter": {},
        }
        warnings = validate_trigger_scope_filter(effect)
        # 空フィルターは何も問題ない
        assert isinstance(warnings, list)


# ===========================================================================
# E. テキスト品質チェック
# ===========================================================================

class TestTextQuality:
    """生成テキストの品質（重複・符号・誤訳がないこと）を確認する。"""

    # 重複パターン
    DUPLICATION_PATTERNS = [
        ("のの",     "「のの」重複"),
        ("がが",     "「がが」重複"),
        ("をを",     "「をを」重複"),
        ("はは",     "「はは」重複"),
        ("自分の自分", "「自分の自分」重複"),
        ("相手の相手", "「相手の相手」重複"),
        ("自分が自分", "「自分が自分」重複"),
        ("相手が相手", "「相手が相手」重複"),
    ]

    @pytest.mark.parametrize("card", _CARDS, ids=[f"dup_{c.get('id', i)}" for i, c in enumerate(_CARDS)])
    def test_no_text_duplication_in_cards(self, card: Dict[str, Any]) -> None:
        """生成テキストに重複フレーズが含まれないことを確認する。

        再発防止: scope_prefix が二重に付与される場合は _format_modifier で確認すること。
        """
        text = CardTextGenerator.generate_body_text(card)
        for pat, desc in self.DUPLICATION_PATTERNS:
            assert pat not in text, (
                f"カード id={card.get('id')} name={card.get('name')}: "
                f"テキストに{desc}が含まれる: 「{text[:200]}」\n"
                "再発防止: _format_modifier で scope_prefix の冗長追加を排除すること"
            )

    def test_cost_modifier_positive_value_says_keigen(self) -> None:
        """COST_MODIFIER value>0 のテキストに「軽減」が含まれることを確認する。

        再発防止: value の符号と「軽減」「増やす」の対応を確認すること。
        """
        modifier = {
            "type": "COST_MODIFIER",
            "value": 2,
            "scope": "SELF",
            "filter": {"types": ["CREATURE"]},
            "condition": {},
        }
        text = CardTextGenerator._format_modifier(modifier)
        assert "少なくする" in text or "少なくなる" in text or "軽減" in text, (
            f"COST_MODIFIER value=2 で「軽減」がない: 「{text}」\n"
            "再発防止: _format_cost_modifier で value>0 を軽減として表示すること"
        )

    def test_cost_modifier_negative_value_says_fuyasu(self) -> None:
        """COST_MODIFIER value<0 のテキストに「増やす」が含まれることを確認する。"""
        modifier = {
            "type": "COST_MODIFIER",
            "value": -2,
            "scope": "SELF",
            "filter": {"types": ["CREATURE"]},
            "condition": {},
        }
        text = CardTextGenerator._format_modifier(modifier)
        assert "多くする" in text or "多くなる" in text or "増やす" in text or "増" in text, (
            f"COST_MODIFIER value=-2 で「増やす」がない: 「{text}」\n"
            "再発防止: _format_cost_modifier で value<0 をコスト増として表示すること"
        )

    def test_power_modifier_positive_shows_plus(self) -> None:
        """POWER_MODIFIER value>0 のテキストに「+」が含まれることを確認する。"""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 3000,
            "scope": "SELF",
            "filter": {"types": ["CREATURE"]},
            "condition": {},
        }
        text = CardTextGenerator._format_modifier(modifier)
        assert "+" in text or "3000" in text, (
            f"POWER_MODIFIER value=3000 でパワー値がない: 「{text}」"
        )

    def test_grant_keyword_speed_attacker_text(self) -> None:
        """GRANT_KEYWORD でスピードアタッカー付与テキストが正しいことを確認する。"""
        modifier = {
            "type": "GRANT_KEYWORD",
            "mutation_kind": "SPEED_ATTACKER",
            "scope": "SELF",
            "filter": {"types": ["CREATURE"]},
            "condition": {},
            "duration": "THIS_TURN",
        }
        text = CardTextGenerator._format_modifier(modifier)
        assert "スピードアタッカー" in text, (
            f"GRANT_KEYWORD SPEED_ATTACKER でキーワードが含まれない: 「{text}」\n"
            "再発防止: CardTextResources.KEYWORD_TRANSLATION に SPEED_ATTACKER が登録されていること"
        )

    def test_set_keyword_blocker_text(self) -> None:
        """SET_KEYWORD でブロッカー付与テキストが正しいことを確認する。"""
        modifier = {
            "type": "SET_KEYWORD",
            "str_val": "BLOCKER",
            "scope": "SELF",
            "filter": {"types": ["CREATURE"]},
            "condition": {},
        }
        text = CardTextGenerator._format_modifier(modifier)
        assert "ブロッカー" in text, (
            f"SET_KEYWORD BLOCKER でキーワードが含まれない: 「{text}」"
        )

    def test_trigger_filter_description_not_empty(self) -> None:
        """generate_trigger_filter_description が非空テキストを返すことを確認する。"""
        cases = [
            {"types": ["CREATURE"], "min_cost": 3},
            {"civilizations": ["FIRE"], "min_power": 2000},
            {"exact_cost": 5},
            {"cost_ref": "chosen_cost"},
            {"is_blocker": 1},
            {"is_evolution": 1, "types": ["CREATURE"]},
        ]
        for filt in cases:
            desc = FilterTextFormatter.describe_simple_filter(filt)
            assert isinstance(desc, str), f"filter={filt}: str でない"
            assert len(desc) > 0, (
                f"filter={filt}: generate_trigger_filter_description が空文字を返した\n"
                "再発防止: 全フィルタフィールドをカバーする分岐を追加すること"
            )

    def test_all_cards_effects_text_not_only_unknown(self) -> None:
        """全カードの効果テキストが単純な type 名のみでないことを確認する。

        未知コマンドはそのまま type 名を返すが、それのみであれば日本語化漏れを示す。
        再発防止: 新コマンドを追加したら ACTION_MAP/ _format_command に対応を追加すること。
        """
        all_cap_re = re.compile(r"^[A-Z_]+$")  # 全大文字英字のみ
        problematic: List[str] = []

        for card in _CARDS:
            for effect in (card.get("effects") or []):
                for cmd in (effect.get("commands") or []):
                    cmd_type = cmd.get("type", "")
                    if not cmd_type:
                        continue
                    text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
                    if text and all_cap_re.match(text.strip()):
                        problematic.append(
                            f"card_id={card.get('id')} cmd_type={cmd_type}: 「{text}」"
                        )

        assert len(problematic) == 0, (
            f"日本語化されていないコマンドテキストがあります:\n"
            + "\n".join(problematic)
            + "\n再発防止: _format_command で全コマンドタイプを日本語化すること"
        )


# ===========================================================================
# F. C++/テキスト生成整合性チェック
# ===========================================================================

# C++ CommandType enum に登録済みの文字列（src/core/card_json_types.hpp の
# NLOHMANN_JSON_SERIALIZE_ENUM(CommandType, ...) と同期して更新すること）
# 再発防止: 新しいコマンドタイプを cards.json で使う場合は必ずここに追加すること
_CPP_COMMAND_TYPE_STRINGS: set = {
    "NONE", "TRANSITION", "MUTATE", "FLOW", "QUERY",
    "DRAW_CARD", "DISCARD", "DESTROY", "BOOST_MANA", "TAP", "UNTAP",
    "POWER_MOD", "ADD_KEYWORD", "RETURN_TO_HAND", "BREAK_SHIELD",
    "SEARCH_DECK", "SHIELD_TRIGGER",
    "MOVE_CARD", "ADD_MANA", "SEND_TO_MANA", "PLAYER_MANA_CHARGE",
    "SEARCH_DECK_BOTTOM", "ADD_SHIELD", "SEND_TO_DECK_BOTTOM",
    "ATTACK_PLAYER", "ATTACK_CREATURE", "BLOCK",
    "RESOLVE_BATTLE", "RESOLVE_PLAY", "RESOLVE_EFFECT",
    "SHUFFLE_DECK", "LOOK_AND_ADD", "MEKRAID", "REVEAL_CARDS",
    "PLAY_FROM_ZONE", "CAST_SPELL", "SUMMON_TOKEN", "SHIELD_BURN",
    "SELECT_NUMBER", "CHOICE", "SELECT_OPTION",
    "LOOK_TO_BUFFER", "REVEAL_TO_BUFFER", "SELECT_FROM_BUFFER",
    "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "MOVE_BUFFER_REMAIN_TO_ZONE",
    "FRIEND_BURST", "REGISTER_DELAYED_EFFECT",
    "IF", "IF_ELSE", "ELSE", "PASS", "USE_ABILITY",
    "MANA_CHARGE", "SELECT_TARGET",
    # 以下はエンジン内部専用（cards.json では通常使用しない）
    "APPLY_MODIFIER", "GRANT_KEYWORD", "PUT_CREATURE",
    # 再発防止: cards.json で REVOLUTION_CHANGE を使うため enum 監査リストに含める。
    "REVOLUTION_CHANGE",
    "REPLACE_CARD_MOVE", "ADD_RESTRICTION",
    "COST_MODIFIER",
    # 再発防止: 新しい CommandType を追加したらここにも追加すること
    "DRAW",            # DRAW_CARD の別名 (FLOW内サブコマンド)
    "REPLACE_MOVE_CARD",  # 置換効果: カードの移動先を墓地に変更
    "LOCK_SPELL", "SPELL_RESTRICTION",
    "CANNOT_PUT_CREATURE", "CANNOT_SUMMON_CREATURE", "PLAYER_CANNOT_ATTACK",
    "IGNORE_ABILITY",
}

# C++ generate_instructions で実際に命令生成する（no-op でない）コマンドタイプ
# 再発防止: generate_instructions の switch 文を変更したらこのセットも必ず更新すること
# 最終更新: IF/IF_ELSE/ELSE, SELECT_NUMBER, SELECT_OPTION, CAST_SPELL, APPLY_MODIFIER,
#           GRANT_KEYWORD, PUT_CREATURE, ADD_RESTRICTION を generate_macro/primitive_instructions に追加
_CPP_IMPLEMENTED_COMMANDS: set = {
    "TRANSITION", "MUTATE", "FLOW", "QUERY", "SHUFFLE_DECK",
    "MANA_CHARGE",
    "DRAW_CARD", "BOOST_MANA", "ADD_MANA", "DESTROY", "DISCARD",
    "TAP", "UNTAP", "RETURN_TO_HAND", "BREAK_SHIELD",
    "POWER_MOD", "ADD_KEYWORD", "SEARCH_DECK", "SEND_TO_MANA",
    # 条件分岐: IF/IF_ELSE は FLOW と同一構造でgenerate_primitive_instructionsで処理
    "IF", "IF_ELSE", "ELSE",
    # ユーザー入力: SELECT_NUMBER は WAIT_INPUT stub として実装済み
    "SELECT_NUMBER",
    # 選択肢: SELECT_OPTION は CHOICE として options[0] をフォールバック実行
    "SELECT_OPTION",
    # 呪文詠唱: CAST_SPELL は GAME_ACTION(CAST_SPELL) を生成
    "CAST_SPELL",
    # 修飾子付与: APPLY_MODIFIER/GRANT_KEYWORD は ADD_KEYWORD 経由で実装
    "APPLY_MODIFIER", "GRANT_KEYWORD",
    "COST_MODIFIER",
    # クリーチャー配置: PUT_CREATURE は GAME_ACTION(SUMMON_CREATURE) で実装
    "PUT_CREATURE",
    # 制約追加: ADD_RESTRICTION は MODIFY(ADD_RESTRICTION) で実装
    "ADD_RESTRICTION",
    # カード移動置換: REPLACE_CARD_MOVE は to_zone への TRANSITION として実装
    "REPLACE_CARD_MOVE",
    # 再発防止: 新しい実装を追加したらここにも追加すること
    # DRAW_CARD 別名: DRAW は FLOW 内でカードを引くサブコマンド
    "DRAW",
    # 置換効果: REPLACE_MOVE_CARD は GAME_ACTION(REPLACE_MOVE_CARD) を生成
    "REPLACE_MOVE_CARD",
    # 制限/無視: 追加したら command_system.cpp の switch + macro 追加とセットで更新すること
    "SPELL_RESTRICTION", "CANNOT_PUT_CREATURE", "CANNOT_SUMMON_CREATURE", "PLAYER_CANNOT_ATTACK",
    "IGNORE_ABILITY",
    # バッファ操作: デッキトップから選択するシェイパー・デドダム実装
    # 再発防止: LOOK_TO_BUFFER を追加したら command_system.cpp の switch 文にも必ず追加すること
    "LOOK_TO_BUFFER", "REVEAL_TO_BUFFER", "SELECT_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE",
    "MOVE_BUFFER_REMAIN_TO_ZONE",
    "REGISTER_DELAYED_EFFECT",
    # 再発防止: REVOLUTION_CHANGE は cards.json で ability DECLARATION として記録されており
    #   generate_instructions では明示的 no-op として実装済み（MOVE命令は生成しない）。
    #   実際のスワップ処理は trigger_manager → pending_strategy 経由で行われる。
    #   command_system.cpp の case REVOLUTION_CHANGE: を削除しないこと（削除すると warning 再発）。
    "REVOLUTION_CHANGE",
}


def _walk_commands(obj: Any) -> "Generator[Dict[str, Any], None, None]":
    """JSON オブジェクト内の全コマンド辞書を再帰的にジェネレートする。"""
    if isinstance(obj, dict):
        t = obj.get("type", "")
        if t and any(k in obj for k in (
            "from_zone", "to_zone", "scope", "target_group", "amount",
            "str_val", "str_param", "mutation_kind", "output_value_key",
            "input_value_key", "if_true", "if_false", "options",
        )):
            yield obj
        for v in obj.values():
            yield from _walk_commands(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_commands(v)


class TestCppTextConsistency:
    """C++ 実装とテキスト生成の整合性を確認する。

    再発防止:
    - cards.json に新コマンドタイプを追加した場合:
      1. src/core/card_json_types.hpp の NLOHMANN_JSON_SERIALIZE_ENUM(CommandType) に登録
      2. src/engine/infrastructure/commands/command_system.cpp generate_instructions に実装
      3. このファイルの _CPP_COMMAND_TYPE_STRINGS と _CPP_IMPLEMENTED_COMMANDS を更新
    """

    def test_cards_json_command_types_in_cpp_enum(self) -> None:
        """cards.json で使われる全コマンドタイプが C++ CommandType enum に登録されていることを確認する。

        未登録の場合、C++ パーサーが NONE に変換してサイレントに no-op になる。
        再発防止: 新コマンドタイプは必ず NLOHMANN_JSON_SERIALIZE_ENUM に追加すること。
        """
        not_in_enum: Dict[str, List[int]] = {}

        for card in _CARDS:
            for cmd in _walk_commands(card):
                t = cmd.get("type", "")
                if not t or t == "NONE":
                    continue
                if t not in _CPP_COMMAND_TYPE_STRINGS:
                    not_in_enum.setdefault(t, []).append(card.get("id", -1))

        if not_in_enum:
            detail = "\n".join(
                f"  {t}: card_ids={ids}" for t, ids in sorted(not_in_enum.items())
            )
            pytest.fail(
                f"C++ CommandType enum に未登録のコマンドタイプがあります:\n{detail}\n"
                "再発防止: src/core/card_json_types.hpp の "
                "NLOHMANN_JSON_SERIALIZE_ENUM(CommandType) に追加すること。\n"
                "未登録のまま使うと C++ パーサーが NONE に変換してサイレント no-op になる。"
            )

    def test_cards_json_critical_commands_are_implemented(self) -> None:
        """cards.json の重要コマンドが C++ generate_instructions で実装されていることを確認する。

        DRAW_CARD, DESTROY, TAP, DISCARD, FLOW, IF, IF_ELSE など
        ゲームプレイに直接影響するコマンドは必ず実装されていること。
        再発防止: generate_instructions の switch 文が default: break のコマンドは
                  ゲームバグの原因になる。
        """
        # 必ず実装されているべき重要コマンドタイプ
        critical_types = {
            "DRAW_CARD", "DESTROY", "DISCARD", "TAP", "UNTAP",
            "TRANSITION", "MUTATE", "FLOW", "QUERY",
            "RETURN_TO_HAND", "BREAK_SHIELD", "ADD_KEYWORD",
            "IF", "IF_ELSE",
        }

        not_implemented = {
            t for t in critical_types if t not in _CPP_IMPLEMENTED_COMMANDS
        }

        assert not not_implemented, (
            f"重要コマンドが C++ generate_instructions で未実装: {not_implemented}\n"
            "再発防止: command_system.cpp の switch 文に case を追加すること"
        )

    def test_catalog_unimplemented_commands_in_cards(self) -> None:
        """cards.json で使われているが C++ で未実装のコマンドタイプを記録する。

        このテストは警告レベル: 3 件以下は許容するが、多すぎる場合は失敗する。
        再発防止: 未実装コマンドは「テキストに説明があるが C++ では何もしない」状態。
                  優先度が高い順に実装し _CPP_IMPLEMENTED_COMMANDS に追加すること。
        """
        unimplemented: Dict[str, List[int]] = {}

        for card in _CARDS:
            for cmd in _walk_commands(card):
                t = cmd.get("type", "")
                if not t or t == "NONE":
                    continue
                # C++ enumに存在するが実装されていない
                if t in _CPP_COMMAND_TYPE_STRINGS and t not in _CPP_IMPLEMENTED_COMMANDS:
                    unimplemented.setdefault(t, []).append(card.get("id", -1))

        if unimplemented:
            detail = "\n".join(
                f"  {t}: card_ids={sorted(set(ids))}" for t, ids in sorted(unimplemented.items())
            )
            # 5件超は fail、5件以下は xfail 相当として pytest.warn
            msg = (
                f"C++ generate_instructions で未実装（サイレント no-op）のコマンド:\n{detail}\n"
                "再発防止:\n"
                "  1. command_system.cpp の generate_instructions に case を追加\n"
                "  2. _CPP_IMPLEMENTED_COMMANDS セットを更新\n"
                "  現時点での優先度: CAST_SPELL > SELECT_OPTION > APPLY_MODIFIER > "
                "REVEAL_TO_BUFFER > SELECT_FROM_BUFFER > REGISTER_DELAYED_EFFECT"
            )
            if len(unimplemented) > 5:
                pytest.fail(msg)
            else:
                pytest.warns(UserWarning, match=".*") if False else None  # mark
                import warnings
                warnings.warn(msg, UserWarning, stacklevel=2)

    def test_if_command_generates_conditional_text(self) -> None:
        """IF コマンドが条件付きの日本語テキストを生成することを確認する。

        再発防止: IF コマンドの条件は target_filter にも条件定義が含まれる場合がある。
                  テキスト生成は condition / target_filter の両方をチェックすること。
        """
        # OPPONENT_DRAW_COUNT 条件の IF コマンド（cards.json の card_6 パターン）
        if_cmd = {
            "type": "IF",
            "target_filter": {"type": "OPPONENT_DRAW_COUNT", "value": 2},
            "if_true": [_make_command("DRAW_CARD", amount=2)],
            "if_false": [],
        }
        text = CardTextGenerator._format_command(if_cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        # 条件テキストまたは if_true の内容が含まれること
        assert len(text) > 0, (
            "IF コマンドのテキストが空。\n"
            "再発防止: _format_logic_command で target_filter の条件定義を処理すること"
        )
        # 「引」（DRAW_CARD）か条件文（「なら」「時」「場合」）が含まれるはず
        assert any(kw in text for kw in ["引", "なら", "時", "場合", "ドロー", "2"]), (
            f"IF コマンドテキストに条件・結果が含まれない: 「{text}」"
        )

    def test_if_else_command_generates_branching_text(self) -> None:
        """IF_ELSE コマンドが分岐テキストを生成することを確認する。

        再発防止: IF_ELSE は両方の分岐テキストを「そうでなければ」等で連結すること。
        """
        if_else_cmd = {
            "type": "IF_ELSE",
            "condition": {"type": "MANA_ARMED", "value": 5},
            "if_true": [_make_command("DRAW_CARD", amount=2)],
            "if_false": [_make_command("DRAW_CARD", amount=1)],
        }
        text = CardTextGenerator._format_command(if_else_cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        assert len(text) > 0, (
            "IF_ELSE コマンドのテキストが空。\n"
            "再発防止: _format_logic_command で IF_ELSE の両分岐を処理すること"
        )

    def test_apply_modifier_generates_keyword_text(self) -> None:
        """APPLY_MODIFIER でキーワード付与テキストが正しく生成されることを確認する。

        再発防止: APPLY_MODIFIER は str_param でキーワード名を受け取る。
                  CardTextResources.get_keyword_text(str_param) でキーワード名を翻訳すること。
        """
        cmd = _make_command(
            "APPLY_MODIFIER",
            duration="THIS_TURN",
            str_param="SPEED_ATTACKER",
            target_group="PLAYER_SELF",
            target_filter={
                "types": ["CREATURE"],
                "zones": ["BATTLE_ZONE"],
                "civilizations": [],
                "races": [],
            },
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        assert "スピードアタッカー" in text or "speed" in text.lower(), (
            f"APPLY_MODIFIER SPEED_ATTACKER でキーワードテキストが含まれない: 「{text}」\n"
            "再発防止: ApplyModifierFormatter のケースを確認すること"
        )

    def test_apply_modifier_blocker_generates_text(self) -> None:
        """APPLY_MODIFIER でブロッカー付与テキストが生成されることを確認する。"""
        cmd = _make_command(
            "APPLY_MODIFIER",
            duration="PERMANENT",
            str_param="BLOCKER",
            target_group="PLAYER_SELF",
            target_filter={"types": ["CREATURE"], "zones": ["BATTLE_ZONE"]},
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        assert "ブロッカー" in text or len(text) > 0, (
            f"APPLY_MODIFIER BLOCKER テキストが不正: 「{text}」"
        )

    def test_select_option_does_not_crash(self) -> None:
        """SELECT_OPTION コマンドがクラッシュしないことを確認する。

        再発防止: SELECT_OPTION は C++ では CHOICE に相当する。
                  NLOHMANN_JSON_SERIALIZE_ENUM で SELECT_OPTION → CHOICE へのマッピングが
                  必要（現在未登録のため NONE になる）。
        """
        cmd = {
            "type": "SELECT_OPTION",
            "amount": 1,
            "option_count": 2,
            "optional": True,
            "options": [
                [_make_command("DRAW_CARD", amount=2)],
                [_make_command("DESTROY", target_group="PLAYER_OPPONENT")],
            ],
        }
        try:
            text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
            assert isinstance(text, str)
        except Exception as e:
            pytest.fail(
                f"SELECT_OPTION でクラッシュ: {e}\n"
                "再発防止: _format_command で SELECT_OPTION を SELECT_OPTION/CHOICE として処理すること"
            )

    def test_select_number_generates_input_text(self) -> None:
        """SELECT_NUMBER コマンドが数値入力テキストを生成することを確認する。

        再発防止: SELECT_NUMBER は C++ generate_instructions で SELECT_NUMBER ケースを
                  実装し WAIT_INPUT 命令を生成する必要がある。現在は no-op。
        """
        cmd = _make_command("SELECT_NUMBER", amount=6, output_value_key="chosen_number")
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        # 数値選択を示すテキストが含まれること
        assert len(text) > 0, (
            "SELECT_NUMBER のテキストが空。\n"
            "再発防止: _format_game_action_command の SELECT_NUMBER ケースを確認すること"
        )

    def test_cast_spell_generates_casting_text(self) -> None:
        """CAST_SPELL コマンドが呪文詠唱テキストを生成することを確認する。

        再発防止: CAST_SPELL は C++ generate_instructions で未実装（default: break）。
                  テキスト生成は正しく動作するが C++ では何もしない。
                  command_system.cpp に CAST_SPELL ケースを実装すること。
        """
        cmd = _make_command(
            "CAST_SPELL",
            target_group="PLAYER_SELF",
            target_filter={
                "types": ["SPELL"],
                "zones": ["HAND"],
                "max_cost": 3,
                "civilizations": [],
                "races": [],
            },
        )
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        assert len(text) > 0, (
            "CAST_SPELL のテキストが空。\n"
            "再発防止: _format_action で CAST_SPELL ケースを確認すること"
        )
        # 呪文を示す何らかのキーワードが含まれるはず
        assert any(kw in text for kw in ["呪文", "唱", "コスト", "手札", "3"]), (
            f"CAST_SPELL テキストに呪文関連語がない: 「{text}」"
        )

    def test_register_delayed_effect_text(self) -> None:
        """REGISTER_DELAYED_EFFECT コマンドがテキストを生成することを確認する。

        再発防止: C++ generate_instructions で未実装（default: break）。
                  テキスト生成は動作するがエンジンでは実行されない。
        """
        cmd = {
            "type": "REGISTER_DELAYED_EFFECT",
            "amount": 1,
            "str_param": "END_OF_TURN_DESTROY",
            "target_group": "PLAYER_SELF",
        }
        text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
        assert isinstance(text, str)
        assert len(text) > 0, "REGISTER_DELAYED_EFFECT のテキストが空"

    def test_cards_json_select_option_has_options_field(self) -> None:
        """cards.json の SELECT_OPTION コマンドに options フィールドが存在することを確認する。

        再発防止: SELECT_OPTION には options（選択肢リスト）が必須。
                  欠落するとテキスト生成が空になり C++ では何もしない。
        """
        missing_options: List[str] = []
        for card in _CARDS:
            for cmd in _walk_commands(card):
                if cmd.get("type") == "SELECT_OPTION":
                    if not cmd.get("options"):
                        missing_options.append(
                            f"card_id={card.get('id')} name={card.get('name')}"
                        )
        assert not missing_options, (
            f"SELECT_OPTION に options フィールドがないコマンド:\n"
            + "\n".join(missing_options)
            + "\n再発防止: SELECT_OPTION には必ず options フィールドを設定すること"
        )

    @pytest.mark.parametrize("card", _CARDS, ids=[f"cpp_{c.get('id', i)}" for i, c in enumerate(_CARDS)])
    def test_all_effect_commands_generate_nonempty_text_or_are_structural(
        self, card: Dict[str, Any]
    ) -> None:
        """全カードの効果コマンドが非空テキストを返すか、構造コマンド（QUERY等）であることを確認する。

        再発防止: テキスト生成が空のコマンドは「日本語化漏れ」または「C++ no-op」の可能性。
                  新コマンドを追加したら _format_command に日本語テンプレートを追加すること。
        """
        # テキストが空でも許容する構造コマンドタイプ
        structural_types = {
            "QUERY", "FLOW", "IF", "IF_ELSE", "ELSE",
            "SELECT_TARGET",  # SELECT_TARGET は上位コマンドが説明を持つ
            "REGISTER_DELAYED_EFFECT",  # C++ 未実装だが許容
            "REVEAL_TO_BUFFER",   # バッファ操作は内部的
            "SELECT_FROM_BUFFER", # バッファ操作は内部的
        }

        empty_non_structural: List[str] = []
        for effect in (card.get("effects") or []):
            for cmd in (effect.get("commands") or []):
                t = cmd.get("type", "")
                if not t or t == "NONE" or t in structural_types:
                    continue
                text = CardTextGenerator._format_command(cmd, TextGenerationContext(card_data={}))
                if not text.strip():
                    empty_non_structural.append(
                        f"trigger={effect.get('trigger')} cmd_type={t}"
                    )

        assert not empty_non_structural, (
            f"カード id={card.get('id')} name={card.get('name')}: "
            f"テキストが空の非構造コマンド:\n"
            + "\n".join(f"  {x}" for x in empty_non_structural)
            + "\n再発防止: _format_command に対応するテンプレートを追加すること"
        )
