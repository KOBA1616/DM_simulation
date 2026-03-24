# -*- coding: utf-8 -*-
"""
スキーマ ↔ モデル整合性テスト
再発防止: スキーマキー（str_param/amount 等）とモデルフィールド名の相互マッピングを検証する。
各モデルが model_validator により schema 由来の dict を受け入れられることを確認する。
"""
import pytest
from dm_toolkit.gui.editor.models import (
    QueryParams, AddKeywordParams, ApplyModifierParams,
    MekraidParams, PlayFromZoneParams, IgnoreAbilityParams,
    CommandModel,
)


class TestQueryParamsSchemaMapping:
    """QUERY: スキーマキー str_param ↔ モデルフィールド query_string の相互マッピング"""

    def test_schema_key_str_param_is_accepted(self):
        """スキーマが生成する str_param キーで model_validate が成功する"""
        params = QueryParams.model_validate({"str_param": "SELECT_OPTION", "amount": 1})
        assert params.query_string == "SELECT_OPTION"
        assert params.str_param == "SELECT_OPTION"

    def test_legacy_key_query_string_is_still_accepted(self):
        """レガシーキー query_string で構築した場合も str_param に同期される"""
        params = QueryParams(query_string="CARDS_MATCHING_FILTER", options=["a"])
        assert params.query_string == "CARDS_MATCHING_FILTER"
        assert params.str_param == "CARDS_MATCHING_FILTER"

    def test_command_model_ingest_with_schema_keys(self):
        """CommandModel がスキーマキーから QUERY を正しく ingest できる"""
        cmd = CommandModel.model_validate({
            "type": "QUERY",
            "str_param": "SELECT_OPTION",
            "amount": 2,
            "target_group": "OPPONENT",
        })
        assert isinstance(cmd.params, QueryParams)
        assert cmd.params.query_string == "SELECT_OPTION"


class TestAddKeywordParamsSchemaMapping:
    """ADD_KEYWORD: スキーマキー str_val ↔ モデルフィールド keyword の相互マッピング"""

    def test_schema_key_str_val_is_accepted(self):
        params = AddKeywordParams.model_validate({"str_val": "BLOCKER", "duration": "THIS_TURN"})
        assert params.keyword == "BLOCKER"
        assert params.str_val == "BLOCKER"
        assert params.duration == "THIS_TURN"

    def test_legacy_key_keyword_is_still_accepted(self):
        params = AddKeywordParams(keyword="SPEED_ATTACKER")
        assert params.keyword == "SPEED_ATTACKER"
        assert params.str_val == "SPEED_ATTACKER"

    def test_duration_accepts_string_not_int(self):
        """duration は SELECT フィールド（文字列）を受け入れる"""
        params = AddKeywordParams.model_validate({"str_val": "BLOCKER", "duration": "PERMANENT"})
        assert params.duration == "PERMANENT"

    def test_command_model_ingest_add_keyword(self):
        cmd = CommandModel.model_validate({
            "type": "ADD_KEYWORD",
            "str_val": "DOUBLE_BREAKER",
            "explicit_self": True,
            "duration": "THIS_TURN",
        })
        assert isinstance(cmd.params, AddKeywordParams)
        assert cmd.params.keyword == "DOUBLE_BREAKER"


class TestApplyModifierParamsSchemaMapping:
    """APPLY_MODIFIER: スキーマキー str_param/amount ↔ モデルフィールド modifier_type/value"""

    def test_schema_keys_are_accepted(self):
        params = ApplyModifierParams.model_validate({
            "str_param": "CANNOT_ATTACK",
            "amount": 1,
            "duration": "THIS_TURN",
        })
        assert params.modifier_type == "CANNOT_ATTACK"
        assert params.str_param == "CANNOT_ATTACK"
        assert params.value == 1
        assert params.amount == 1
        assert params.duration == "THIS_TURN"

    def test_legacy_keys_are_still_accepted(self):
        params = ApplyModifierParams.model_validate({"modifier_type": "COST_MOD", "value": 2})
        assert params.modifier_type == "COST_MOD"
        assert params.str_param == "COST_MOD"
        assert params.value == 2
        assert params.amount == 2

    def test_command_model_ingest_apply_modifier(self):
        cmd = CommandModel.model_validate({
            "type": "APPLY_MODIFIER",
            "str_param": "CANNOT_BLOCK",
            "amount": 0,
            "duration": "PERMANENT",
            "target_group": "OPPONENT",
        })
        assert isinstance(cmd.params, ApplyModifierParams)
        assert cmd.params.modifier_type == "CANNOT_BLOCK"


class TestMekraidParamsSchemaMapping:
    """MEKRAID: スキーマキー amount/val2/select_count ↔ モデルフィールド evolution_cost/reveal_count"""

    def test_schema_keys_are_accepted(self):
        params = MekraidParams.model_validate({"amount": 7, "val2": 3, "select_count": 1})
        assert params.evolution_cost == 7
        assert params.reveal_count == 3
        assert params.select_count == 1

    def test_legacy_keys_are_still_accepted(self):
        params = MekraidParams.model_validate({"reveal_count": 4, "evolution_cost": 5})
        assert params.reveal_count == 4
        assert params.val2 == 4
        assert params.evolution_cost == 5
        assert params.amount == 5

    def test_command_model_ingest_mekraid(self):
        cmd = CommandModel.model_validate({
            "type": "MEKRAID",
            "amount": 6,
            "val2": 3,
            "select_count": 1,
        })
        assert isinstance(cmd.params, MekraidParams)
        assert cmd.params.evolution_cost == 6
        assert cmd.params.reveal_count == 3


class TestPlayFromZoneParamsSchemaMapping:
    """PLAY_FROM_ZONE: スキーマキー from_zone/to_zone/target_filter が正しくマッピングされる"""

    def test_schema_zone_keys_are_accepted(self):
        params = PlayFromZoneParams.model_validate({
            "from_zone": "HAND",
            "to_zone": "BATTLE_ZONE",
            "amount": 5,
        })
        assert params.source_zone == "HAND"
        assert params.destination_zone == "BATTLE_ZONE"

    def test_target_filter_is_accepted(self):
        """target_filter キーが filter フィールドにマッピングされる"""
        tf = {"types": ["SPELL"], "max_cost": 4}
        params = PlayFromZoneParams.model_validate({"from_zone": "HAND", "target_filter": tf})
        assert params.filter == tf

    def test_command_model_ingest_play_from_zone(self):
        cmd = CommandModel.model_validate({
            "type": "PLAY_FROM_ZONE",
            "from_zone": "HAND",
            "to_zone": "BATTLE_ZONE",
            "amount": 7,
            "target_filter": {"types": ["CREATURE"]},
            "play_flags": True,
        })
        assert isinstance(cmd.params, PlayFromZoneParams)
        assert cmd.params.source_zone == "HAND"
        assert cmd.params.filter == {"types": ["CREATURE"]}


class TestIgnoreAbilityParamsDurationType:
    """IGNORE_ABILITY: duration は文字列型（SELECT フィールド）を受け入れる"""

    def test_duration_accepts_string(self):
        params = IgnoreAbilityParams.model_validate({
            "duration": "THIS_TURN",
            "target_group": "OPPONENT",
        })
        assert params.duration == "THIS_TURN"

    def test_duration_none_is_allowed(self):
        params = IgnoreAbilityParams.model_validate({})
        assert params.duration is None


class TestAlreadyTypedParamsNotOverwritten:
    """再発防止: params に型モデルが既に入っている場合、ingest_legacy_structure が上書きしない"""

    def test_typed_params_preserved_through_commandmodel(self):
        qp = QueryParams(query_string="find", options=["a", "b"])
        cmd = CommandModel(type="QUERY", params=qp)
        out = cmd.model_dump()
        # query_string が None に上書きされていないことを確認
        assert out.get("query_string") == "find"
        assert out.get("options") == ["a", "b"]
