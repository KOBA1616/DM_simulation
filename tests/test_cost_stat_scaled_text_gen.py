from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_cost_reduction_stat_scaled():
    """STAT_SCALED 軽減の計算式テキスト化を検証"""
    data = {
        "name": "StatCard",
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "value_mode": "STAT_SCALED",
                "stat_key": "TOTAL_POWER",
                "per_value": 3,
                "increment_cost": 1,
                "min_stat": 1,
                "max_reduction": 4,
                "unit_cost": {"target_filter": {"civilizations": ["DARKNESS"]}},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    # Expect wording that describes scaling and mention of max reduction
    assert ("大きいほど" in text) or ("ごと" in text)
    assert ("最大4" in text) or ("最大 4" in text) or ("最大4削減" in text)


def test_cost_reduction_stat_scaled_uses_stat_key_label() -> None:
    """STAT_SCALED は stat_key をそのまま表示せず日本語ラベルで表示する。"""
    data = {
        "name": "StatCard",
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "value_mode": "STAT_SCALED",
                "stat_key": "MANA_CIVILIZATION_COUNT",
                "per_value": 1,
                "increment_cost": 1,
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "マナゾーンの文明数" in text
    assert "MANA_CIVILIZATION_COUNT" not in text


def test_cost_reduction_stat_scaled_supports_alias_key_label() -> None:
    """別名 stat_key でも正規化済みラベルで表示する。"""
    data = {
        "name": "StatCard",
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "value_mode": "STAT_SCALED",
                "stat_key": "SPELL_CAST_COUNT_THIS_TURN",
                "per_value": 1,
                "increment_cost": 1,
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに唱えた呪文" in text
    assert "SPELL_CAST_COUNT_THIS_TURN" not in text


def test_cost_reduction_stat_scaled_uses_unified_phrase() -> None:
    """cost_reductions 側も COST_MODIFIER 側と同じ文体で出力されること。"""
    data = {
        "name": "StatCard",
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "value_mode": "STAT_SCALED",
                "stat_key": "CARDS_DRAWN_THIS_TURN",
                "per_value": 1,
                "increment_cost": 1,
                "min_stat": 1,
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに引いたカードの値に応じて1枚につき1軽減する" in text
    assert "このターンに引いたカードが1枚以上で適用" in text
    assert "大きいほど" not in text
