from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_static_cost_modifier_stat_scaled_uses_stat_key_label() -> None:
    data = {
        "name": "StaticCostCard",
        "type": "CREATURE",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CARDS_DRAWN_THIS_TURN",
                "per_value": 1,
                "min_stat": 0,
                "max_reduction": 3,
                "value": 1,
                "scope": "SELF", "target_filter": {},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに引いたカード" in text
    assert "1枚につき1軽減" in text
    assert "最大3軽減" in text


def test_static_cost_modifier_stat_scaled_threshold_phrase_is_explicit() -> None:
    data = {
        "name": "StaticCostCard",
        "type": "CREATURE",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CARDS_DRAWN_THIS_TURN",
                "per_value": 2,
                "min_stat": 1,
                "value": 1,
                "scope": "SELF", "target_filter": {},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに引いたカードが1枚以上で適用" in text
    assert "2枚につき1軽減" in text


def test_static_cost_modifier_stat_scaled_zero_value_does_not_fallback_to_generic() -> None:
    """UI既定値(value=0)でも『修正する』に落とさず、統計キー文面を維持する。"""
    data = {
        "name": "StaticCostCard",
        "type": "CREATURE",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CARDS_DRAWN_THIS_TURN",
                "per_value": 1,
                "value": 0,
                "scope": "SELF", "target_filter": {},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに引いたカード" in text
    assert "1枚につき1軽減" in text
    assert "修正する" not in text


def test_static_cost_modifier_legacy_without_value_mode_is_inferred_stat_scaled() -> None:
    """旧データ互換: value_mode欠落でも stat_key/per_value があれば STAT_SCALED として表示する。"""
    data = {
        "name": "LegacyStatCard",
        "type": "CREATURE",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "stat_key": "CARDS_DRAWN_THIS_TURN",
                "per_value": 1,
                "min_stat": 0,
                "value": 0,
                "scope": "ALL", "target_filter": {},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "このターンに引いたカード" in text
    assert "1枚につき1軽減" in text
    assert "修正する" not in text


def test_static_cost_modifier_fixed_keeps_existing_style() -> None:
    data = {
        "name": "StaticCostCard",
        "type": "CREATURE",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "FIXED",
                "value": 2,
                "scope": "SELF", "target_filter": {},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "コストを2軽減" in text
