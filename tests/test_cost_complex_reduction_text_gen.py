import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def test_complex_reduction_with_compare_stat():
    data = {
        "cost": 9,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {
                    "type": "COMPARE_STAT",
                    "stat_key": "TOTAL_POWER",
                    "value": 3000,
                    "op": ">="
                },
                "value_mode": "FIXED",
                "value": 2
            }
        ]
    }
    text = CardTextGenerator.generate_body_text(data)
    assert "自分のバトルゾーンの合計パワーが3000以上なら、このカードの召喚コストは2少なくなる。" in text

def test_complex_reduction_with_cards_matching_filter():
    data = {
        "cost": 9,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {
                    "type": "CARDS_MATCHING_FILTER",
                    "target_filter": {
                        "civilizations": ["DARKNESS"]
                    },
                    "count": 3,
                    "op": ">="
                },
                "value_mode": "FIXED",
                "value": 4
            }
        ]
    }
    text = CardTextGenerator.generate_body_text(data)
    assert "闇の文明が3枚以上あるなら、このカードの召喚コストは4少なくなる。" in text

def test_complex_reduction_with_stat_scaled():
    data = {
        "cost": 8,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {
                    "type": "CARDS_MATCHING_FILTER",
                    "target_filter": {
                        "civilizations": ["LIGHT"]
                    },
                    "count": 1,
                    "op": ">="
                },
                "value_mode": "STAT_SCALED",
                "stat_key": "TOTAL_POWER",
                "per_value": 1,
                "increment_cost": 1,
                "min_stat": 1,
                "max_reduction": 4
            }
        ]
    }
    text = CardTextGenerator.generate_body_text(data)
    assert "光の文明が1枚以上あるなら、このカードの召喚コストを、自分のバトルゾーンの合計パワーの値に応じて1ごとに1軽減する" in text
    assert "（最大4軽減）" in text
