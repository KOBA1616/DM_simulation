import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def test_compare_stat_condition():
    condition = {
        "type": "COMPARE_STAT",
        "stat_key": "TOTAL_POWER",
        "op": ">=",
        "value": 3000
    }
    text = CardTextGenerator._format_condition(condition)
    assert "自分のバトルゾーンの合計パワーが3000以上なら: " in text

def test_compare_stat_condition_less_than():
    condition = {
        "type": "COMPARE_STAT",
        "stat_key": "MY_SHIELD_COUNT",
        "op": "<=",
        "value": 2
    }
    text = CardTextGenerator._format_condition(condition)
    assert "自分のシールドが2つ以下なら: " in text

def test_cards_matching_filter_condition():
    condition = {
        "type": "CARDS_MATCHING_FILTER",
        "filter": {
            "civilizations": ["DARKNESS"]
        },
        "count": 3,
        "op": ">="
    }
    text = CardTextGenerator._format_condition(condition)
    assert "闇の文明が3枚以上あるなら: " in text

def test_cards_matching_filter_condition_complex():
    condition = {
        "type": "CARDS_MATCHING_FILTER",
        "filter": {
            "races": ["Demon Command"],
            "zone": ["BATTLE_ZONE"]
        },
        "count": 1,
        "op": ">="
    }
    text = CardTextGenerator._format_condition(condition)
    assert "バトルゾーンにDemon Commandが1体以上いるなら: " in text
