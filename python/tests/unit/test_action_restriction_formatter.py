# -*- coding: utf-8 -*-
"""Action restriction formatter tests."""

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_cannot_put_creature_includes_filter_description() -> None:
    cmd = {
        "type": "CANNOT_PUT_CREATURE",
        "target_group": "PLAYER_OPPONENT",
        "duration": "THIS_TURN",
        "target_filter": {
            "civilizations": ["FIRE"],
            "max_cost": 5,
        },
    }

    text = CardTextGenerator.format_command(cmd)

    assert "出せない" in text
    assert "クリーチャー" in text
    assert "コスト5以下" in text
    assert "火" in text


def test_cannot_summon_creature_supports_linked_power_filter() -> None:
    cmd = {
        "type": "CANNOT_SUMMON_CREATURE",
        "target_group": "PLAYER_OPPONENT",
        "duration": "THIS_TURN",
        "target_filter": {
            "max_power": {
                "input_link": "selected_power",
                "input_value_usage": "MAX_POWER",
            },
        },
    }

    text = CardTextGenerator.format_command(cmd)

    assert "召喚できない" in text
    assert "クリーチャー" in text
    assert "パワーその数以下" in text


def test_limit_put_creature_per_turn_includes_filter_description() -> None:
    cmd = {
        "type": "LIMIT_PUT_CREATURE_PER_TURN",
        "target_group": "PLAYER_OPPONENT",
        "duration": "THIS_TURN",
        "amount": 2,
        "target_filter": {
            "min_cost": 3,
            "max_cost": 6,
        },
    }

    text = CardTextGenerator.format_command(cmd)

    assert "各ターン" in text
    assert "2体まで" in text
    assert "コスト3～6" in text
    assert "クリーチャー" in text


def test_cannot_put_creature_respects_count_limit_filter() -> None:
    cmd = {
        "type": "CANNOT_PUT_CREATURE",
        "target_group": "PLAYER_OPPONENT",
        "duration": "UNTIL_START_OF_OPPONENT_TURN",
        "target_filter": {
            "max_count": 1,
        },
    }

    text = CardTextGenerator.format_command(cmd)

    assert "次の相手のターンのはじめまで" in text
    assert "1体までしか出せない" in text


def test_cannot_summon_creature_respects_exact_count_filter() -> None:
    cmd = {
        "type": "CANNOT_SUMMON_CREATURE",
        "target_group": "PLAYER_OPPONENT",
        "duration": "THIS_TURN",
        "target_filter": {
            "exact_count": 2,
        },
    }

    text = CardTextGenerator.format_command(cmd)

    assert "このターン" in text or "次の相手のターンの終わりまで" in text
    assert "2体までしか召喚できない" in text
