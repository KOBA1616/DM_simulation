# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_generate_body_text_skips_effect_line_for_revolution_change_command_type():
    card = {
        "type": "CREATURE",
        "keywords": {"revolution_change": True},
        "effects": [
            {
                "trigger": "ON_ATTACK",
                "commands": [
                    {
                        "type": "REVOLUTION_CHANGE",
                        "target_filter": {"races": ["Dragon"]},
                    }
                ],
            }
        ],
    }

    body = CardTextGenerator.generate_body_text(card)

    # Regression guard: special keyword effects should not be duplicated as normal effect lines.
    assert "■ 革命チェンジ：Dragon" in body
    assert "このクリーチャーが攻撃する時: (革命チェンジ)" not in body


def test_get_rc_filter_ignores_legacy_mutate_revolution_change():
    card = {
        "effects": [
            {
                "commands": [
                    {
                        "type": "MUTATE",
                        "mutation_kind": "REVOLUTION_CHANGE",
                        "target_filter": {"races": ["Dragon"]},
                    }
                ]
            }
        ]
    }

    # 最新仕様移管: MUTATE互換は解釈しない
    from dm_toolkit.gui.editor.formatters.special_keywords import RevolutionChangeFormatter
    tf = RevolutionChangeFormatter.get_rc_filter_from_effects(card)
    assert tf == {}


def test_generate_body_text_renders_rc_condition_even_without_keyword_flag():
    card = {
        "type": "CREATURE",
        "keywords": {},
        "effects": [
            {
                "trigger": "ON_ATTACK",
                "trigger_scope": "PLAYER_SELF",
                "commands": [
                    {
                        "type": "REVOLUTION_CHANGE",
                        "target_filter": {
                            "civilizations": ["MULTICOLOR"],
                            "races": ["マジック"],
                            "max_cost": 5,
                        },
                    }
                ],
            }
        ],
    }

    body = CardTextGenerator.generate_body_text(card)

    assert "■ 革命チェンジ：多色のコスト5以下のマジック" in body


def test_generate_body_text_uses_only_command_filter_for_revolution_change():
    card = {
        "type": "CREATURE",
        "keywords": {"revolution_change": True},
        "effects": [
            {
                "trigger": "ON_ATTACK",
                "commands": [
                    {
                        "type": "REVOLUTION_CHANGE",
                        "target_filter": {"races": ["NewRace"]},
                    }
                ],
            }
        ],
    }

    body = CardTextGenerator.generate_body_text(card)

    assert "■ 革命チェンジ：NewRace" in body


def test_get_rc_filter_uses_target_filter_only():
    card = {
        "effects": [
            {
                "commands": [
                    {
                        "type": "REVOLUTION_CHANGE",
                        "target_filter": {"races": ["AliasRace"]},
                    }
                ]
            }
        ]
    }

    from dm_toolkit.gui.editor.formatters.special_keywords import RevolutionChangeFormatter
    tf = RevolutionChangeFormatter.get_rc_filter_from_effects(card)
    assert tf == {"races": ["AliasRace"]}
