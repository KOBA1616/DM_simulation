from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_cost_reduction_passive_fixed():
    """PASSIVE な固定軽減を日本語テキストに変換することを検証"""
    data = {
        "name": "TestCard",
        "cost": 5,
        "cost_reductions": [
            {
                "type": "PASSIVE",
                "condition": {
                    "type": "CIVILIZATION",
                    "civilization": "DARKNESS",
                    "count": 2,
                },
                "value_mode": "FIXED",
                "value": 2,
                "unit_cost": {"filter": {"civilizations": ["DARKNESS"], "min_cost": 0}},
            }
        ],
    }

    text = CardTextGenerator.generate_body_text(data)
    assert "召喚コスト" in text
    assert "2少なくなる" in text or "2削減" in text
    # Civilization name should be mentioned (localized or raw)
    assert "闇" in text or "DARKNESS" in text
