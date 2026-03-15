# -*- coding: utf-8 -*-
from __future__ import annotations

from dm_toolkit.gui.editor.models import CardModel
from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService


def test_inject_keyword_logic_accepts_keywords_model_instance() -> None:
    card_data = {
        "keywords": CardModel.KeywordsModel(),
        "effects": [
            {"commands": [{"type": "REVOLUTION_CHANGE"}]},
            {
                "commands": [
                    {
                        "type": "FRIEND_BURST",
                        "target_filter": {"races": ["マジック"]},
                    }
                ]
            },
        ],
    }

    EditorFeatureService.inject_keyword_logic(card_data)

    assert isinstance(card_data["keywords"], dict)
    assert card_data["keywords"].get("revolution_change") is True
    assert card_data["keywords"].get("friend_burst") is True
    assert card_data.get("friend_burst_condition") == {"races": ["マジック"]}
