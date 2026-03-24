# -*- coding: utf-8 -*-
"""Regression tests for replacement trigger text/template handling."""

from dm_toolkit.gui.editor.models import EffectModel
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_effect_model_keeps_replacement_fields() -> None:
    """EffectModel must preserve replacement metadata used by preview reconstruction."""
    effect = EffectModel(
        trigger="ON_PLAY",
        mode="REPLACEMENT",
        timing_mode="PRE",
        trigger_scope="PLAYER_OPPONENT",
        trigger_filter={"types": ["CREATURE"]},
        commands=[],
    )

    dumped = effect.model_dump(exclude_none=True)
    assert dumped.get("mode") == "REPLACEMENT"
    assert dumped.get("timing_mode") == "PRE"
    assert dumped.get("trigger_scope") == "PLAYER_OPPONENT"
    assert dumped.get("trigger_filter") == {"types": ["CREATURE"]}


def test_trigger_scope_template_uses_pre_timing_for_replacement() -> None:
    """Template composition must switch to PRE timing phrasing when timing_mode=PRE."""
    text = CardTextGenerator._apply_trigger_scope(
        trigger_text="呪文を唱える時",
        scope="PLAYER_OPPONENT",
        trigger_type="ON_CAST_SPELL",
        trigger_filter={"types": ["SPELL"]},
        timing_mode="PRE",
    )

    assert "相手の呪文を唱える時" in text
    assert "相手の呪文を唱えた時" not in text
