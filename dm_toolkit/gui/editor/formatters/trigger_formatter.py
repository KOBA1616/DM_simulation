# -*- coding: utf-8 -*-
from typing import Dict, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import TimingMode

class TriggerFormatter:
    @classmethod
    def resolve_effect_timing_mode(cls, effect: Dict[str, Any]) -> str:
        """Normalize effect timing mode for text composition."""
        if not isinstance(effect, dict):
            return TimingMode.POST.value
        mode = str(effect.get("timing_mode", "") or "").upper()
        if mode in (TimingMode.PRE.value, TimingMode.POST.value):
            return mode
        return TimingMode.PRE.value if cls.is_replacement_effect(effect) else TimingMode.POST.value

    @classmethod
    def to_replacement_trigger_text(cls, trigger_text: str) -> str:
        """Convert post-event trigger text (〜た時) into replacement tone (〜る時)."""
        text = trigger_text
        for src, dst in CardTextResources.TRIGGER_REPLACEMENT_MAP:
            if src in text:
                return text.replace(src, dst)
        return text

    @classmethod
    def is_replacement_effect(cls, effect: Dict[str, Any]) -> bool:
        """Return True if the effect should be rendered as PRE/replacement timing."""
        if not isinstance(effect, dict):
            return False
        return effect.get("mode") == "REPLACEMENT" or effect.get("timing_mode") == TimingMode.PRE.value

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False, effect: Dict[str, Any] = None) -> str:
        """Get Japanese trigger text, applying replacement phrasing when needed."""
        base = CardTextResources.get_trigger_text(trigger, is_spell=is_spell)
        if effect is not None and cls.is_replacement_effect(effect):
            return cls.to_replacement_trigger_text(base)

        return base

class ReplacementEffectFormatter:
    """Formatter to strictly decouple and structure Replacement Effects."""

    @classmethod
    def format(cls, trigger_text: str, actions_text: str) -> str:
        if not trigger_text:
            return actions_text
        if trigger_text.endswith("、"):
            return f"{trigger_text}かわりに{actions_text}"
        return f"{trigger_text}、かわりに{actions_text}"
