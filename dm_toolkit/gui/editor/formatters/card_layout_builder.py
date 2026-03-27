# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import CardType
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class CardLayoutBuilder:
    """
    Builds the top-level layout of card text, combining headers,
    creature bodies, and spell sides (Twinpact) cleanly.
    """

    @classmethod
    def build_text(cls, data: Dict[str, Any], include_twinpact: bool = True, sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        """
        if not data:
            return ""

        if ctx is None:
            ctx = TextGenerationContext(data, sample)

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        lines.extend(cls.build_header_lines(data))

        # 2. Body (Keywords, Effects, etc.)
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        body_text = CardTextGenerator.generate_body_text(data, ctx=ctx)
        if body_text:
            lines.append(body_text)

        # 3. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            spell_ctx = TextGenerationContext(spell_side, ctx.sample)
            lines.append(cls.build_text(spell_side, include_twinpact=False, ctx=spell_ctx))

        return "\n".join(lines)

    @classmethod
    def build_header_lines(cls, data: Dict[str, Any]) -> List[str]:
        lines = []
        name = data.get("name") or tr("Unknown")
        cost = data.get("cost", 0)

        civs_data = data.get("civilizations", [])
        if not civs_data and "civilization" in data:
            civ_single = data.get("civilization")
            if civ_single:
                civs_data = [civ_single]
        civs = cls._format_civs(civs_data)

        raw_type = data.get("type", CardType.CREATURE.value)
        type_str = CardTextResources.get_card_type_text(raw_type)
        races = " / ".join(data.get("races", []))

        header = f"【{name}】 {civs} コスト{cost}"
        if races:
            header += f" {races}"
        lines.append(header)
        lines.append(f"[{type_str}]")

        power = data.get("power", 0)
        if power > 0:
             lines.append(f"パワー {power}")

        lines.append("-" * 20)
        return lines

    @classmethod
    def build_body_text_lines(cls, data: Dict[str, Any], include_twinpact: bool = True, sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        """
        Generates just the body text (keywords, effects, etc.) without the header.
        """
        lines = []
        if ctx is None:
            ctx = TextGenerationContext(data, sample)

        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        body_text = CardTextGenerator.generate_body_text(data, ctx=ctx)
        if body_text:
            lines.append(body_text)

        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            spell_ctx = TextGenerationContext(spell_side, ctx.sample)
            lines.append(cls.build_text(spell_side, include_twinpact=False, ctx=spell_ctx))

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([CardTextResources.get_civilization_text(c) for c in civs])
