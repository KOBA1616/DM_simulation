# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.action_text_generator import ActionTextGenerator

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    @classmethod
    def generate_text(cls, data: Dict[str, Any], include_twinpact: bool = True) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        """
        if not data:
            return ""

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        lines.extend(cls.generate_header_lines(data))

        # 2. Body (Keywords, Effects, etc.)
        lines.append(cls.generate_body_text_lines(data, include_twinpact=False)) # Don't recurse here, handle manually

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def generate_header_lines(cls, data: Dict[str, Any]) -> List[str]:
        lines = []
        name = data.get("name") or tr("Unknown")
        cost = data.get("cost", 0)

        # Handle both list and string formats for civilization
        civs_data = data.get("civilizations", [])
        if not civs_data and "civilization" in data:
            civ_single = data.get("civilization")
            if civ_single:
                civs_data = [civ_single]
        civs = cls._format_civs(civs_data)

        # Use CardTextResources for translation
        raw_type = data.get("type", "CREATURE")
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
    def generate_body_text_lines(cls, data: Dict[str, Any], include_twinpact: bool = True) -> str:
        """
        Generates just the body text (keywords, effects, etc.) without the header.
        """
        lines = []

        # Body Text (Keywords, Effects, etc.)
        body_text = cls.generate_body_text(data)
        if body_text:
            lines.append(body_text)

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def generate_body_text(cls, data: Dict[str, Any], sample: List[Any] = None) -> str:
        """
        Generates only the body text (Keywords, Effects, Reactions) without headers.
        Useful for structured preview and Twinpact separation.
        """
        if not data:
            return ""

        lines = []

        # 2. Keywords (ordered: basic -> special)
        keywords = data.get("keywords", {})
        basic_kw_lines = []
        special_kw_lines = []
        if keywords:
            for k, v in keywords.items():
                if not v:
                    continue

                # Build string for this keyword
                kw_str = CardTextResources.get_keyword_text(k)

                # Basic keywords: everything except the special set
                if k not in ("revolution_change", "mekraid", "friend_burst"):
                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
                    elif k == "hyper_energy":
                        kw_str += "（このクリーチャーを召喚する時、コストが異なる自分のクリーチャーを好きな数タップしてもよい、こうしてタップしたクリーチャー1体につき、このクリーチャーの召喚コストを2少なくする、ただし、コストは0以下にならない。）"
                    elif k == "mega_last_burst":
                        kw_str += "（このクリーチャーが手札、マナゾーン、または墓地に置かれた時、このカードの呪文側をコストを支払わずに唱えてもよい）"
                    elif k == "just_diver":
                        kw_str += "（このクリーチャーが出た時、次の自分のターンのはじめまで、このクリーチャーは相手に選ばれず、攻撃されない）"
                    basic_kw_lines.append(f"■ {kw_str}")
                else:
                    # Special keywords: single-line concise style. Show selected tribe/civ as requested.
                    if k == "revolution_change":
                        cond = data.get("revolution_change_condition", {})
                        if cond and isinstance(cond, dict):
                            parts = []

                            # Civilizations
                            civs = cond.get("civilizations", []) or []
                            if civs:
                                parts.append(cls._format_civs(civs))

                            # Cost
                            min_cost = cond.get("min_cost", 0)
                            max_cost = cond.get("max_cost", 999)
                            if isinstance(min_cost, dict):
                                parts.append("コストその数以上")
                            elif isinstance(max_cost, dict):
                                parts.append("コストその数以下")
                            else:
                                if min_cost > 0 and max_cost < 999:
                                    parts.append(f"コスト{min_cost}～{max_cost}")
                                elif min_cost > 0:
                                    parts.append(f"コスト{min_cost}以上")
                                elif max_cost < 999:
                                    parts.append(f"コスト{max_cost}以下")

                            # Race / Noun
                            races = cond.get("races", []) or []
                            noun = ""
                            if races:
                                noun = "/".join(races)
                            else:
                                noun = "クリーチャー"

                            # Is Evolution?
                            is_evo = cond.get("is_evolution")
                            if is_evo is True:
                                noun = "進化" + noun
                            elif is_evo is False:
                                parts.append("進化以外の")

                            # Assemble
                            adjs = "の".join(parts)
                            full_str = f"{adjs}の{noun}" if adjs else noun

                            kw_str += f"：{full_str}"
                    elif k == "friend_burst":
                        cond = data.get("friend_burst_condition", {})
                        if cond and isinstance(cond, dict):
                            races = cond.get("races", []) or []
                            if races:
                                kw_str += f"：{'/'.join(races)}"
                    special_kw_lines.append(f"■ {kw_str}")

        # Append in required order
        if basic_kw_lines:
            lines.extend(basic_kw_lines)
        if special_kw_lines:
            lines.extend(special_kw_lines)

        # 2.5 Cost Reductions
        cost_reductions = data.get("cost_reductions", [])
        for cr in cost_reductions:
            text = cls._format_cost_reduction(cr)
            if text:
                lines.append(f"■ {text}")

        # 2.6 Reaction Abilities
        reactions = data.get("reaction_abilities", [])
        for r in reactions:
            text = cls._format_reaction(r)
            if text:
                lines.append(f"■ {text}")

        # 3. Effects (Configured actions). Skip effects that only realize special keywords
        effects = data.get("effects", [])
        is_spell = data.get("type", "CREATURE") == "SPELL"

        def _is_special_only_effect(eff: Dict[str, Any]) -> bool:
            cmds = eff.get("commands", []) or []
            if not cmds:
                return False
            special_seen = False
            for cmd in cmds:
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if ctype == "MUTATE" and cmd.get("mutation_kind") == "REVOLUTION_CHANGE":
                    special_seen = True
                elif ctype == "MEKRAID" or ctype == "FRIEND_BURST":
                    special_seen = True
                elif ctype == "CAST_SPELL" and cmd.get("str_param") == "SPELL_SIDE":
                    # Use data directly to ensure safety
                    if data.get("keywords", {}).get("mega_last_burst"):
                        special_seen = True
                    else:
                        return False
                else:
                    # Found a non-special command -> not special-only
                    return False
            # All commands were special
            return special_seen

        for effect in effects:
            if _is_special_only_effect(effect):
                continue
            # Check if this card has mega_last_burst keyword and pass it to _format_effect
            has_mega_last_burst = data.get("keywords", {}).get("mega_last_burst", False)
            text = cls._format_effect(effect, is_spell, sample=sample, card_mega_last_burst=has_mega_last_burst)
            if text:
                lines.append(f"■ {text}")

        # 3.1 Static Abilities (常在効果)
        # Process static_abilities array which contains Modifier objects
        static_abilities = data.get("static_abilities", [])
        for static_ability in static_abilities:
            if static_ability and isinstance(static_ability, dict):
                text = cls._format_effect(static_ability, is_spell, sample=sample)
                if text:
                    lines.append(f"■ {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("【追加能力】")
            for effect in metamorphs:
                text = cls._format_effect(effect, is_spell, sample=sample)
                if text:
                    lines.append(f"■ {text}")

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([CardTextResources.get_civilization_text(c) for c in civs])

    @classmethod
    def _describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        types = filter_def.get("types", [])
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", 999)
        exact_cost = filter_def.get("exact_cost")
        cost_ref = filter_def.get("cost_ref")

        adjectives = []
        if civs:
            adjectives.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))

        # Handle cost filtering
        if cost_ref:
            adjectives.append("選択した数字と同じコスト")
        elif exact_cost is not None:
            adjectives.append(f"コスト{exact_cost}")
        elif isinstance(min_cost, dict):
            usage = min_cost.get("input_value_usage", "")
            if usage == "MIN_COST":
                adjectives.append("コストその数以上")
        elif min_cost > 0:
            adjectives.append(f"コスト{min_cost}以上")
        
        # Handle max_cost that might be int or dict with input_link
        if isinstance(max_cost, dict):
            usage = max_cost.get("input_value_usage", "")
            if usage == "MAX_COST":
                adjectives.append("コストその数以下")
        elif max_cost < 999:
            adjectives.append(f"コスト{max_cost}以下")

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        noun_str = "クリーチャー"
        if "ELEMENT" in types:
            noun_str = "エレメント"
        elif "SPELL" in types:
            noun_str = "呪文"

        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str

    @classmethod
    def _format_reaction(cls, reaction: Dict[str, Any]) -> str:
        if not reaction:
            return ""
        rtype = reaction.get("type", "NONE")
        if rtype == "NINJA_STRIKE":
             cost = reaction.get("cost", 0)
             return f"ニンジャ・ストライク {cost}"
        elif rtype == "STRIKE_BACK":
             return "ストライク・バック"
        elif rtype == "REVOLUTION_0_TRIGGER":
             return "革命0トリガー"
        return tr(rtype)

    @classmethod
    def _format_cost_reduction(cls, cr: Dict[str, Any]) -> str:
        if not cr:
            return ""
        ctype = cr.get("type", "PASSIVE")
        name = cr.get("name", "")
        if name:
            return f"{name}"

        unit_cost = cr.get("unit_cost", {})
        filter_def = unit_cost.get("filter", {})
        desc = cls._describe_simple_filter(filter_def)
        return f"コスト軽減: {desc}"

    # =========================================================================
    # Proxies to ActionTextGenerator
    # =========================================================================

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        return ActionTextGenerator.format_effect(effect, is_spell, sample, card_mega_last_burst)

    @classmethod
    def _format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
        return ActionTextGenerator.format_modifier(modifier, sample)
        
    @classmethod
    def _format_command(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        return ActionTextGenerator.format_command(command, is_spell, sample, card_mega_last_burst)

    @classmethod
    def generate_trigger_filter_description(cls, trigger_filter: Dict[str, Any]) -> str:
        return ActionTextGenerator.generate_trigger_filter_description(trigger_filter)

    @classmethod
    def _apply_trigger_scope(cls, trigger_text: str, scope: str, trigger_type: str, trigger_filter: Dict[str, Any] = None) -> str:
        return ActionTextGenerator._apply_trigger_scope(trigger_text, scope, trigger_type, trigger_filter)

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False) -> str:
        return ActionTextGenerator.trigger_to_japanese(trigger, is_spell)

    @classmethod
    def _zone_to_japanese(cls, zone: str) -> str:
        return ActionTextGenerator._zone_to_japanese(zone)
