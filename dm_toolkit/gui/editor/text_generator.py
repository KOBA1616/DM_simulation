# -*- coding: utf-8 -*-
from typing import Dict, Any, List

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    CIVILIZATION_MAP = {
        "LIGHT": "光",
        "WATER": "水",
        "DARKNESS": "闇",
        "FIRE": "火",
        "NATURE": "自然",
        "ZERO": "ゼロ"
    }

    TYPE_MAP = {
        "CREATURE": "クリーチャー",
        "SPELL": "呪文",
        "EVOLUTION_CREATURE": "進化クリーチャー",
        "TAMASEED": "タマシード",
        "CROSS_GEAR": "クロスギア",
        "CASTLE": "城",
        "PSYCHIC_CREATURE": "サイキック・クリーチャー",
        "GR_CREATURE": "GRクリーチャー"
    }

    KEYWORD_MAP = {
        "speed_attacker": "スピードアタッカー",
        "blocker": "ブロッカー",
        "slayer": "スレイヤー",
        "double_breaker": "W・ブレイカー",
        "triple_breaker": "T・ブレイカー",
        "shield_trigger": "S・トリガー",
        "evolution": "進化",
        "just_diver": "ジャストダイバー",
        "mach_fighter": "マッハファイター",
        "g_strike": "G・ストライク",
        "hyper_energy": "ハイパーエナジー",
        "shield_burn": "シールド焼却",
        "revolution_change": "革命チェンジ",
        "untap_in": "タップして出る",
        "meta_counter_play": "メタカウンター",
        "power_attacker": "パワーアタッカー"
    }

    TRIGGER_MAP = {
        "ON_PLAY": "このクリーチャーが出た時",
        "AT_ATTACK": "このクリーチャーが攻撃する時",
        "ON_DESTROY": "このクリーチャーが破壊された時",
        "AT_END_OF_TURN": "自分のターンの終わりに",
        "AT_END_OF_OPPONENT_TURN": "相手のターンの終わりに",
        "ON_BLOCK": "このクリーチャーがブロックした時",
        "ON_ATTACK_FROM_HAND": "手札から攻撃する時", # Revolution Change context
        "NONE": "" # Passive
    }

    ACTION_MAP = {
        "DRAW_CARD": "カードを{value1}枚引く。",
        "ADD_MANA": "自分の山札の上から{value1}枚をマナゾーンに置く。",
        "DESTROY": "{target}を{value1}{unit}破壊する。",
        "TAP": "{target}を{value1}{unit}選び、タップする。",
        "UNTAP": "{target}を{value1}{unit}選び、アンタップする。",
        "RETURN_TO_HAND": "{target}を{value1}{unit}選び、手札に戻す。",
        "SEND_TO_MANA": "{target}を{value1}{unit}選び、マナゾーンに置く。",
        "MODIFY_POWER": "{target}のパワーを{value1}する。", # Duration?
        "BREAK_SHIELD": "相手のシールドを{value1}つブレイクする。",
        "LOOK_AND_ADD": "自分の山札の上から{value1}枚を見る。その中から{value2}枚を手札に加え、残りを{zone}に置く。",
        "SEARCH_DECK_BOTTOM": "自分の山札の下から{value1}枚を見る...", # Context dependent
        "SEARCH_DECK": "自分の山札を見る。その中から{filter}を1枚選び、{zone}に置く。その後、山札をシャッフルする。",
        "MEKRAID": "メクレイド {value1}",
        "DISCARD": "手札を{value1}枚捨てる。",
        "PLAY_FROM_ZONE": "{source_zone}からコスト{value1}以下のカードをプレイしてもよい。",
        "COUNT_CARDS": "", # Internal logic
        "GET_GAME_STAT": "", # Internal logic
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "G_ZERO": "G・ゼロ"
    }

    ZONE_MAP = {
        "HAND": "手札",
        "MANA_ZONE": "マナゾーン",
        "GRAVEYARD": "墓地",
        "DECK": "山札",
        "BATTLE_ZONE": "バトルゾーン",
        "SHIELD_ZONE": "シールドゾーン",
        "DECK_BOTTOM": "山札の下"
    }

    @classmethod
    def generate_text(cls, data: Dict[str, Any]) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        Handles Twinpact by recursively calling itself or formatting specifically.
        """
        if not data:
            return ""

        lines = []

        # 1. Header (Name / Cost / Civ / Race) - Simplified for preview
        name = data.get("name", "Unknown")
        cost = data.get("cost", 0)
        civs = cls._format_civs(data.get("civilizations", []))
        type_str = cls.TYPE_MAP.get(data.get("type", "CREATURE"), data.get("type", ""))
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

        # 2. Keywords
        keywords = data.get("keywords", {})
        kw_lines = []
        for k, v in keywords.items():
            if v:
                kw_str = cls.KEYWORD_MAP.get(k, k)
                if k == "power_attacker":
                    # Usually Power Attacker +X
                    pass
                kw_lines.append(f"■ {kw_str}")

        if kw_lines:
            lines.extend(kw_lines)

        # 3. Effects
        effects = data.get("effects", [])
        for effect in effects:
            text = cls._format_effect(effect)
            if text:
                lines.append(f"■ {text}")

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side:
            lines.append("\n" + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([cls.CIVILIZATION_MAP.get(c, c) for c in civs])

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any]) -> str:
        condition = effect.get("condition", {})
        action = effect.get("action", {})

        cond_type = condition.get("type", "NONE")
        cond_text = cls.TRIGGER_MAP.get(cond_type, "")

        # Handle specific conditions like MANA_ARMED
        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ = cls.CIVILIZATION_MAP.get(condition.get("str_val", ""), "")
            cond_text = f"マナ武装 {val} ({civ}): "

        action_text = cls._format_action(action)

        if cond_text:
            return f"{cond_text}\n  {action_text}"
        return action_text

    @classmethod
    def _format_action(cls, action: Dict[str, Any]) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = cls.ACTION_MAP.get(atype, "")

        if not template:
            # Fallback for composite actions or unknown types
            if atype == "SEQ_ACTION": # Sequence
                sub_actions = action.get("children", []) # Assuming structure
                return " -> ".join([cls._format_action(sa) for sa in sub_actions])
            return atype

        # Parameter substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")

        # Scope/Filter resolution for {target}
        target_str, unit = cls._resolve_target(action)

        zone_str = "どこか" # Default

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str) # Placeholder

        # Optional handling
        if action.get("optional", False):
            text += " (そうしてもよい)"

        return text

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any]) -> tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        scope = action.get("scope", "")
        filter_def = action.get("filter", {})

        target_desc = ""
        unit = "枚" # Default unit

        if scope == "OPPONENT":
            target_desc += "相手の"
        elif scope == "SELF":
            target_desc += "自分の"

        # Parse filter
        if filter_def:
            zones = filter_def.get("zones", [])
            card_types = filter_def.get("card_types", [])

            if "BATTLE_ZONE" in zones:
                if "CREATURE" in card_types or not card_types:
                    target_desc += "クリーチャー"
                    unit = "体"
                elif "CROSS_GEAR" in card_types:
                    target_desc += "クロスギア"
            elif "MANA_ZONE" in zones:
                target_desc += "マナ"
            elif "HAND" in zones:
                target_desc += "手札"
            elif "SHIELD_ZONE" in zones:
                target_desc += "シールド"
                unit = "つ"
        else:
            # Implicit targets based on action type
            atype = action.get("type", "")
            if atype == "DESTROY":
                 if scope == "OPPONENT":
                     target_desc += "クリーチャー" # Default assumption
                     unit = "体"
            elif atype == "BREAK_SHIELD":
                 pass # 'shield' is in the template

        if not target_desc:
            target_desc = "カード" # Fallback

        return target_desc, unit
