# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple

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
        "world_breaker": "ワールド・ブレイカー",
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
        "power_attacker": "パワーアタッカー",
        "g_zero": "G・ゼロ"
    }

    TRIGGER_MAP = {
        "ON_PLAY": "このクリーチャーが出た時",
        "ON_OTHER_ENTER": "他のクリーチャーが出た時",
        "AT_ATTACK": "このクリーチャーが攻撃する時",
        "ON_DESTROY": "このクリーチャーが破壊された時",
        "AT_END_OF_TURN": "自分のターンの終わりに",
        "AT_END_OF_OPPONENT_TURN": "相手のターンの終わりに",
        "ON_BLOCK": "このクリーチャーがブロックした時",
        "ON_ATTACK_FROM_HAND": "手札から攻撃する時", # Revolution Change context
        "TURN_START": "自分のターンのはじめに",
        "S_TRIGGER": "S・トリガー",
        "PASSIVE_CONST": "（常在効果）",
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
        "SEARCH_DECK_BOTTOM": "自分の山札の下から{value1}枚を見る。",
        "SEARCH_DECK": "自分の山札を見る。その中から{filter}を1枚選び、{zone}に置く。その後、山札をシャッフルする。",
        "MEKRAID": "メクレイド {value1}",
        "DISCARD": "手札を{value1}枚捨てる。",
        "PLAY_FROM_ZONE": "{source_zone}からコスト{value1}以下のカードをプレイしてもよい。",
        "COUNT_CARDS": "（{filter}の数を数える）", # Internal logic
        "GET_GAME_STAT": "（ゲーム統計を取得）", # Internal logic
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "CAST_SPELL": "{target}をコストを支払わずに唱える。",
        "PUT_CREATURE": "{target}をバトルゾーンに出す。",
        "GRANT_KEYWORD": "{target}に「{str_val}」を与える。",
        "MOVE_CARD": "{target}を{zone}に置く。",
        "COST_REFERENCE": "（コスト参照処理）"
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

        # Twinpact split check
        spell_side = data.get("spell_side")

        # 1. Header (Name / Cost / Civ / Race) - Simplified for preview
        name = data.get("name", "Unknown")
        cost = data.get("cost", 0)
        civs = cls._format_civs(data.get("civilizations", []))
        type_str = cls.TYPE_MAP.get(data.get("type", "CREATURE"), data.get("type", ""))
        races = " / ".join(data.get("races", []))

        # Determine Layout based on type/twinpact (Conceptual)
        # Here we just dump text

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
        if keywords:
            for k, v in keywords.items():
                if v:
                    kw_str = cls.KEYWORD_MAP.get(k, k)
                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
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
        if spell_side:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([cls.CIVILIZATION_MAP.get(c, c) for c in civs])

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any]) -> str:
        trigger = effect.get("trigger", "NONE")
        condition = effect.get("condition", {})
        actions = effect.get("actions", [])

        # Trigger text
        trigger_text = cls.TRIGGER_MAP.get(trigger, trigger)

        # Condition text (e.g., Mana Armed)
        cond_text = ""
        cond_type = condition.get("type", "NONE")
        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ_raw = condition.get("str_val", "")
            civ = cls.CIVILIZATION_MAP.get(civ_raw, civ_raw)
            cond_text = f"マナ武装 {val} ({civ}): "
        elif cond_type == "SHIELD_COUNT":
            val = condition.get("value", 0)
            op = condition.get("op", ">=")
            cond_text = f"シールドが{val}枚{op}なら: "
        elif cond_type == "CIVILIZATION_MATCH":
             cond_text = "マナゾーンに同じ文明があれば: "
        elif cond_type == "COMPARE_STAT":
             key = condition.get("stat_key", "")
             op = condition.get("op", "=")
             val = condition.get("value", 0)
             cond_text = f"{key} {op} {val}: "

        # Actions text
        action_texts = []
        for action in actions:
            action_texts.append(cls._format_action(action))

        full_action_text = " ".join(action_texts)

        if trigger_text and trigger != "NONE" and trigger != "PASSIVE_CONST":
             return f"{trigger_text}: {cond_text}{full_action_text}"
        elif trigger == "PASSIVE_CONST":
             return f"{cond_text}{full_action_text}"
        else:
             # Just actions (e.g. Spell effect)
             return f"{cond_text}{full_action_text}"

    @classmethod
    def _format_action(cls, action: Dict[str, Any]) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = cls.ACTION_MAP.get(atype, "")

        if not template:
            return f"({atype})"

        # Parameter substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")

        # Handling for GRANT_KEYWORD str_val localization
        if atype == "GRANT_KEYWORD":
            str_val = cls.KEYWORD_MAP.get(str_val, str_val)

        # Scope/Filter resolution for {target}
        target_str, unit = cls._resolve_target(action)

        # Destination resolution for {zone}
        dest_zone = action.get("destination_zone", "")
        zone_str = cls.ZONE_MAP.get(dest_zone, dest_zone) if dest_zone else "どこか"

        # Source resolution
        src_zone = action.get("source_zone", "")
        src_str = cls.ZONE_MAP.get(src_zone, src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        # Filter substitution if {filter} exists in template (like SEARCH_DECK)
        if "{filter}" in text:
             # Basic filter description
             text = text.replace("{filter}", target_str)

        # Optional handling
        if action.get("optional", False):
            text += " (そうしてもよい)"

        return text

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any]) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        scope = action.get("scope", "NONE")
        filter_def = action.get("filter", {})

        target_desc = ""
        unit = "枚" # Default unit

        if scope == "PLAYER_OPPONENT":
            target_desc += "相手の"
        elif scope == "PLAYER_SELF" or scope == "SELF":
            target_desc += "自分の"

        # Parse filter
        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])

            # Adjectives
            if civs:
                civ_names = [cls.CIVILIZATION_MAP.get(c, c) for c in civs]
                target_desc += "/".join(civ_names) + "の"

            if races:
                target_desc += "/".join(races) + "の"

            # Nouns
            if "BATTLE_ZONE" in zones:
                if "CREATURE" in types or (not types):
                    target_desc += "クリーチャー"
                    unit = "体"
                elif "CROSS_GEAR" in types:
                    target_desc += "クロスギア"
            elif "MANA_ZONE" in zones:
                target_desc += "マナ"
            elif "HAND" in zones:
                target_desc += "手札"
            elif "SHIELD_ZONE" in zones:
                target_desc += "シールド"
                unit = "つ"
            elif "GRAVEYARD" in zones:
                if "CREATURE" in types:
                     target_desc += "クリーチャー"
                     unit = "体"
                elif "SPELL" in types:
                     target_desc += "呪文"
                else:
                     target_desc += "カード"
            elif "DECK" in zones:
                 target_desc += "カード"

            if filter_def.get("is_tapped", None) is True:
                 target_desc = "タップされている" + target_desc
            elif filter_def.get("is_tapped", None) is False:
                 target_desc = "アンタップされている" + target_desc

            if filter_def.get("is_blocker", None) is True:
                 target_desc = "ブロッカーを持つ" + target_desc

        else:
            # Implicit targets based on action type
            atype = action.get("type", "")
            if atype == "DESTROY":
                 if scope == "PLAYER_OPPONENT" or scope == "OPPONENT": # Legacy compat
                     target_desc += "クリーチャー" # Default assumption
                     unit = "体"
            elif atype == "BREAK_SHIELD":
                 pass # 'shield' is in the template
            elif atype == "TAP" or atype == "UNTAP":
                 if "クリーチャー" not in target_desc:
                      target_desc += "クリーチャー"
                      unit = "体"

        if not target_desc:
            target_desc = "カード" # Fallback

        return target_desc, unit
