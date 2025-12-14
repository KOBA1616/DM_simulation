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
        "g_zero": "G・ゼロ",
        "ex_life": "EXライフ"
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
        "GET_GAME_STAT": "（{str_val}を参照）", # Updated to show stat key
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "CAST_SPELL": "{target}をコストを支払わずに唱える。",
        "PUT_CREATURE": "{target}をバトルゾーンに出す。",
        "GRANT_KEYWORD": "{target}に「{str_val}」を与える。",
        "MOVE_CARD": "{target}を{zone}に置く。",
        "COST_REFERENCE": "", # Handled specifically in _format_action

        # New Additions
        "SUMMON_TOKEN": "「{str_val}」を{value1}体出す。",
        "RESET_INSTANCE": "{target}の状態をリセットする（アンタップ等）。",
        "REGISTER_DELAYED_EFFECT": "「{str_val}」の効果を{value1}ターン登録する。",
        "FRIEND_BURST": "{str_val}のフレンド・バースト",
        "MOVE_TO_UNDER_CARD": "{target}を{value1}{unit}選び、カードの下に置く。",
        "SELECT_NUMBER": "数字を1つ選ぶ。", # Placeholder, logic handled in method
        "DECLARE_NUMBER": "{value1}〜{value2}の数字を1つ宣言する。",
        "COST_REDUCTION": "{target}のコストを{value1}少なくする。ただし、コストは0以下にはならない。",
        "LOOK_TO_BUFFER": "{source_zone}から{value1}枚を見る（バッファへ）。",
        "SELECT_FROM_BUFFER": "バッファから{value1}枚選ぶ（{filter}）。",
        "PLAY_FROM_BUFFER": "バッファからプレイする。",
        "MOVE_BUFFER_TO_ZONE": "バッファから{zone}に置く。"
    }

    ZONE_MAP = {
        "HAND": "手札",
        "MANA_ZONE": "マナゾーン",
        "GRAVEYARD": "墓地",
        "DECK": "山札",
        "BATTLE_ZONE": "バトルゾーン",
        "SHIELD_ZONE": "シールドゾーン",
        "DECK_BOTTOM": "山札の下",
        "DECK_TOP": "山札の上"
    }

    STAT_KEY_MAP = {
        "MANA_COUNT": ("マナゾーンのカード", "枚"),
        "CREATURE_COUNT": ("クリーチャー", "体"),
        "SHIELD_COUNT": ("シールド", "つ"),
        "HAND_COUNT": ("手札", "枚"),
        "GRAVEYARD_COUNT": ("墓地のカード", "枚"),
        "BATTLE_ZONE_COUNT": ("バトルゾーンのカード", "枚"),
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
        cond_text = cls._format_condition(condition)

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
    def _format_condition(cls, condition: Dict[str, Any]) -> str:
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ_raw = condition.get("str_val", "")
            civ = cls.CIVILIZATION_MAP.get(civ_raw, civ_raw)
            return f"マナ武装 {val} ({civ}): "

        elif cond_type == "SHIELD_COUNT":
            val = condition.get("value", 0)
            op = condition.get("op", ">=")
            return f"シールドが{val}枚{op}なら: "

        elif cond_type == "CIVILIZATION_MATCH":
             return "マナゾーンに同じ文明があれば: "

        elif cond_type == "COMPARE_STAT":
             key = condition.get("stat_key", "")
             op = condition.get("op", "=")
             val = condition.get("value", 0)

             # Resolve key and unit
             stat_name, unit = cls.STAT_KEY_MAP.get(key, (key, ""))

             # Resolve op
             op_text = f"{op} {val}"
             if op == ">=":
                 op_text = f"が{val}{unit}以上"
             elif op == "<=":
                 op_text = f"が{val}{unit}以下"
             elif op == "=" or op == "==":
                 op_text = f"が{val}{unit}"
             elif op == ">":
                 op_text = f"が{val}{unit}より多い"
             elif op == "<":
                 op_text = f"が{val}{unit}より少ない"

             return f"{stat_name}{op_text}なら: "

        elif cond_type == "OPPONENT_PLAYED_WITHOUT_MANA":
            return "相手がマナゾーンのカードをタップせずにクリーチャーを出すか呪文を唱えた時: "

        elif cond_type == "DURING_YOUR_TURN":
            return "自分のターン中: "

        elif cond_type == "DURING_OPPONENT_TURN":
            return "相手のターン中: "

        return ""

    @classmethod
    def _format_action(cls, action: Dict[str, Any]) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = cls.ACTION_MAP.get(atype, "")

        # Special handling for SELECT_NUMBER to show range, overriding default template if range exists
        if atype == "SELECT_NUMBER":
            val1 = action.get("value1", 0)
            val2 = action.get("value2", 0)
            if val1 > 0 and val2 > 0:
                 template = f"{val1}〜{val2}の数字を1つ選ぶ。"

        # COST_REFERENCE handling
        if atype == "COST_REFERENCE":
            str_val = action.get("str_val", "")
            if str_val == "G_ZERO":
                # G-Zero: condition inside action
                cond = action.get("condition", {})
                cond_text = cls._format_condition(cond).strip().rstrip(':') # remove trailing colon
                return f"G・ゼロ：{cond_text}（このクリーチャーをコストを支払わずに召喚してもよい）"
            elif str_val == "HYPER_ENERGY":
                return "ハイパーエナジー"
            elif str_val in ["SYM_CREATURE", "SYM_SPELL", "SYM_SHIELD"]:
                # Sympathy: Target is counted
                target_desc, unit = cls._resolve_target(action)
                val1 = action.get("value1", 0)
                # "Target" 1 count reduces cost by value1
                # If target_desc implies "My ...", it works.
                # Use "召喚コスト" if it's counting creatures/context implies summon, but safest is "コスト".
                # However, Sympathy is usually for summoning.
                cost_term = "召喚コスト" if "CREATURE" in str_val else "コスト"
                return f"{target_desc}1{unit}につき、このクリーチャーの{cost_term}を{val1}少なくする。ただし、コストは0以下にはならない。"
            else:
                return "コストを軽減する"

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

        # Post-processing for specific actions to improve natural language
        if atype == "COST_REDUCTION":
            # If target is "Card", replace with "This creature" contextually?
            # Or if it is "My Card", replace with "This creature".
            # For simplicity, if target_str is exactly "カード" or "自分のカード" and filter is empty-ish?
            # _resolve_target returns "カード" if no filter.
            if target_str == "カード" or target_str == "自分のカード":
                text = text.replace("カード", "このクリーチャー")
                text = text.replace("自分のカード", "このクリーチャー") # Just in case

        # Optional handling ("Up to" / "You may")
        if action.get("optional", False):
            # Generalized replacement for "N[unit]Select" -> "N[unit]Up to Select"
            # We look for "{val1}{unit}選び" or "{val1}{unit}破壊する" or similar patterns implied by template

            # Common patterns in ACTION_MAP:
            # "{val1}{unit}選び" (TAP, UNTAP, RETURN_TO_HAND, SEND_TO_MANA, etc.)
            # "{val1}{unit}破壊する" (DESTROY)

            # We construct the phrase that was likely generated:
            phrase_select = f"{val1}{unit}選び"
            phrase_destroy = f"{val1}{unit}破壊する"
            phrase_discard = f"{val1}{unit}捨てる"
            phrase_break = f"{val1}{unit}ブレイクする"

            if phrase_select in text:
                text = text.replace(phrase_select, f"{val1}{unit}まで選び")
            elif phrase_destroy in text:
                text = text.replace(phrase_destroy, f"{val1}{unit}まで破壊する")
            elif phrase_discard in text:
                text = text.replace(phrase_discard, f"{val1}{unit}まで捨てる")
            elif phrase_break in text:
                text = text.replace(phrase_break, f"{val1}{unit}までブレイクする")

        return text

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any]) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        scope = action.get("scope", "NONE")
        filter_def = action.get("filter", {})
        atype = action.get("type", "")

        prefix = "" # Owner scope (My, Opponent's)
        adjectives = "" # Civs, Races, Costs
        noun = "" # Zone or Type (Card, Creature)
        unit = "枚" # Default unit
        target_desc = ""

        # Implicit handling for certain actions that don't always specify scope but imply it (e.g., DISCARD imply SELF)
        if atype == "DISCARD" and scope == "NONE":
             scope = "PLAYER_SELF"
        # COST_REDUCTION on self (empty filter often implies self or all own cards?)
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
             return ("このクリーチャー", "枚")

        # 1. Scope Prefix
        if scope == "PLAYER_OPPONENT":
            prefix = "相手の"
        elif scope == "PLAYER_SELF" or scope == "SELF":
            prefix = "自分の"
        elif scope == "ALL_PLAYERS":
            prefix = "すべてのプレイヤーの"
        elif scope == "RANDOM":
            prefix = "ランダムな"

        # 2. Parse Filter (Adjectives & Noun)
        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])

            # Adjectives
            temp_adjs = []
            if civs:
                temp_adjs.append("/".join([cls.CIVILIZATION_MAP.get(c, c) for c in civs]))
            if races:
                temp_adjs.append("/".join(races))

            # Combine Civ/Race with "の"
            if temp_adjs:
                adjectives += "/".join(temp_adjs) + "の"

            # Add cost constraints
            if filter_def.get("min_cost", 0) > 0:
                 adjectives += f"コスト{filter_def['min_cost']}以上の"
            if filter_def.get("max_cost", 999) < 999:
                 adjectives += f"コスト{filter_def['max_cost']}以下の"

            # Tapped state is also an adjective
            if filter_def.get("is_tapped", None) is True:
                 adjectives = "タップされている" + adjectives
            elif filter_def.get("is_tapped", None) is False:
                 adjectives = "アンタップされている" + adjectives

            if filter_def.get("is_blocker", None) is True:
                 adjectives = "ブロッカーを持つ" + adjectives

            # Noun (Type or Zone context)
            zone_noun = ""
            type_noun = "カード"

            if "BATTLE_ZONE" in zones:
                zone_noun = "バトルゾーン"
                if "CREATURE" in types or (not types):
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "CROSS_GEAR" in types:
                    type_noun = "クロスギア"
            elif "MANA_ZONE" in zones:
                zone_noun = "マナゾーン"
                type_noun = "カード"
            elif "HAND" in zones:
                zone_noun = "手札"
                type_noun = "カード"
            elif "SHIELD_ZONE" in zones:
                zone_noun = "シールドゾーン"
                type_noun = "カード"
                unit = "つ"
            elif "GRAVEYARD" in zones:
                zone_noun = "墓地"
                if "CREATURE" in types:
                     type_noun = "クリーチャー"
                     unit = "体"
                elif "SPELL" in types:
                     type_noun = "呪文"
            elif "DECK" in zones:
                 zone_noun = "山札"
                 type_noun = "カード"

            # If types are specified but no zone, use type noun
            if not zone_noun:
                if "CREATURE" in types:
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "SPELL" in types:
                    type_noun = "呪文"

            # Construct description
            # Pattern: [Prefix] [Zone]の [Adj] [Type]
            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "の")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)

            target_desc = "".join(parts)

            # Refinement
            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"

            if "BATTLE_ZONE" in zones:
                 target_desc = target_desc.replace("バトルゾーンの", "")

        else:
            # Implicit targets based on action type
            if atype == "DESTROY":
                 if scope == "PLAYER_OPPONENT" or scope == "OPPONENT": # Legacy compat
                     target_desc = "相手のクリーチャー"
                     unit = "体"
            elif atype == "BREAK_SHIELD":
                 pass
            elif atype == "TAP" or atype == "UNTAP":
                 if "クリーチャー" not in target_desc:
                      target_desc = prefix + "クリーチャー"
                      unit = "体"
            elif atype == "DISCARD":
                 target_desc = "手札"

        if not target_desc:
            target_desc = "カード" # Fallback

        return target_desc, unit
