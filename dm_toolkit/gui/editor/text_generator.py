# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.localization import tr

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    # Maps kept for reverse lookups or specific logic, but tr() should be preferred for output
    CIVILIZATION_MAP = {
        "LIGHT": "光",
        "WATER": "水",
        "DARKNESS": "闇",
        "FIRE": "火",
        "NATURE": "自然",
        "ZERO": "ゼロ"
    }

    # Backup map if tr() returns the key itself
    KEYWORD_TRANSLATION = {
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
        "ex_life": "EXライフ",
        "unblockable": "ブロックされない"
    }

    ACTION_MAP = {
        "DRAW_CARD": "カードを{value1}枚引く。",
        "ADD_MANA": "自分の山札の上から{value1}枚をマナゾーンに置く。",
        "DESTROY": "{target}を{value1}{unit}破壊する。",
        "TAP": "{target}を{value1}{unit}選び、タップする。",
        "UNTAP": "{target}を{value1}{unit}選び、アンタップする。",
        "RETURN_TO_HAND": "{target}を{value1}{unit}選び、手札に戻す。",
        "SEND_TO_MANA": "{target}を{value1}{unit}選び、マナゾーンに置く。",
        "MODIFY_POWER": "{target}のパワーを{value1}する。",
        "BREAK_SHIELD": "相手のシールドを{value1}つブレイクする。",
        "LOOK_AND_ADD": "自分の山札の上から{value1}枚を見る。その中から{value2}枚を手札に加え、残りを{zone}に置く。",
        "SEARCH_DECK_BOTTOM": "自分の山札の下から{value1}枚を見る。",
        "SEARCH_DECK": "自分の山札を見る。その中から{filter}を1枚選び、{zone}に置く。その後、山札をシャッフルする。",
        "MEKRAID": "メクレイド{value1}",
        "DISCARD": "手札を{value1}枚捨てる。",
        "PLAY_FROM_ZONE": "{source_zone}からコスト{value1}以下の{target}をプレイしてもよい。",
        "COUNT_CARDS": "（{filter}の数を数える）",
        "GET_GAME_STAT": "（{str_val}を参照）",
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "CAST_SPELL": "{target}をコストを支払わずに唱える。",
        "PUT_CREATURE": "{target}をバトルゾーンに出す。",
        "GRANT_KEYWORD": "{target}に「{str_val}」を与える。",
        "MOVE_CARD": "{target}を{zone}に置く。",
        "COST_REFERENCE": "",
        "SUMMON_TOKEN": "「{str_val}」を{value1}体出す。",
        "RESET_INSTANCE": "{target}の状態をリセットする（アンタップ等）。",
        "REGISTER_DELAYED_EFFECT": "「{str_val}」の効果を{value1}ターン登録する。",
        "FRIEND_BURST": "{str_val}のフレンド・バースト",
        "MOVE_TO_UNDER_CARD": "{target}を{value1}{unit}選び、カードの下に置く。",
        "SELECT_NUMBER": "数字を1つ選ぶ。",
        "DECLARE_NUMBER": "{value1}〜{value2}の数字を1つ宣言する。",
        "COST_REDUCTION": "{target}のコストを{value1}少なくする。ただし、コストは0以下にはならない。",
        "LOOK_TO_BUFFER": "{source_zone}から{value1}枚を見る（バッファへ）。",
        "SELECT_FROM_BUFFER": "バッファから{value1}枚選ぶ（{filter}）。",
        "PLAY_FROM_BUFFER": "バッファからプレイする。",
        "MOVE_BUFFER_TO_ZONE": "バッファから{zone}に置く。",
        "SELECT_OPTION": "次の中から選ぶ。",
        "LOCK_SPELL": "相手は呪文を唱えられない。",
        "APPLY_MODIFIER": "効果を付与する。",
    }

    STAT_KEY_MAP = {
        "MANA_COUNT": ("マナゾーンのカード", "枚"),
        "CREATURE_COUNT": ("クリーチャー", "体"),
        "SHIELD_COUNT": ("シールド", "つ"),
        "HAND_COUNT": ("手札", "枚"),
        "GRAVEYARD_COUNT": ("墓地のカード", "枚"),
        "BATTLE_ZONE_COUNT": ("バトルゾーンのカード", "枚"),
        "OPPONENT_MANA_COUNT": ("相手のマナゾーンのカード", "枚"),
        "OPPONENT_CREATURE_COUNT": ("相手のクリーチャー", "体"),
        "OPPONENT_SHIELD_COUNT": ("相手のシールド", "つ"),
        "OPPONENT_HAND_COUNT": ("相手の手札", "枚"),
        "OPPONENT_GRAVEYARD_COUNT": ("相手の墓地のカード", "枚"),
        "OPPONENT_BATTLE_ZONE_COUNT": ("相手のバトルゾーンのカード", "枚"),
        "CARDS_DRAWN_THIS_TURN": ("このターンに引いたカード", "枚"),
        "MANA_CIVILIZATION_COUNT": ("マナゾーンの文明数", ""),
    }

    @classmethod
    def generate_text(cls, data: Dict[str, Any]) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        """
        if not data:
            return ""

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        name = data.get("name", "Unknown")
        cost = data.get("cost", 0)

        # Handle both list and string formats for civilization
        civs_data = data.get("civilizations", [])
        if not civs_data and "civilization" in data:
            civ_single = data.get("civilization")
            if civ_single:
                civs_data = [civ_single]
        civs = cls._format_civs(civs_data)

        type_str = tr(data.get("type", "CREATURE"))
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
        if keywords:
            for k, v in keywords.items():
                if v:
                    # Try explicit map first, then tr()
                    kw_str = cls.KEYWORD_TRANSLATION.get(k, tr(k))

                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
                    elif k == "revolution_change":
                        cond = data.get("revolution_change_condition", {})
                        if cond:
                            cond_text = cls._describe_simple_filter(cond)
                            kw_str += f"：{cond_text}"
                            kw_str += f"（自分の{cond_text}が攻撃する時、そのクリーチャーと手札のこのクリーチャーと入れ替えてもよい）"
                    elif k == "hyper_energy":
                        kw_str += "（このクリーチャーを召喚する時、コストの代わりに自分のクリーチャーを2体タップしてもよい）"
                    elif k == "just_diver":
                        kw_str += "（このクリーチャーが出た時、次の自分のターンのはじめまで、このクリーチャーは相手に選ばれず、攻撃されない）"

                    kw_lines.append(f"■ {kw_str}")
        if kw_lines:
            lines.extend(kw_lines)

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

        # 3. Effects
        effects = data.get("effects", [])
        is_spell = data.get("type", "CREATURE") == "SPELL"

        for effect in effects:
            text = cls._format_effect(effect, is_spell)
            if text:
                lines.append(f"■ {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("【追加能力】")
            for effect in metamorphs:
                 text = cls._format_effect(effect, is_spell)
                 if text:
                     lines.append(f"■ {text}")

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([tr(c) for c in civs])

    @classmethod
    def _describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", 999)

        adjectives = []
        if civs:
            adjectives.append("/".join([tr(c) for c in civs]))

        if min_cost > 0:
            adjectives.append(f"コスト{min_cost}以上")
        if max_cost < 999:
            adjectives.append(f"コスト{max_cost}以下")

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        noun_str = "クリーチャー"
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

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False) -> str:
        trigger = effect.get("trigger", "NONE")
        condition = effect.get("condition", {})
        actions = effect.get("actions", [])

        trigger_text = cls.trigger_to_japanese(trigger, is_spell)
        cond_text = cls._format_condition(condition)
        cond_type = condition.get("type", "NONE")

        # Refined natural language logic
        if trigger != "NONE" and trigger != "PASSIVE_CONST":
            if cond_type == "DURING_YOUR_TURN" or cond_type == "DURING_OPPONENT_TURN":
                base_cond = cond_text.replace(": ", "")
                trigger_text = f"{base_cond}、{trigger_text}" # 自分のターン中、このクリーチャーが出た時
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"相手がカードを引いた時、{val}枚目以降なら"
                cond_text = ""

        action_texts = []
        for action in actions:
            action_texts.append(cls._format_action(action))

        full_action_text = " ".join(action_texts).strip()

        # If it's a Spell's main effect (ON_PLAY), we can often omit the trigger text "Played/Cast"
        # unless there's a condition or it's a specific sub-trigger.
        # But usually spells just list the effect. S-Trigger is handled as a Keyword (mostly),
        # but in legacy JSON it might be in effects? No, keywords.
        # However, "S-Trigger" is displayed via keywords.
        # If trigger is ON_PLAY and is_spell is True, we might suppress "このクリーチャーが出た時"
        # but if it was mapped to "呪文を唱えた時", we might keep it?
        # Standard duel masters text: Spells don't say "When you cast this spell" for the main effect.
        if is_spell and trigger == "ON_PLAY":
            trigger_text = ""

        if trigger_text and trigger != "NONE" and trigger != "PASSIVE_CONST":
             if not full_action_text:
                 return ""
             return f"{trigger_text}: {cond_text}{full_action_text}"
        elif trigger == "PASSIVE_CONST":
             return f"{cond_text}{full_action_text}"
        else:
             return f"{cond_text}{full_action_text}"

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False) -> str:
        mapping = {
            "ON_PLAY": "このクリーチャーが出た時" if not is_spell else "この呪文を唱えた時", # Suppressed later for main spell effect
            "ON_OTHER_ENTER": "他のクリーチャーが出た時",
            "AT_ATTACK": "このクリーチャーが攻撃する時",
            "ON_DESTROY": "このクリーチャーが破壊された時",
            "AT_END_OF_TURN": "自分のターンの終わりに",
            "AT_END_OF_OPPONENT_TURN": "相手のターンの終わりに",
            "ON_BLOCK": "このクリーチャーがブロックした時",
            "ON_ATTACK_FROM_HAND": "手札から攻撃する時",
            "TURN_START": "自分のターンのはじめに",
            "S_TRIGGER": "S・トリガー",
            "PASSIVE_CONST": "（常在効果）",
            "ON_SHIELD_ADD": "カードがシールドゾーンに置かれた時",
            "AT_BREAK_SHIELD": "シールドをブレイクする時",
            "ON_CAST_SPELL": "呪文を唱えた時",
            "ON_OPPONENT_DRAW": "相手がカードを引いた時",
            "NONE": ""
        }
        return mapping.get(trigger, trigger)

    @classmethod
    def _format_condition(cls, condition: Dict[str, Any]) -> str:
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ_raw = condition.get("str_val", "")
            civ = tr(civ_raw)
            return f"マナ武装 {val} ({civ}): "

        elif cond_type == "SHIELD_COUNT":
            val = condition.get("value", 0)
            op = condition.get("op", ">=")
            op_text = "以上" if op == ">=" else "以下" if op == "<=" else ""
            if op == "=": op_text = ""
            return f"自分のシールドが{val}つ{op_text}なら: "

        elif cond_type == "CIVILIZATION_MATCH":
             return "マナゾーンに同じ文明があれば: "

        elif cond_type == "COMPARE_STAT":
             key = condition.get("stat_key", "")
             op = condition.get("op", "=")
             val = condition.get("value", 0)
             stat_name, unit = cls.STAT_KEY_MAP.get(key, (key, ""))

             op_text = ""
             if op == ">=":
                 op_text = f"{val}{unit}以上"
             elif op == "<=":
                 op_text = f"{val}{unit}以下"
             elif op == "=" or op == "==":
                 op_text = f"{val}{unit}"
             elif op == ">":
                 op_text = f"{val}{unit}より多い"
             elif op == "<":
                 op_text = f"{val}{unit}より少ない"
             return f"自分の{stat_name}が{op_text}なら: "

        elif cond_type == "OPPONENT_PLAYED_WITHOUT_MANA":
            return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた時: "

        elif cond_type == "OPPONENT_DRAW_COUNT":
            val = condition.get("value", 0)
            return f"{val}枚目以降なら: "

        elif cond_type == "DURING_YOUR_TURN":
            return "自分のターン中: "
        elif cond_type == "DURING_OPPONENT_TURN":
            return "相手のターン中: "
        elif cond_type == "EVENT_FILTER_MATCH":
            return ""

        return ""

    @classmethod
    def _format_action(cls, action: Dict[str, Any]) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = cls.ACTION_MAP.get(atype, "")

        # Determine verb form (standard or optional)
        optional = action.get("optional", False)

        # Resolve dynamic target strings
        target_str, unit = cls._resolve_target(action)

        # Complex Action Logic
        if atype == "MODIFY_POWER":
            val = action.get("value1", 0)
            sign = "+" if val >= 0 else ""
            return f"{target_str}のパワーを{sign}{val}する。"

        elif atype == "SELECT_NUMBER":
            val1 = action.get("value1", 0)
            val2 = action.get("value2", 0)
            if val1 > 0 and val2 > 0:
                 template = f"{val1}〜{val2}の数字を1つ選ぶ。"

        elif atype == "COST_REFERENCE":
            str_val = action.get("str_val", "")
            if str_val == "G_ZERO":
                cond = action.get("condition", {})
                cond_text = cls._format_condition(cond).strip().rstrip(':')
                return f"G・ゼロ：{cond_text}（このクリーチャーをコストを支払わずに召喚してもよい）"
            elif str_val == "HYPER_ENERGY":
                return "ハイパーエナジー"
            elif str_val in ["SYM_CREATURE", "SYM_SPELL", "SYM_SHIELD"]:
                val1 = action.get("value1", 0)
                cost_term = "召喚コスト" if "CREATURE" in str_val else "コスト"
                return f"{target_str}1{unit}につき、このクリーチャーの{cost_term}を{val1}少なくする。ただし、コストは0以下にはならない。"
            elif str_val == "CARDS_DRAWN_THIS_TURN":
                val1 = action.get("value1", 0)
                return f"このターンに引いたカード1枚につき、このクリーチャーの召喚コストを{val1}少なくする。ただし、コストは0以下にはならない。"
            else:
                filter_def = action.get("filter")
                if filter_def:
                     val1 = action.get("value1", 0)
                     return f"{target_str}1{unit}につき、このクリーチャーのコストを{val1}少なくする。ただし、コストは0以下にはならない。"
                return "コストを軽減する。"

        elif atype == "SELECT_OPTION":
            options = action.get("options", [])
            lines = []
            val1 = action.get("value1", 1)
            lines.append(f"次の中から{val1}回選ぶ。（同じものを選んでもよい）")
            for i, opt_chain in enumerate(options):
                chain_text = " ".join([cls._format_action(a) for a in opt_chain])
                lines.append(f"▶ {chain_text}")
            return "\n".join(lines)

        elif atype == "MEKRAID":
            val1 = action.get("value1", 0)
            # Add simple civ detection if possible, otherwise generic
            return f"メクレイド{val1}（自分の山札の上から3枚を見る。その中からコスト{val1}以下のクリーチャーを1体、コストを支払わずに召喚してもよい。残りを山札の下に好きな順序で置く）"

        elif atype == "FRIEND_BURST":
            str_val = action.get("str_val", "")
            return f"＜{str_val}＞のフレンド・バースト（このクリーチャーが出た時、このクリーチャーをタップしてもよい。そうしたら、このクリーチャーの呪文側を、バトルゾーンに置いたままコストを支払わずに唱える。）"

        elif atype == "REVOLUTION_CHANGE":
             return "" # Covered by keyword

        elif atype == "APPLY_MODIFIER":
             # Need to know WHAT modifier
             str_val = action.get("str_val", "")
             val1 = action.get("value1", 0)
             if str_val == "SPEED_ATTACKER":
                 return f"{target_str}に「スピードアタッカー」を与える。"
             elif str_val == "BLOCKER":
                 return f"{target_str}に「ブロッカー」を与える。"
             elif str_val == "SLAYER":
                 return f"{target_str}に「スレイヤー」を与える。"
             elif str_val == "COST":
                 sign = "少なくする" if val1 > 0 else "増やす"
                 return f"{target_str}のコストを{abs(val1)}{sign}。"
             else:
                 return f"{target_str}に効果（{str_val}）を与える。"

        if not template:
            return f"({tr(atype)})"

        # Parameter Substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")
        input_key = action.get("input_value_key", "")

        is_generic_selection = atype in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA", "MOVE_CARD"]

        if input_key:
             val1 = "（その数）"
        elif val1 == 0 and is_generic_selection:
             # Logic for "All" if 0 and generic
             if atype == "DESTROY": template = "{target}をすべて破壊する。"
             elif atype == "TAP": template = "{target}をすべてタップする。"
             elif atype == "UNTAP": template = "{target}をすべてアンタップする。"
             elif atype == "RETURN_TO_HAND": template = "{target}をすべて手札に戻す。"
             elif atype == "SEND_TO_MANA": template = "{target}をすべてマナゾーンに置く。"
             elif atype == "MOVE_CARD":
                 # Fallback handled in specific logic below, this is just template swap
                 pass

        if atype == "GRANT_KEYWORD":
            str_val = tr(str_val)

        elif atype == "GET_GAME_STAT":
            stat_name = cls.STAT_KEY_MAP.get(str_val, (str_val, ""))[0]
            return f"（{stat_name}を参照）"

        elif atype == "COUNT_CARDS":
            mode = str_val
            if not mode or mode == "CARDS_MATCHING_FILTER":
                 return f"（{target_str}の数を数える）"
            else:
                 stat_name = cls.STAT_KEY_MAP.get(mode, (mode, ""))[0]
                 return f"（{stat_name}を数える）"

        elif atype == "MOVE_CARD":
            dest_zone = action.get("destination_zone", "")
            # Check for "All" condition
            is_all = (val1 == 0 and not input_key)

            if dest_zone == "HAND":
                template = "{target}を{value1}{unit}選び、手札に戻す。"
                if is_all: template = "{target}をすべて手札に戻す。"
            elif dest_zone == "MANA_ZONE":
                template = "{target}を{value1}{unit}選び、マナゾーンに置く。"
                if is_all: template = "{target}をすべてマナゾーンに置く。"
            elif dest_zone == "GRAVEYARD":
                template = "{target}を{value1}{unit}選び、墓地に置く。"
                if is_all: template = "{target}をすべて墓地に置く。"
            elif dest_zone == "DECK_BOTTOM":
                template = "{target}を{value1}{unit}選び、山札の下に置く。"
                if is_all: template = "{target}をすべて山札の下に置く。"

        elif atype == "PLAY_FROM_ZONE":
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter

            if not action.get("source_zone") and "zones" in temp_filter:
                zones = temp_filter["zones"]
                if len(zones) == 1:
                    action["source_zone"] = zones[0]

            if action.get("value1", 0) == 0:
                max_cost = temp_filter.get("max_cost", 999)
                if max_cost < 999:
                    action["value1"] = max_cost
                    if not input_key: val1 = max_cost
                    if "max_cost" in temp_filter: del temp_filter["max_cost"]

            if "zones" in temp_filter: temp_filter["zones"] = []
            scope = action.get("scope", "NONE")
            if scope in ["PLAYER_SELF", "SELF"]: action["scope"] = "NONE"

            # Re-resolve target with cleaned filter
            target_str, unit = cls._resolve_target(action)
            verb = "プレイする"
            types = temp_filter.get("types", [])
            if "SPELL" in types and "CREATURE" not in types:
                verb = "唱える"
            elif "CREATURE" in types:
                verb = "召喚する"

            if action.get("source_zone"):
                template = "{source_zone}からコスト{value1}以下の{target}を" + verb + "。"
            else:
                template = "コスト{value1}以下の{target}を" + verb + "。"

        # Destination/Source Resolution
        dest_zone = action.get("destination_zone", "")
        zone_str = tr(dest_zone) if dest_zone else "どこか"
        src_zone = action.get("source_zone", "")
        src_str = tr(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        if "{filter}" in text:
             text = text.replace("{filter}", target_str)

        if atype == "COST_REDUCTION":
            if target_str == "カード" or target_str == "自分のカード":
                text = text.replace("カード", "このクリーチャー")
                text = text.replace("自分のカード", "このクリーチャー")
            cond = action.get("condition", {})
            if cond:
                cond_text = cls._format_condition(cond)
                text = f"{cond_text}{text}"

        # Verb Conjugation for Optional Actions
        if optional:
            if text.endswith("する。"):
                text = text[:-3] + "してもよい。"
            elif text.endswith("く。"): # 引く、置く
                text = text[:-2] + "いてもよい。" # 引いてもよい
            elif text.endswith("す。"): # 戻す、出す
                text = text[:-2] + "してもよい。" # 戻してもよい
            elif text.endswith("る。"): # 見る、捨てる、唱える
                text = text[:-2] + "てもよい。"
            elif text.endswith("う。"): # 支払う
                text = text[:-2] + "ってもよい。"
            else:
                if not text.endswith("てもよい。"):
                    text = text[:-1] + "てもよい。"

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

        target_desc = ""
        prefix = ""
        adjectives = ""
        zone_noun = ""
        type_noun = "カード"
        unit = "枚"

        if atype == "DISCARD" and scope == "NONE":
             scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
             return ("このクリーチャー", "枚")

        if scope == "PLAYER_OPPONENT": prefix = "相手の"
        elif scope == "PLAYER_SELF" or scope == "SELF": prefix = "自分の"
        elif scope == "ALL_PLAYERS": prefix = "すべてのプレイヤーの"
        elif scope == "RANDOM": prefix = "ランダムな"

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])
            owner = filter_def.get("owner", "NONE")

            # Handle explicit owner filter if scope is generic
            if owner == "PLAYER_OPPONENT" and not prefix: prefix = "相手の"
            elif owner == "PLAYER_SELF" and not prefix: prefix = "自分の"

            temp_adjs = []
            if civs: temp_adjs.append("/".join([tr(c) for c in civs]))
            if races: temp_adjs.append("/".join(races))

            if temp_adjs: adjectives += "/".join(temp_adjs) + "の"

            if filter_def.get("min_cost", 0) > 0: adjectives += f"コスト{filter_def['min_cost']}以上の"
            if filter_def.get("max_cost", 999) < 999: adjectives += f"コスト{filter_def['max_cost']}以下の"

            if filter_def.get("is_tapped", None) is True: adjectives = "タップされている" + adjectives
            elif filter_def.get("is_tapped", None) is False: adjectives = "アンタップされている" + adjectives
            if filter_def.get("is_blocker", None) is True: adjectives = "ブロッカーを持つ" + adjectives
            if filter_def.get("is_evolution", None) is True: adjectives = "進化" + adjectives

            if "BATTLE_ZONE" in zones:
                zone_noun = "バトルゾーン"
                if "CARD" in types:
                    type_noun = "カード"
                    unit = "枚"
                elif "ELEMENT" in types:
                    type_noun = "エレメント"
                    unit = "つ"
                elif "CREATURE" in types or (not types):
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "CROSS_GEAR" in types:
                    type_noun = "クロスギア"
            elif "MANA_ZONE" in zones:
                zone_noun = "マナゾーン"
            elif "HAND" in zones:
                zone_noun = "手札"
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

            if not zone_noun:
                if "CARD" in types:
                    type_noun = "カード"
                    unit = "枚"
                elif "ELEMENT" in types:
                    type_noun = "エレメント"
                    unit = "つ"
                elif "CREATURE" in types:
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "SPELL" in types:
                    type_noun = "呪文"
                elif len(types) > 1:
                     # Join multiple types (e.g., Creature/Spell)
                     type_noun = "または".join([tr(t) for t in types])

            # Special case for SEARCH_DECK
            if atype == "SEARCH_DECK":
                 # Usually searching for a specific card in deck
                 pass

            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "の")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)
            target_desc = "".join(parts)

            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"
            if "BATTLE_ZONE" in zones:
                 target_desc = target_desc.replace("バトルゾーンの", "")

        else:
            if atype == "DESTROY":
                 if scope == "PLAYER_OPPONENT" or scope == "OPPONENT":
                     target_desc = "相手のクリーチャー"
                     unit = "体"
            elif atype == "TAP" or atype == "UNTAP":
                 if "クリーチャー" not in target_desc:
                      target_desc = prefix + "クリーチャー"
                      unit = "体"
            elif atype == "DISCARD":
                 target_desc = "手札"
            else:
                 target_desc = "カード"

        if not target_desc: target_desc = "カード"
        return target_desc, unit
