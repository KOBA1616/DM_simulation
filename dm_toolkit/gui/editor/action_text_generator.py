# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources

class ActionTextGenerator:
    """
    Handles text generation for individual actions and commands.
    Separated from CardTextGenerator to improve maintainability.
    """

    @classmethod
    def format_command(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        return cls._format_command(command, is_spell, sample, card_mega_last_burst)

    @classmethod
    def format_effect(cls, effect: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        return cls._format_effect(effect, is_spell, sample, card_mega_last_burst)

    @classmethod
    def format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
        return cls._format_modifier(modifier, sample)

    @classmethod
    def generate_trigger_filter_description(cls, trigger_filter: Dict[str, Any]) -> str:
        if not trigger_filter:
            return ""

        descriptions = []
        types = trigger_filter.get("types", [])
        if types:
            type_names = []
            for t in types:
                if t == "CREATURE": type_names.append("クリーチャー")
                elif t == "SPELL": type_names.append("呪文")
                elif t == "ELEMENT": type_names.append("エレメント")
                elif t == "CARD": type_names.append("カード")
            if type_names:
                descriptions.append("/".join(type_names))

        civs = trigger_filter.get("civilizations", [])
        if civs:
            civ_names = [CardTextResources.get_civilization_text(c) for c in civs]
            descriptions.append("/".join([c for c in civ_names if c]))

        races = trigger_filter.get("races", [])
        if races:
            descriptions.append("/".join(races))

        exact_cost = trigger_filter.get("exact_cost")
        cost_ref = trigger_filter.get("cost_ref")
        min_cost = trigger_filter.get("min_cost", 0)
        if min_cost is None: min_cost = 0
        max_cost = trigger_filter.get("max_cost", 999)
        if max_cost is None: max_cost = 999

        if cost_ref:
            descriptions.append("コスト【選択数字】")
        elif exact_cost is not None:
            descriptions.append(f"コスト{exact_cost}")
        else:
            if isinstance(min_cost, dict) and min_cost.get("input_value_usage") == "MIN_COST":
                descriptions.append("コスト【入力値】以上")
            elif isinstance(max_cost, dict) and max_cost.get("input_value_usage") == "MAX_COST":
                descriptions.append("コスト【入力値】以下")
            else:
                if min_cost > 0 and max_cost < 999:
                    descriptions.append(f"コスト{min_cost}～{max_cost}")
                elif min_cost > 0:
                    descriptions.append(f"コスト{min_cost}以上")
                elif max_cost < 999:
                    descriptions.append(f"コスト{max_cost}以下")

        min_power = trigger_filter.get("min_power", 0)
        if min_power is None: min_power = 0
        max_power = trigger_filter.get("max_power", 999999)
        if max_power is None: max_power = 999999
        power_max_ref = trigger_filter.get("power_max_ref")

        if power_max_ref:
            descriptions.append("パワー【入力値】以下")
        else:
            if min_power > 0 and max_power < 999999:
                descriptions.append(f"パワー{min_power}～{max_power}")
            elif min_power > 0:
                descriptions.append(f"パワー{min_power}以上")
            elif max_power < 999999:
                descriptions.append(f"パワー{max_power}以下")

        is_tapped = trigger_filter.get("is_tapped")
        if is_tapped == 1: descriptions.append("(タップ状態)")
        elif is_tapped == 0: descriptions.append("(アンタップ状態)")

        is_blocker = trigger_filter.get("is_blocker")
        if is_blocker == 1: descriptions.append("(ブロッカー)")
        elif is_blocker == 0: descriptions.append("(ブロッカー以外)")

        is_evolution = trigger_filter.get("is_evolution")
        if is_evolution == 1: descriptions.append("(進化)")
        elif is_evolution == 0: descriptions.append("(進化以外)")

        is_summoning_sick = trigger_filter.get("is_summoning_sick")
        if is_summoning_sick == 1: descriptions.append("(召喚酔い)")
        elif is_summoning_sick == 0: descriptions.append("(召喚酔い解除)")

        zones = trigger_filter.get("zones", [])
        if zones and "BATTLE_ZONE" not in zones:
            zone_names = []
            for z in zones:
                zone_text = cls._normalize_zone_name(z)
                if zone_text:
                    zone_names.append(zone_text)
            if zone_names:
                descriptions.append("[" + "/".join(zone_names) + "]")

        return "、".join(descriptions) if descriptions else ""

    # =========================================================================
    # Internal Logic
    # =========================================================================

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        if isinstance(effect, dict):
            effect_type = effect.get("type", "")
            trigger = effect.get("trigger", "NONE")
            if effect_type in ("COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD", "ADD_RESTRICTION"):
                if trigger == "NONE" or trigger not in effect:
                    return cls._format_modifier(effect, sample=sample)

        trigger = effect.get("trigger", "NONE")
        trigger_scope = effect.get("trigger_scope", "NONE")
        condition = effect.get("condition", {})
        if condition is None:
            condition = {}

        trigger_text = cls.trigger_to_japanese(trigger, is_spell)

        if trigger_scope and trigger_scope != "NONE" and trigger != "PASSIVE_CONST":
            trigger_text = cls._apply_trigger_scope(trigger_text, trigger_scope, trigger, effect.get("trigger_filter", {}))

        cond_text = cls._format_condition(condition)
        cond_type = condition.get("type", "NONE")

        if trigger != "NONE" and trigger != "PASSIVE_CONST":
            if cond_type == "DURING_YOUR_TURN" or cond_type == "DURING_OPPONENT_TURN":
                base_cond = cond_text.strip("、: ")
                trigger_text = f"{base_cond}、{trigger_text}"
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"相手がカードを引いた時、{val}枚目以降なら"
                cond_text = ""

        action_texts = []
        raw_items = []

        commands = effect.get("commands", [])
        for command in commands:
            raw_items.append(command)
            action_texts.append(cls._format_command(command, is_spell, sample=sample, card_mega_last_burst=card_mega_last_burst))

        full_action_text = cls._merge_action_texts(raw_items, action_texts)

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
    def _format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
        from dm_toolkit.consts import TargetScope

        mtype = modifier.get("type", "NONE")
        condition = modifier.get("condition", {})
        filter_def = modifier.get("filter", {})
        value = modifier.get("value", 0)

        keyword = modifier.get("mutation_kind", "") or modifier.get("str_val", "")

        scope = modifier.get("scope", TargetScope.ALL)
        scope = TargetScope.normalize(scope)

        cond_text = cls._format_condition(condition)
        if cond_text and not cond_text.endswith("、"):
            cond_text += "、"

        scope_prefix = cls._get_scope_prefix(scope)

        effective_filter = filter_def.copy() if filter_def else {}
        if scope and scope != TargetScope.ALL:
            effective_filter["owner"] = scope

        target_str = cls._format_modifier_target(effective_filter) if effective_filter else "対象"

        full_target = scope_prefix + target_str if scope_prefix else target_str

        if mtype == "COST_MODIFIER":
            return cls._format_cost_modifier(cond_text, full_target, value)
        elif mtype == "POWER_MODIFIER":
            return cls._format_power_modifier(cond_text, full_target, value)
        elif mtype == "GRANT_KEYWORD":
            return cls._format_grant_keyword(cond_text, full_target, modifier)
        elif mtype == "SET_KEYWORD":
            return cls._format_set_keyword(cond_text, full_target, keyword)
        elif mtype == "ADD_RESTRICTION":
            restriction_text = CardTextResources.get_keyword_text(keyword)
            return f"{cond_text}{scope_prefix}{restriction_text}を与える。"
        else:
            return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"

    @classmethod
    def _format_command(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        if not command:
            return ""

        cmd_type = command.get("type") or command.get("name") or "NONE"
        command_copy = command.copy()

        original_cmd_type = cmd_type
        if cmd_type == "POWER_MOD": cmd_type = "MODIFY_POWER"
        elif cmd_type == "ADD_KEYWORD":
            cmd_type = "ADD_KEYWORD"
            if not command_copy.get("str_val") and command_copy.get("mutation_kind"):
                command_copy["str_val"] = command_copy["mutation_kind"]
        elif cmd_type == "MANA_CHARGE": cmd_type = "SEND_TO_MANA"
        elif cmd_type == "CHOICE": cmd_type = "SELECT_OPTION"

        input_value_key = command.get("input_value_key") or command.get("input_link") or ""
        input_value_usage = command.get("input_value_usage") or command.get("input_usage") or ""

        action_proxy = {
            "type": cmd_type,
            "scope": command_copy.get("target_group", "NONE"),
            "filter": command_copy.get("target_filter") or command_copy.get("filter", {}),
            "value1": command_copy.get("amount") if command_copy.get("amount") is not None else command_copy.get("value1", 0),
            "value2": command_copy.get("val2") or command_copy.get("value2", 0),
            "optional": command_copy.get("optional", False),
            "up_to": command_copy.get("up_to", False),
            "str_val": command_copy.get("str_param") or command_copy.get("str_val", ""),
            "input_value_key": input_value_key,
            "input_value_usage": input_value_usage,
            "from_zone": command_copy.get("from_zone", ""),
            "to_zone": command_copy.get("to_zone", ""),
            "original_to_zone": command_copy.get("original_to_zone", ""),
            "mutation_kind": command_copy.get("mutation_kind", ""),
            "destination_zone": command_copy.get("to_zone") or command_copy.get("destination_zone", ""),
            "result": command_copy.get("result") or command_copy.get("str_param", ""),
            "is_mega_last_burst": card_mega_last_burst,
            "duration": command_copy.get("duration", "")
        }

        if "options" in command_copy: action_proxy["options"] = command_copy.get("options")
        if "flags" in command_copy: action_proxy["flags"] = command_copy.get("flags")
        if "look_count" in command_copy: action_proxy["look_count"] = command_copy.get("look_count")
        if "add_count" in command_copy: action_proxy["add_count"] = command_copy.get("add_count")
        if "rest_zone" in command_copy: action_proxy["rest_zone"] = command_copy.get("rest_zone")
        if "max_cost" in command_copy: action_proxy["max_cost"] = command_copy.get("max_cost")
        if "token_id" in command_copy: action_proxy["token_id"] = command_copy.get("token_id")
        if "play_flags" in command_copy: action_proxy["play_flags"] = command_copy.get("play_flags")
        if "select_count" in command_copy: action_proxy["select_count"] = command_copy.get("select_count")

        if "if_true" in command_copy: action_proxy["if_true"] = command_copy.get("if_true")
        if "if_false" in command_copy: action_proxy["if_false"] = command_copy.get("if_false")
        if "condition" in command_copy: action_proxy["condition"] = command_copy.get("condition")
        if cmd_type == "IF" and "target_filter" in command_copy:
            if "condition" not in action_proxy or not action_proxy["condition"]:
                action_proxy["target_filter"] = command_copy.get("target_filter")

        action_proxy["source_zone"] = command_copy.get("from_zone") or command_copy.get("source_zone", "")

        if original_cmd_type == "MANA_CHARGE":
            if action_proxy["scope"] == "NONE":
                 action_proxy["type"] = "ADD_MANA"
            else:
                 action_proxy["type"] = "SEND_TO_MANA"

        if original_cmd_type == "MEASURE_COUNT": action_proxy["type"] = "COUNT_CARDS"
        if original_cmd_type == "SHIELD_TRIGGER": return "S・トリガー"

        if original_cmd_type == "QUERY":
            query_mode = command_copy.get("str_param") or command_copy.get("query_mode") or ""
            action_proxy["query_mode"] = query_mode
            if query_mode and query_mode != "CARDS_MATCHING_FILTER":
                action_proxy["str_param"] = query_mode
                action_proxy["str_val"] = query_mode

        if original_cmd_type == "LOOK_AND_ADD":
            if "look_count" in command_copy and command_copy.get("look_count") is not None:
                action_proxy["value1"] = command_copy.get("look_count")
            if "add_count" in command_copy and command_copy.get("add_count") is not None:
                action_proxy["value2"] = command_copy.get("add_count")
            rz = command_copy.get("rest_zone") or command_copy.get("destination_zone") or command_copy.get("to_zone")
            if rz:
                action_proxy["rest_zone"] = rz
                action_proxy["destination_zone"] = rz
        elif original_cmd_type == "MEKRAID":
            max_cost_src = command_copy.get("max_cost")
            if max_cost_src is None and "target_filter" in command_copy:
                max_cost_src = command_copy.get("target_filter", {}).get("max_cost")
            if max_cost_src is not None and not isinstance(max_cost_src, dict):
                action_proxy["value1"] = max_cost_src
            if "look_count" in command_copy: action_proxy["look_count"] = command_copy.get("look_count")
            if "rest_zone" in command_copy: action_proxy["rest_zone"] = command_copy.get("rest_zone")
        elif original_cmd_type == "SUMMON_TOKEN":
            if "token_id" in command_copy and command_copy.get("token_id") is not None:
                action_proxy["str_val"] = command_copy.get("token_id")
        elif original_cmd_type == "PLAY_FROM_ZONE":
            if not action_proxy["source_zone"]:
                 action_proxy["source_zone"] = command_copy.get("from_zone", "")
            max_cost = command_copy.get("max_cost")
            if max_cost is None and "target_filter" in command_copy:
                max_cost = command_copy.get("target_filter", {}).get("max_cost")
            if max_cost is not None and not isinstance(max_cost, dict):
                action_proxy["value1"] = max_cost
        elif original_cmd_type == "SELECT_NUMBER" or original_cmd_type == "DECLARE_NUMBER":
            action_proxy["value1"] = command_copy.get("min_value", 1)
            action_proxy["value2"] = command_copy.get("amount", 6)
        elif original_cmd_type == "CHOICE":
            flags = command_copy.get("flags", []) or []
            if isinstance(flags, list) and "ALLOW_DUPLICATES" in flags:
                action_proxy["optional"] = True
            action_proxy["value1"] = command_copy.get("amount", 1)
        elif original_cmd_type == "REGISTER_DELAYED_EFFECT":
            action_proxy["str_val"] = command_copy.get("str_param") or command_copy.get("str_val", "")
        elif original_cmd_type == "COST_REFERENCE":
             action_proxy["ref_mode"] = command_copy.get("ref_mode")

        return cls._format_action(action_proxy, is_spell, sample=sample, card_mega_last_burst=card_mega_last_burst)

    @classmethod
    def _format_action(cls, action: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = CardTextResources.ACTION_MAP.get(atype, "")

        if atype == 'DRAW_CARD':
            if bool(action.get('up_to', False)):
                template = "最大{value1}枚まで引く。"

        if atype == 'TRANSITION':
            from_zone = cls._normalize_zone_name(action.get('from_zone') or action.get('fromZone') or '')
            to_zone = cls._normalize_zone_name(action.get('to_zone') or action.get('toZone') or '')
            amt = action.get('amount') or action.get('value1') or 0
            up_to = bool(action.get('up_to', False))

            alias = CardTextResources.TRANSITION_ALIASES.get((from_zone, to_zone))
            if alias:
                 if alias == "破壊":
                      return f"{{target}}を{amt}体破壊する。" if amt > 0 else f"{{target}}をすべて破壊する。"
                 elif alias == "捨てる":
                      return f"手札を{amt}枚捨てる。" if amt > 0 else "手札をすべて捨てる。"
                 elif alias == "手札に戻す":
                      target_str, unit = cls._resolve_target(action, is_spell)
                      if up_to and amt > 0:
                          t = f"{target_str}を{amt}{unit}まで選び、手札に戻す。"
                      elif amt == 0:
                          t = f"{target_str}をすべて手札に戻す。"
                      else:
                          t = f"{target_str}を{amt}{unit}手札に戻す。"

                      if bool(action.get("optional", False)):
                          if t.endswith("す。"):
                              t = t[:-2] + "してもよい。"
                          else:
                              t = t[:-1] + "てもよい。"
                      return t
                 elif alias == "マナチャージ":
                      return f"自分の山札の上から{amt}枚をマナゾーンに置く。"
                 elif alias == "シールド焼却":
                      return f"相手のシールドを{amt}つ選び、墓地に置く。"

            if (from_zone == 'DECK' or from_zone == '') and to_zone == 'HAND':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                if up_to:
                    return f"山札からカードを最大{amt}枚まで手札に加える。"
                else:
                    return f"カードを{amt}枚引く。"
            if (from_zone == 'DECK' or from_zone == '') and to_zone == 'MANA_ZONE':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                return f"自分の山札の上から{amt}枚をマナゾーンに置く。"

        optional = action.get("optional", False)
        target_str, unit = cls._resolve_target(action, is_spell)

        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")
        input_key = action.get("input_value_key", "")
        input_usage = action.get("input_value_usage") or action.get("input_usage")

        is_generic_selection = atype in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA", "MOVE_CARD", "TRANSITION", "DISCARD"]

        if input_key:
            usage_label_suffix = ""
            if input_usage:
                label = cls._format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"

            if atype == "DRAW_CARD":
                up_to_flag = bool(action.get('up_to', False))
                template = f"カードをその同じ枚数引く。{usage_label_suffix}"
                if up_to_flag:
                    template = f"カードをその同じ枚数まで引く。{usage_label_suffix}"
            elif atype == "DESTROY":
                up_to_flag = bool(action.get('up_to', False))
                if up_to_flag:
                    template = f"{{target}}をその同じ数だけまで選び、破壊する。{usage_label_suffix}"
                else:
                    template = f"{{target}}をその同じ数だけ破壊する。{usage_label_suffix}"
            elif atype == "TAP":
                up_to_flag = bool(action.get('up_to', False))
                if up_to_flag:
                    template = f"{{target}}をその同じ数だけまで選び、タップする。{usage_label_suffix}"
                else:
                    template = f"{{target}}をその同じ数だけ選び、タップする。{usage_label_suffix}"
            elif atype == "UNTAP":
                up_to_flag = bool(action.get('up_to', False))
                if up_to_flag:
                    template = f"{{target}}をその同じ数だけまで選び、アンタップする。{usage_label_suffix}"
                else:
                    template = f"{{target}}をその同じ数だけ選び、アンタップする。{usage_label_suffix}"
            elif atype == "RETURN_TO_HAND":
                up_to_flag = bool(action.get('up_to', False))
                if up_to_flag:
                    template = f"{{target}}をその同じ数だけまで選び、手札に戻す。{usage_label_suffix}"
                else:
                    template = f"{{target}}をその同じ数だけ選び、手札に戻す。{usage_label_suffix}"
            elif atype == "SEND_TO_MANA":
                up_to_flag = bool(action.get('up_to', False))
                if up_to_flag:
                    template = f"{{target}}をその同じ数だけまで選び、マナゾーンに置く。{usage_label_suffix}"
                else:
                    template = f"{{target}}をその同じ数だけ選び、マナゾーンに置く。{usage_label_suffix}"
            elif atype == "TRANSITION":
                val1 = "その同じ枚数"
                if bool(action.get('up_to', False)):
                    val1 = "その同じ枚数まで"
            elif atype == "MOVE_CARD":
                dest_zone = action.get("destination_zone", "")
                up_to_flag = bool(action.get('up_to', False))
                if dest_zone == "DECK_BOTTOM":
                    if up_to_flag:
                        template = f"{{target}}をその同じ数だけまで選び、山札の下に置く。{usage_label_suffix}"
                    else:
                        template = f"{{target}}をその同じ数だけ選び、山札の下に置く。{usage_label_suffix}"
                elif dest_zone == "GRAVEYARD":
                    if up_to_flag:
                        template = f"{{target}}をその同じ数だけまで選び、墓地に置く。{usage_label_suffix}"
                    else:
                        template = f"{{target}}をその同じ数だけ選び、墓地に置く。{usage_label_suffix}"
                elif dest_zone == "HAND":
                    if up_to_flag:
                        template = f"{{target}}をその同じ数だけまで選び、手札に戻す。{usage_label_suffix}"
                    else:
                        template = f"{{target}}をその同じ数だけ選び、手札に戻す。{usage_label_suffix}"
                elif dest_zone == "MANA_ZONE":
                    if up_to_flag:
                        template = f"{{target}}をその同じ数だけまで選び、マナゾーンに置く。{usage_label_suffix}"
                    else:
                        template = f"{{target}}をその同じ数だけ選び、マナゾーンに置く。{usage_label_suffix}"
            elif atype == "DISCARD":
                up_to_discard = bool(action.get('up_to', False))
                if up_to_discard:
                    template = f"手札をその同じ枚数まで捨てる。{usage_label_suffix}"
                else:
                    template = f"手札をその同じ枚数捨てる。{usage_label_suffix}"
            else:
                val1 = "その数"
        elif (val1 == 0 or (atype == "TRANSITION" and action.get("amount", 0) == 0)) and is_generic_selection:
             if atype == "DESTROY": template = "{target}をすべて破壊する。"
             elif atype == "TAP": template = "{target}をすべてタップする。"
             elif atype == "UNTAP": template = "{target}をすべてアンタップする。"
             elif atype == "RETURN_TO_HAND": template = "{target}をすべて手札に戻す。"
             elif atype == "SEND_TO_MANA": template = "{target}をすべてマナゾーンに置く。"
             elif atype == "MOVE_CARD": pass
             elif atype == "TRANSITION": pass
             elif atype == "DISCARD": template = "手札をすべて捨てる。"

        if atype == "DISCARD":
            amt = action.get('amount', val1 if val1 else 1)
            up_to_discard = bool(action.get('up_to', False))
            if amt == 0:
                template = "手札をすべて捨てる。"
            elif up_to_discard:
                template = f"手札を{amt}枚まで捨てる。"
            else:
                template = f"手札を{amt}枚捨てる。"
            return template

        elif atype == "MEKRAID" or atype == "FRIEND_BURST" or atype == "APPLY_MODIFIER" or atype == "ADD_KEYWORD" or atype == "MUTATE" or atype == "REGISTER_DELAYED_EFFECT" or atype == "SUMMON_TOKEN":
             text = cls._format_special_effect_command(atype, action, is_spell, val1, target_str, unit)
             if text: return text

        elif atype == "TRANSITION" or atype == "MOVE_CARD" or atype == "REPLACE_CARD_MOVE":
             t = cls._format_zone_move_command(atype, action, is_spell, val1, target_str)
             if t:
                 template = t

        elif atype == "IF" or atype == "IF_ELSE" or atype == "ELSE":
            text = cls._format_logic_command(atype, action, is_spell, sample, card_mega_last_burst)
            if text: return text

        text = cls._format_game_action_command(atype, action, is_spell, val1, val2, target_str, unit, input_key, input_usage, sample)
        if text: return text

        buf = cls._format_buffer_command(atype, action, is_spell, val1)
        if buf:
            template = buf

        if not template:
            return f"({tr(atype)})"

        if atype == "GRANT_KEYWORD" or atype == "ADD_KEYWORD":
            keyword = CardTextResources.get_keyword_text(str_val)
            str_val = keyword

        elif atype == "CAST_SPELL":
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter

            is_mega_last_burst = action.get("is_mega_last_burst", False) or action.get("mega_last_burst", False)
            mega_burst_prefix = ""
            if is_mega_last_burst:
                mega_burst_prefix = "このクリーチャーがバトルゾーンから離れて、"

            usage_label_suffix = ""
            if input_key and input_usage:
                label = cls._format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"

            types = temp_filter.get("types", [])
            if "SPELL" in types or not types:
                zones = temp_filter.get("zones", [])
                zone_phrase = ""
                if zones:
                    zone_names = []
                    for z in zones:
                        if z == "HAND": zone_names.append("手札")
                        elif z == "GRAVEYARD": zone_names.append("墓地")
                        elif z == "MANA_ZONE": zone_names.append("マナゾーン")
                        elif z == "BATTLE_ZONE": zone_names.append("バトルゾーン")
                        else: zone_names.append(tr(z))
                    if len(zone_names) == 1:
                        zone_phrase = zone_names[0] + "から"
                    else:
                        zone_phrase = "または".join(zone_names) + "から"

                tf_no_zones = temp_filter.copy()
                if "zones" in tf_no_zones:
                    tf_no_zones["zones"] = []
                action_no_zone = action.copy()
                action_no_zone["filter"] = tf_no_zones
                target_str, unit = cls._resolve_target(action_no_zone)

                if target_str.endswith("呪文"):
                    template = f"{mega_burst_prefix}{zone_phrase}{target_str}をコストを支払わずに唱える。{usage_label_suffix}" if zone_phrase else f"{mega_burst_prefix}{target_str}をコストを支払わずに唱える。{usage_label_suffix}"
                elif target_str == "カード" or target_str == "":
                    template = f"{mega_burst_prefix}{zone_phrase}呪文をコストを支払わずに唱える。{usage_label_suffix}" if zone_phrase else f"{mega_burst_prefix}呪文をコストを支払わずに唱える。{usage_label_suffix}"
                else:
                    template = f"{mega_burst_prefix}{zone_phrase}{target_str}の呪文をコストを支払わずに唱える。{usage_label_suffix}" if zone_phrase else f"{mega_burst_prefix}{target_str}の呪文をコストを支払わずに唱える。{usage_label_suffix}"
            else:
                target_str, unit = cls._resolve_target(action)
                if target_str == "" or target_str == "カード":
                    template = f"{mega_burst_prefix}カードをコストを支払わずに唱える。{usage_label_suffix}"
                else:
                    template = f"{mega_burst_prefix}{target_str}をコストを支払わずに唱える。{usage_label_suffix}"

        elif atype == "PLAY_FROM_ZONE":
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter

            usage_label_suffix = ""
            if input_key and input_usage:
                label = cls._format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"

            if not action.get("source_zone") and "zones" in temp_filter:
                zones = temp_filter["zones"]
                if len(zones) == 1:
                    action["source_zone"] = zones[0]

            if action.get("value1", 0) == 0:
                max_cost = temp_filter.get("max_cost", 999)
                if isinstance(max_cost, dict):
                    pass
                elif max_cost < 999:
                    action["value1"] = max_cost
                    if not input_key: val1 = max_cost
                    if "max_cost" in temp_filter: del temp_filter["max_cost"]
            else:
                 if "max_cost" in temp_filter:
                      del temp_filter["max_cost"]

            if "zones" in temp_filter: temp_filter["zones"] = []
            scope = action.get("scope", "NONE")
            if scope in ["PLAYER_SELF", "SELF"]: action["scope"] = "NONE"

            target_str, unit = cls._resolve_target(action)
            verb = "プレイする"
            types = temp_filter.get("types", [])
            if "SPELL" in types and "CREATURE" not in types:
                verb = "唱える"
            elif "CREATURE" in types:
                verb = "召喚する"

            play_flags = action.get("play_flags")
            is_free = False
            if isinstance(play_flags, bool) and play_flags:
                is_free = True
            elif isinstance(play_flags, list) and ("FREE" in play_flags or "COST_FREE" in play_flags):
                is_free = True

            if is_free:
                verb = f"コストを支払わずに{verb}"

            use_linked_cost = False
            max_cost = temp_filter.get("max_cost")
            if isinstance(max_cost, dict) and max_cost.get("input_value_usage") == "MAX_COST":
                use_linked_cost = True

            if use_linked_cost:
                if action.get("source_zone"):
                    template = "{source_zone}から{target}を" + verb + f"。{usage_label_suffix}"
                else:
                    template = "{target}を" + verb + f"。{usage_label_suffix}"
            else:
                if action.get("source_zone"):
                    template = "{source_zone}からコスト{value1}以下の{target}を" + verb + f"。{usage_label_suffix}"
                else:
                    template = "コスト{value1}以下の{target}を" + verb + f"。{usage_label_suffix}"

        if atype == "DESTROY" and action.get('filter', {}).get('is_trigger_source'):
            template = "{target}を破壊する。"

        dest_zone = action.get("destination_zone", "")
        zone_str = cls._zone_to_japanese(dest_zone) if dest_zone else "どこか"
        src_zone = action.get("source_zone", "")
        src_str = cls._zone_to_japanese(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        if atype == "PLAY_FROM_ZONE" and action.get("value1") == 0:
            max_cost = action.get("filter", {}).get("max_cost")
            if isinstance(max_cost, dict) and max_cost.get("input_value_usage") == "MAX_COST":
                text = text.replace("コスト0以下の", "")

        if atype in ["TRANSITION", "MUTATE"]:
            text = text.replace("{amount}", str(val1))

        if "{filter}" in text:
             text = text.replace("{filter}", target_str)

        if "{result}" in text:
             res = action.get("result", "")
             text = text.replace("{result}", tr(res))

        if atype == "COST_REDUCTION":
            if target_str == "カード" or target_str == "自分のカード":
                replacement = "この呪文" if is_spell else "このクリーチャー"
                text = text.replace("カード", replacement)
                text = text.replace("自分のカード", replacement)
            cond = action.get("condition", {})
            if cond:
                cond_text = cls._format_condition(cond)
                text = f"{cond_text}{text}"

        if optional:
            if text.endswith("する。"):
                text = text[:-3] + "してもよい。"
            elif text.endswith("く。"):
                text = text[:-2] + "いてもよい。"
            elif text.endswith("す。"):
                text = text[:-2] + "してもよい。"
            elif text.endswith("る。"):
                text = text[:-2] + "てもよい。"
            elif text.endswith("う。"):
                text = text[:-2] + "ってもよい。"
            else:
                if not text.endswith("てもよい。"):
                    text = text[:-1] + "てもよい。"

        return text

    # =========================================================================
    # Helpers
    # =========================================================================

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False) -> str:
        return CardTextResources.get_trigger_text(trigger, is_spell=is_spell)

    @classmethod
    def _format_input_usage_label(cls, usage: Any) -> str:
        if usage is None:
            return ""
        norm = str(usage).upper()
        if norm == "MAX_COST":
            return ""
        if norm in CardTextResources.INPUT_USAGE_LABELS:
            return CardTextResources.INPUT_USAGE_LABELS[norm]
        return tr(str(usage)) if str(usage) else ""

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False) -> Tuple[str, str]:
        scope = action.get("scope", action.get('target_group', "NONE"))
        filter_def = action.get("filter", action.get('target_filter', {}))
        atype = action.get("type", "")

        if filter_def.get('is_trigger_source'):
            types = filter_def.get('types', [])
            if 'SPELL' in types:
                return ("その呪文", "枚")
            elif 'CARD' in types:
                return ("そのカード", "枚")
            return ("そのクリーチャー", "体")

        target_desc = ""
        prefix = ""
        adjectives = ""
        zone_noun = ""
        type_noun = "カード"
        unit = "枚"

        if atype == "DISCARD" and scope == "NONE":
             scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
             target = "この呪文" if is_spell else "このクリーチャー"
             return (target, "枚")

        prefix = CardTextResources.get_scope_text(scope)
        if not prefix:
            if scope == "ALL_PLAYERS": prefix = "すべてのプレイヤーの"
            elif scope == "RANDOM": prefix = "ランダムな"

        if prefix in ["自分", "相手"]:
            prefix += "の"

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])
            owner = filter_def.get("owner", "NONE")

            if not civs and "civilization" in filter_def:
                single = filter_def.get("civilization")
                if single: civs = [single]

            if not prefix and owner != "NONE":
                 owner_text = CardTextResources.get_scope_text(owner)
                 if owner_text:
                     prefix = owner_text

            temp_adjs = []
            if civs: temp_adjs.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))
            if races: temp_adjs.append("/".join(races))

            if temp_adjs: adjectives += "/".join(temp_adjs) + "の"

            min_cost = filter_def.get("min_cost", 0)
            if min_cost is None: min_cost = 0
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            has_input_key = bool(action.get("input_value_key"))
            if isinstance(min_cost, dict):
                usage = min_cost.get("input_value_usage", "")
                if usage == "MIN_COST":
                    adjectives += "コストその数以上の"
            elif min_cost > 0:
                adjectives += f"コスト{min_cost}以上の"
            elif has_input_key and input_usage == "MIN_COST":
                adjectives += "コストその数以上の"

            max_cost = filter_def.get("max_cost", 999)
            if max_cost is None: max_cost = 999
            if isinstance(max_cost, dict):
                usage = max_cost.get("input_value_usage", "")
                if usage == "MAX_COST":
                    adjectives += "コストその数以下の"
            elif max_cost < 999:
                adjectives += f"コスト{max_cost}以下の"
            elif has_input_key and input_usage == "MAX_COST":
                adjectives += "コストその数以下の"

            min_power = filter_def.get("min_power", 0)
            if min_power is None: min_power = 0
            if isinstance(min_power, dict):
                usage = min_power.get("input_value_usage", "")
                if usage == "MIN_POWER":
                    adjectives += "パワーその数以上の"
            elif min_power > 0:
                adjectives += f"パワー{min_power}以上の"
            elif has_input_key and input_usage == "MIN_POWER":
                adjectives += "パワーその数以上の"

            max_power = filter_def.get("max_power", 999999)
            if max_power is None: max_power = 999999
            if isinstance(max_power, dict):
                usage = max_power.get("input_value_usage", "")
                if usage == "MAX_POWER":
                    adjectives += "パワーその数以下の"
            elif max_power < 999999:
                adjectives += f"パワー{max_power}以下の"
            elif has_input_key and input_usage == "MAX_POWER":
                adjectives += "パワーその数以下の"

            if filter_def.get("is_tapped", None) is True: adjectives = "タップされている" + adjectives
            elif filter_def.get("is_tapped", None) is False: adjectives = "アンタップされている" + adjectives
            if filter_def.get("is_blocker", None) is True: adjectives = "ブロッカーを持つ" + adjectives
            if filter_def.get("is_evolution", None) is True: adjectives = "進化" + adjectives

            if "BATTLE_ZONE" in zones: zone_noun = "バトルゾーン"
            elif "MANA_ZONE" in zones: zone_noun = "マナゾーン"
            elif "HAND" in zones: zone_noun = "手札"
            elif "SHIELD_ZONE" in zones: zone_noun = "シールドゾーン"
            elif "GRAVEYARD" in zones: zone_noun = "墓地"
            elif "DECK" in zones: zone_noun = "山札"

            if "ELEMENT" in types:
                type_noun = "エレメント"
                unit = "体"
            elif "CREATURE" in types:
                type_noun = "クリーチャー"
                unit = "体"
            elif "SPELL" in types:
                type_noun = "呪文"
            elif "CROSS_GEAR" in types:
                type_noun = "クロスギア"
            elif "CARD" in types:
                type_noun = "カード"
                unit = "枚"
            elif len(types) > 1:
                 type_words = []
                 if "CREATURE" in types: type_words.append("クリーチャー")
                 if "SPELL" in types: type_words.append("呪文")
                 if "ELEMENT" in types: type_words.append("エレメント")
                 if "CROSS_GEAR" in types: type_words.append("クロスギア")
                 if type_words: type_noun = "/".join(type_words)

            if "BATTLE_ZONE" in zones:
                if "CREATURE" in types or not types:
                    type_noun = "クリーチャー"
                    unit = "体"
                if "ELEMENT" in types:
                    type_noun = "エレメント"
                    unit = "枚"
            elif "SHIELD_ZONE" in zones:
                type_noun = "カード"
                unit = "つ"
            elif "GRAVEYARD" in zones:
                if "CREATURE" in types:
                     type_noun = "クリーチャー"
                     unit = "体"

            if atype == "SEARCH_DECK":
                 zone_noun = ""

            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "の")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)
            target_desc = "".join(parts)

            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"

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

    @classmethod
    def _merge_action_texts(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        if not formatted_texts:
            return ""

        def is_draw_item(it):
            if not isinstance(it, dict): return False
            t = it.get('type', '')
            if t == 'DRAW_CARD': return True
            if t == 'TRANSITION':
                from_z = (it.get('from_zone') or it.get('fromZone') or '').upper()
                to_z = (it.get('to_zone') or it.get('toZone') or '').upper()
                if (from_z == '' or 'DECK' in from_z) and 'HAND' in to_z: return True
            return False

        def is_deck_bottom_move(it):
            if not isinstance(it, dict): return False
            dest = (it.get('destination_zone') or it.get('to_zone') or it.get('toZone') or '').upper()
            if 'DECK_BOTTOM' in dest or 'DECKBOTTOM' in dest: return True
            t = (it.get('type') or '').upper()
            if 'DECK_BOTTOM' in t: return True
            return False

        def is_cast_spell_item(it):
            if not isinstance(it, dict): return False
            t = it.get('type', '').upper()
            return t == 'CAST_SPELL'

        def is_replace_card_move(it):
            if not isinstance(it, dict): return False
            t = it.get('type', '').upper()
            return t == 'REPLACE_CARD_MOVE'

        if len(raw_items) >= 2 and is_cast_spell_item(raw_items[0]) and is_replace_card_move(raw_items[1]):
            from_zone_key = raw_items[1].get('from_zone', 'GRAVEYARD')
            to_zone_key = raw_items[1].get('to_zone', 'DECK_BOTTOM')
            from_zone_text = tr(from_zone_key)
            to_zone_text = tr(to_zone_key)
            merged = f"その呪文を唱えた後、{from_zone_text}に置くかわりに{to_zone_text}に置く。"
            if len(formatted_texts) > 2:
                rest = ' '.join(formatted_texts[2:]).strip()
                if rest:
                    merged = merged.rstrip('。') + '、' + rest
            return merged

        if len(raw_items) >= 2 and is_draw_item(raw_items[0]) and is_deck_bottom_move(raw_items[1]):
            first = formatted_texts[0].rstrip('。')
            tail = 'その後、引いた枚数と同じ枚数を山札の下に置く。'
            merged = f"{first}。{tail}"
            if len(formatted_texts) > 2:
                rest = ' '.join(formatted_texts[2:]).strip()
                if rest:
                    merged = merged.rstrip('。') + '、' + rest
            return merged

        return ' '.join([t for t in formatted_texts if t]).strip()

    @classmethod
    def _apply_trigger_scope(cls, trigger_text: str, scope: str, trigger_type: str, trigger_filter: Dict[str, Any] = None) -> str:
        if not scope or scope == "NONE" or scope == "ALL":
            return trigger_text

        scope_text = CardTextResources.get_scope_text(scope)
        if not scope_text:
            return trigger_text

        scope_text = scope_text.rstrip("の")

        if "相手が" in trigger_text and (scope == "OPPONENT" or scope == "PLAYER_OPPONENT"):
            return trigger_text
        if "自分が" in trigger_text and (scope == "SELF" or scope == "PLAYER_SELF"):
            return trigger_text

        def _compose_subject_from_filter(default_type: str) -> str:
            f = trigger_filter or {}
            civs = f.get("civilizations", [])
            races = f.get("races", [])
            types = f.get("types", [])
            zones = f.get("zones", [])
            min_cost = f.get("min_cost", 0)
            if min_cost is None: min_cost = 0
            max_cost = f.get("max_cost", 999)
            if max_cost is None: max_cost = 999
            exact_cost = f.get("exact_cost")
            cost_ref = f.get("cost_ref")
            min_power = f.get("min_power", 0)
            if min_power is None: min_power = 0
            max_power = f.get("max_power", 999999)
            if max_power is None: max_power = 999999
            power_max_ref = f.get("power_max_ref")
            is_tapped = f.get("is_tapped")
            is_blocker = f.get("is_blocker")
            is_evolution = f.get("is_evolution")
            is_summoning_sick = f.get("is_summoning_sick")
            flags = f.get("flags", [])

            noun = "クリーチャー" if default_type == "CREATURE" else ("呪文" if default_type == "SPELL" else "カード")
            if types:
                if "ELEMENT" in types: noun = "エレメント"
                elif "SPELL" in types: noun = "呪文"
                elif "CREATURE" in types: noun = "クリーチャー"
                elif "CARD" in types: noun = "カード"

            adjs: List[str] = []

            if zones and "BATTLE_ZONE" not in zones:
                zone_names = []
                for z in zones:
                    zone_text = cls._normalize_zone_name(z)
                    if zone_text:
                        zone_names.append(zone_text)
                if zone_names:
                    adjs.append("/".join(zone_names))

            if civs: adjs.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))
            if races: adjs.append("/".join(races))

            if cost_ref:
                adjs.append("選択した数字と同じコスト")
            elif exact_cost is not None:
                adjs.append(f"コスト{exact_cost}")
            else:
                if isinstance(min_cost, dict) and min_cost.get("input_value_usage") == "MIN_COST":
                    adjs.append("コストその数以上")
                elif isinstance(max_cost, dict) and max_cost.get("input_value_usage") == "MAX_COST":
                    adjs.append("コストその数以下")
                else:
                    if min_cost > 0 and max_cost < 999:
                        adjs.append(f"コスト{min_cost}～{max_cost}")
                    elif min_cost > 0:
                        adjs.append(f"コスト{min_cost}以上")
                    elif max_cost < 999:
                        adjs.append(f"コスト{max_cost}以下")

            if power_max_ref:
                adjs.append("パワーその数以下")
            elif isinstance(min_power, dict) and min_power.get("input_value_usage") == "MIN_POWER":
                adjs.append("パワーその数以上")
            elif isinstance(max_power, dict) and max_power.get("input_value_usage") == "MAX_POWER":
                adjs.append("パワーその数以下")
            else:
                if min_power > 0 and max_power < 999999:
                    adjs.append(f"パワー{min_power}～{max_power}")
                elif min_power > 0:
                    adjs.append(f"パワー{min_power}以上")
                elif max_power < 999999:
                    adjs.append(f"パワー{max_power}以下")

            if is_tapped == 1: adjs.append("タップ状態")
            elif is_tapped == 0: adjs.append("アンタップ状態")
            if is_blocker == 1: adjs.append("ブロッカー")
            elif is_blocker == 0: adjs.append("ブロッカー以外")
            if is_evolution == 1: adjs.append("進化")
            elif is_evolution == 0: adjs.append("進化以外")
            if is_summoning_sick == 1: adjs.append("召喚酔い")
            elif is_summoning_sick == 0: adjs.append("召喚酔い以外")

            if flags:
                for flag in flags:
                    if flag == "BLOCKER":
                        if "ブロッカー" not in adjs:
                            adjs.append("ブロッカー")

            adj_str = "の".join(adjs)
            if adj_str:
                return f"{adj_str}の{noun}"
            return noun

        if trigger_type == "ON_PLAY" and (scope == "OPPONENT" or scope == "PLAYER_OPPONENT"):
            subject = _compose_subject_from_filter("CREATURE")
            return f"{scope_text}の{subject}がバトルゾーンに出た時"

        if trigger_type == "ON_PLAY" and (scope == "SELF" or scope == "PLAYER_SELF"):
            subject = _compose_subject_from_filter("CREATURE")
            return f"{scope_text}の{subject}がバトルゾーンに出た時"

        if trigger_type == "ON_OTHER_ENTER":
            subject = _compose_subject_from_filter("CREATURE")
            if subject:
                subject = "他の" + subject
            return f"{scope_text}の{subject}がバトルゾーンに出た時"

        if trigger_type == "ON_CAST_SPELL":
            subject = _compose_subject_from_filter("SPELL")
            return f"{scope_text}の{subject}を唱えた時"

        if trigger_type == "ON_SHIELD_ADD":
             if "シールドゾーン" in trigger_text:
                 return trigger_text.replace("シールドゾーン", f"{scope_text}のシールドゾーン")

        if trigger_text.startswith("この"):
             return f"{scope_text}の{trigger_text}"

        return f"{scope_text}の{trigger_text}"

    @classmethod
    def _zone_to_japanese(cls, zone: str) -> str:
        z = cls._normalize_zone_name(zone)
        jp = {
            "BATTLE_ZONE": "バトルゾーン",
            "MANA_ZONE": "マナゾーン",
            "SHIELD_ZONE": "シールドゾーン",
            "HAND": "手札",
            "GRAVEYARD": "墓地",
            "DECK": "山札",
            "DECK_TOP": "山札の上",
            "DECK_BOTTOM": "山札の下",
            "BUFFER": "バッファ",
            "UNDER_CARD": "下",
        }
        return jp.get(z, tr(z))

    @classmethod
    def _normalize_zone_name(cls, zone: str) -> str:
        if not zone:
            return ""
        z = str(zone).split(".")[-1].upper()
        zone_map = {
            "BATTLE": "BATTLE_ZONE",
            "MANA": "MANA_ZONE",
            "SHIELD": "SHIELD_ZONE",
            "BATTLE_ZONE": "BATTLE_ZONE",
            "MANA_ZONE": "MANA_ZONE",
            "SHIELD_ZONE": "SHIELD_ZONE",
            "HAND": "HAND",
            "GRAVEYARD": "GRAVEYARD",
            "DECK": "DECK",
            "DECK_TOP": "DECK_TOP",
            "DECK_BOTTOM": "DECK_BOTTOM",
            "BUFFER": "BUFFER",
            "UNDER_CARD": "UNDER_CARD",
        }
        return zone_map.get(z, z)

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
             stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ""))

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
            return CardTextResources.get_condition_text("DURING_YOUR_TURN")
        elif cond_type == "DURING_OPPONENT_TURN":
            return CardTextResources.get_condition_text("DURING_OPPONENT_TURN")
        elif cond_type == "EVENT_FILTER_MATCH":
            return ""

        return ""

    @classmethod
    def _format_cost_modifier(cls, cond: str, target: str, value: int) -> str:
        if value > 0:
            return f"{cond}{target}のコストを{value}軽減する。"
        elif value < 0:
            return f"{cond}{target}のコストを{abs(value)}増やす。"
        return f"{cond}{target}のコストを修正する。"

    @classmethod
    def _format_power_modifier(cls, cond: str, target: str, value: int) -> str:
        sign = "+" if value >= 0 else ""
        if value == 0:
            return f"{cond}{target}のパワーは不変。"
        return f"{cond}{target}のパワーを{sign}{value}する。"

    @classmethod
    def _format_grant_keyword(cls, cond: str, target: str, modifier: Dict[str, Any]) -> str:
        str_val = modifier.get('mutation_kind') or modifier.get('str_val', '')

        if not str_val:
            return f"{cond}{target}に能力を与える。"

        keyword = CardTextResources.get_keyword_text(str_val)
        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = ""
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + "、"
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

        amt = modifier.get('value') if modifier.get('value') not in (None, 0) else modifier.get('amount', 0)
        if not isinstance(amt, int) or amt <= 0:
            amt = None

        restriction_keys = [
            'CANNOT_ATTACK', 'CANNOT_BLOCK', 'CANNOT_ATTACK_OR_BLOCK', 'CANNOT_ATTACK_AND_BLOCK'
        ]

        subject = f"{cond}{target}"
        if amt:
            subject_phrase = f"{subject}を{amt}体は、"
        else:
            subject_phrase = f"{subject}を選び、"

        if duration_text and not duration_text.endswith('、'):
            duration_text = duration_text + '、'

        if str_val in restriction_keys or str_val.upper() in restriction_keys:
            return f"{subject_phrase}{duration_text}そのクリーチャーに{keyword}を与える。"

        return f"{subject_phrase}{duration_text}そのクリーチャーに「{keyword}」を与える。"

    @classmethod
    def _format_set_keyword(cls, cond: str, target: str, str_val: str) -> str:
        if str_val:
            keyword = CardTextResources.get_keyword_text(str_val)
            result = f"{cond}{target}は「{keyword}」を得る。"
            return result
        return f"{cond}{target}は能力を得る。"

    @classmethod
    def _format_modifier_target(cls, filter_def: Dict[str, Any]) -> str:
        if not filter_def:
            return "対象"

        zones = filter_def.get("zones", [])
        types = filter_def.get("types", [])
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        owner = filter_def.get("owner", "")
        min_cost = filter_def.get("min_cost", 0)
        if min_cost is None: min_cost = 0
        max_cost = filter_def.get("max_cost", 999)
        if max_cost is None: max_cost = 999
        min_power = filter_def.get("min_power", 0)
        if min_power is None: min_power = 0
        max_power = filter_def.get("max_power", 999999)
        if max_power is None: max_power = 999999
        is_tapped = filter_def.get("is_tapped")
        is_blocker = filter_def.get("is_blocker")
        is_evolution = filter_def.get("is_evolution")

        parts = []

        if zones:
            zone_names = []
            for z in zones:
                if z == "BATTLE_ZONE": zone_names.append("バトルゾーン")
                elif z == "MANA_ZONE": zone_names.append("マナゾーン")
                elif z == "HAND": zone_names.append("手札")
                elif z == "GRAVEYARD": zone_names.append("墓地")
                else: zone_names.append(tr(z))

            if len(zone_names) == 1:
                parts.append(zone_names[0] + "の")
            else:
                parts.append("または".join(zone_names) + "から")

        if civs:
            parts.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]) + "の")

        if races:
            parts.append("/".join(races) + "の")

        exact_cost = filter_def.get("exact_cost")
        cost_ref = filter_def.get("cost_ref")

        if cost_ref:
            parts.append("選択した数字と同じコストの")
        elif exact_cost is not None:
            parts.append(f"コスト{exact_cost}の")
        elif isinstance(min_cost, dict):
            usage = min_cost.get("input_value_usage", "")
            if usage == "MIN_COST":
                parts.append("コストその数以上の")
        elif isinstance(max_cost, dict):
            usage = max_cost.get("input_value_usage", "")
            if usage == "MAX_COST":
                parts.append("コストその数以下の")
        else:
            if min_cost > 0 and max_cost < 999:
                parts.append(f"コスト{min_cost}～{max_cost}の")
            elif min_cost > 0:
                parts.append(f"コスト{min_cost}以上の")
            elif max_cost < 999:
                parts.append(f"コスト{max_cost}以下の")

        if isinstance(min_power, dict):
            usage = min_power.get("input_value_usage", "")
            if usage == "MIN_POWER":
                parts.append("パワーその数以上の")
        elif isinstance(max_power, dict):
            usage = max_power.get("input_value_usage", "")
            if usage == "MAX_POWER":
                parts.append("パワーその数以下の")
        else:
            if min_power > 0 and max_power < 999999:
                parts.append(f"パワー{min_power}～{max_power}の")
            elif min_power > 0:
                parts.append(f"パワー{min_power}以上の")
            elif max_power < 999999:
                parts.append(f"パワー{max_power}以下の")

        type_noun = "カード"
        if types:
            if len(types) == 1:
                if types[0] == "CREATURE": type_noun = "クリーチャー"
                elif types[0] == "SPELL": type_noun = "呪文"
                elif types[0] == "ELEMENT": type_noun = "エレメント"
            else:
                type_words = []
                if "CREATURE" in types: type_words.append("クリーチャー")
                if "SPELL" in types: type_words.append("呪文")
                if "ELEMENT" in types: type_words.append("エレメント")
                if type_words: type_noun = "/".join(type_words)

        flag_parts = []
        if is_tapped == 1: flag_parts.append("タップ状態の")
        elif is_tapped == 0: flag_parts.append("アンタップ状態の")
        if is_blocker == 1: flag_parts.append("ブロッカーの")
        elif is_blocker == 0: flag_parts.append("ブロッカー以外の")
        if is_evolution == 1: flag_parts.append("進化クリーチャーの")
        elif is_evolution == 0: flag_parts.append("進化以外の")

        if flag_parts:
            parts.extend(flag_parts)

        result = "".join(parts) + type_noun
        result = result.replace("のの", "の").replace("、の", "の")

        return result if result else "対象"

    @classmethod
    def _compute_stat_from_sample(cls, key: str, sample: List[Any]) -> Any:
        if not sample:
            return None

        if key == "MANA_CIVILIZATION_COUNT":
            civs = set()
            for entry in sample:
                if isinstance(entry, str):
                    civs.add(entry)
                elif isinstance(entry, dict):
                    for c in entry.get('civilizations', []):
                        civs.add(c)
            return len(civs)

        count_stats = {
            "MANA_COUNT", "CREATURE_COUNT", "SHIELD_COUNT", "HAND_COUNT",
            "GRAVEYARD_COUNT", "BATTLE_ZONE_COUNT", "OPPONENT_MANA_COUNT",
            "OPPONENT_CREATURE_COUNT", "OPPONENT_SHIELD_COUNT", "OPPONENT_HAND_COUNT",
            "OPPONENT_GRAVEYARD_COUNT", "OPPONENT_BATTLE_ZONE_COUNT", "CARDS_DRAWN_THIS_TURN"
        }
        if key in count_stats:
            return len(sample)

        return None

    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        return CardTextResources.get_scope_text(scope)

    @classmethod
    def _format_buffer_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int) -> str:
        if atype == "LOOK_TO_BUFFER":
             src_zone = tr(action.get("from_zone", "DECK"))
             amt = val1 if val1 > 0 else 1
             return f"{src_zone}から{amt}枚を見る。"

        elif atype == "REVEAL_TO_BUFFER":
             src_zone = tr(action.get("from_zone", "DECK"))
             amt = val1 if val1 > 0 else 1
             return f"{src_zone}から{amt}枚を表向きにしてバッファに置く。"

        elif atype == "SELECT_FROM_BUFFER":
             if val1 == 0:
                 return f"見たカードすべてを選ぶ。"
             amt = val1 if val1 > 0 else 1
             return f"見たカードの中から{amt}枚を選ぶ。"

        elif atype == "PLAY_FROM_BUFFER":
             target_str, unit = cls._resolve_target(action, is_spell)
             return f"選んだカード（{target_str}）を使う。"

        elif atype == "MOVE_BUFFER_TO_ZONE":
             to_zone = tr(action.get("to_zone", "HAND"))
             amt = val1 if val1 > 0 else 1
             return f"選んだカードを{amt}枚{to_zone}に置く。"

        return ""

    @classmethod
    def _format_special_effect_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int, target_str: str, unit: str) -> str:
        if atype == "MEKRAID":
            val2 = action.get("value2", 3)
            select_count = action.get("select_count", 1)
            input_key = action.get("input_value_key", "")
            input_usage = action.get("input_value_usage") or action.get("input_usage")

            use_token = str(val1)
            if input_key and input_usage == "MAX_COST":
                use_token = "その数"
            elif val1 == 0 and input_usage == "MAX_COST":
                use_token = "その数"

            count_str = "1体" if select_count == 1 else f"{select_count}体まで"
            return f"メクレイド{use_token}（自分の山札の上から{val2}枚を見る。その中からコスト{use_token}以下のクリーチャーを{count_str}、コストを支払わずに召喚してもよい。残りを山札の下に好きな順序で置く）"

        elif atype == "FRIEND_BURST":
            str_val = action.get("str_val", "")
            if not str_val:
                races = action.get("filter", {}).get("races", [])
                if races:
                    str_val = races[0]
            return f"＜{str_val}＞のフレンド・バースト（このクリーチャーが出た時、自分の他の{str_val}・クリーチャーを1体タップしてもよい。そうしたら、このクリーチャーの呪文側をバトルゾーンに置いたまま、コストを支払わずに唱える。）"

        elif atype == "APPLY_MODIFIER":
            str_param = action.get('str_param') or action.get('str_val') or action.get('mutation_kind') or ''
            duration_key = action.get('duration') or action.get('input_value_key', '')

            input_key = action.get("input_value_key", "")
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            is_target_linked = bool(input_key) and (not input_usage or input_usage == "TARGET")

            duration_text = ""
            if duration_key:
                trans = CardTextResources.get_duration_text(duration_key)
                if trans and trans != duration_key:
                    duration_text = trans + "、"
                elif duration_key in CardTextResources.DURATION_TRANSLATION:
                    duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

            effect_text = CardTextResources.get_keyword_text(str_param) if str_param else "（効果）"
            if isinstance(effect_text, str):
                effect_text = effect_text.strip() or "（効果）"

            amt = action.get('amount')
            if amt is None:
                amt = action.get('value1')

            if str_param == "COST":
                amt_val = amt if isinstance(amt, int) else 0
                if is_target_linked:
                    select_phrase = ""
                elif isinstance(amt, int) and amt > 0:
                    select_phrase = f"{target_str}を{amt}{unit}は、"
                else:
                    select_phrase = f"{target_str}を選び、"
                return f"{select_phrase}{duration_text}そのクリーチャーにコスト修正（{amt_val}）を与える。"

            return cls._format_keyword_grant_text(target_str, str_param, effect_text, duration_text, amt, skip_selection=is_target_linked)

        elif atype == "ADD_KEYWORD":
            str_val = action.get("str_val") or action.get("str_param") or action.get("str_val") or ""
            duration_key = action.get("duration") or action.get("input_value_key", "")

            input_key = action.get("input_value_key", "")
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            is_target_linked = bool(input_key) and (not input_usage or input_usage == "TARGET")

            duration_text = ""
            if duration_key:
                trans = CardTextResources.get_duration_text(duration_key)
                if trans and trans != duration_key:
                    duration_text = trans + "、"
                elif duration_key in CardTextResources.DURATION_TRANSLATION:
                    duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

            keyword = CardTextResources.get_keyword_text(str_val)
            if action.get("explicit_self"):
                target_str = "このカード"

            filt = action.get("filter") or action.get("target_filter") or {}
            if isinstance(filt, dict) and "zones" in filt and filt.get("zones"):
                if "SHIELD_ZONE" in filt.get("zones") or "SHIELD" in filt.get("zones"):
                    target_str = "カード"

            amt = action.get('amount', 1)
            return cls._format_keyword_grant_text(target_str, str_val, keyword, duration_text, amt, skip_selection=is_target_linked)

        elif atype == "MUTATE":
             mkind = action.get("mutation_kind", "")
             str_param = action.get("str_param") or action.get("str_val", "")

             input_key = action.get("input_value_key", "")
             input_usage = action.get("input_value_usage") or action.get("input_usage")
             is_target_linked = bool(input_key) and (not input_usage or input_usage == "TARGET")

             duration_key = action.get("duration") or action.get("input_value_key", "")

             duration_text = ""
             if duration_key:
                 trans = CardTextResources.get_duration_text(duration_key)
                 if trans and trans != duration_key:
                     duration_text = trans + "、"
                 elif duration_key in CardTextResources.DURATION_TRANSLATION:
                     duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

             if mkind == "TAP":
                 template = "{target}を{amount}{unit}選び、タップする。"
             elif mkind == "UNTAP":
                 template = "{target}を{amount}{unit}選び、アンタップする。"
             elif mkind == "POWER_MOD" or mkind == "GIVE_POWER":
                 sign = "+" if val1 >= 0 else ""
                 return f"{duration_text}{target_str}のパワーを{sign}{val1}する。"
             elif mkind == "ADD_KEYWORD" or mkind == "GIVE_ABILITY":
                 keyword = CardTextResources.get_keyword_text(str_param)
                 return cls._format_keyword_grant_text(target_str, str_param, keyword, duration_text, val1, skip_selection=is_target_linked)
             elif mkind == "REMOVE_KEYWORD":
                 keyword = CardTextResources.get_keyword_text(str_param)
                 return f"{duration_text}{target_str}の「{keyword}」を無視する。"
             elif mkind == "ADD_PASSIVE_EFFECT" or mkind == "ADD_MODIFIER":
                 if str_param:
                     kw = CardTextResources.get_keyword_text(str_param)
                     return f"{duration_text}{target_str}に「{kw}」を与える。"
                 else:
                     return f"{duration_text}{target_str}にパッシブ効果を与える。"
             elif mkind == "ADD_COST_MODIFIER":
                 return f"{duration_text}{target_str}にコスト修正を追加する。"
             else:
                 template = f"状態変更({tr(mkind)}): {{target}} (値:{val1})"

             if val1 == 0:
                 template = template.replace("{amount}{unit}選び、", "すべて")
                 val1 = ""
             return template

        elif atype == "SUMMON_TOKEN":
             token_id = action.get("str_val", "")
             count = val1 if val1 > 0 else 1

             token_name = "トークン"
             if token_id:
                 translated = tr(token_id)
                 if translated == token_id and "_" in token_id and token_id.isupper():
                     token_name = "トークン"
                 else:
                     token_name = translated

             return f"{token_name}を{count}体出す。"

        elif atype == "REGISTER_DELAYED_EFFECT":
             str_val = action.get("str_val", "")
             effect_text = CardTextResources.get_delayed_effect_text(str_val)
             if effect_text == str_val:
                  duration = val1 if val1 > 0 else 1
                  return f"遅延効果（{str_val}）を{duration}ターン登録する。"
             return effect_text

        return ""

    @classmethod
    def _format_keyword_grant_text(cls, target_str: str, key_id: str, display_text: str, duration_text: str, amount: int = None, skip_selection: bool = False) -> str:
        restriction_keys = [
            'CANNOT_ATTACK', 'CANNOT_BLOCK', 'CANNOT_ATTACK_OR_BLOCK', 'CANNOT_ATTACK_AND_BLOCK'
        ]
        is_restriction = (key_id in restriction_keys) or (str(key_id).upper() in restriction_keys)

        if is_restriction:
            if skip_selection:
                select_phrase = ""
            elif isinstance(amount, int) and amount > 0:
                select_phrase = f"{target_str}を{amount}体選び、"
            else:
                select_phrase = f"{target_str}を選び、"

            if duration_text and not duration_text.endswith('、'):
                duration_text += "、"

            return f"{select_phrase}{duration_text}そのクリーチャーは{display_text}。"

        if duration_text and not duration_text.endswith('、'):
            duration_text += "、"

        if skip_selection:
            return f"{duration_text}そのクリーチャーに「{display_text}」を与える。"

        return f"{duration_text}{target_str}に「{display_text}」を与える。"

    @classmethod
    def _format_logic_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, sample: List[Any], card_mega_last_burst: bool) -> str:
        cond_detail = action.get("condition", {}) or action.get("target_filter", {})
        cond_text = ""

        if isinstance(cond_detail, dict):
            cond_type = cond_detail.get("type", "NONE")
            if cond_type == "OPPONENT_DRAW_COUNT":
                val = cond_detail.get("value", 0)
                cond_text = f"相手がカードを{val}枚目以上引いたなら"
            elif cond_type == "COMPARE_STAT":
                key = cond_detail.get("stat_key", "")
                op = cond_detail.get("op", "=")
                val = cond_detail.get("value", 0)
                stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ""))
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
                cond_text = f"自分の{stat_name}が{op_text}なら"
            elif cond_type == "SHIELD_COUNT":
                val = cond_detail.get("value", 0)
                op = cond_detail.get("op", ">=")
                op_text = "以上" if op == ">=" else "以下" if op == "<=" else ""
                if op == "=": op_text = ""
                cond_text = f"自分のシールドが{val}つ{op_text}なら"
            elif cond_type == "COMPARE_INPUT":
                val = cond_detail.get("value", 0)
                op = cond_detail.get("op", ">=")
                input_key = action.get("input_value_key", "")
                input_desc_map = {
                    "spell_count": "墓地の呪文の数",
                    "card_count": "カードの数",
                    "creature_count": "クリーチャーの数",
                    "element_count": "エレメントの数"
                }
                input_desc = input_desc_map.get(input_key, input_key if input_key else "入力値")
                op_text = ""
                if op == ">=":
                    try:
                        op_text = f"{int(val) + 1}以上"
                    except Exception:
                        op_text = f"{val}以上"
                elif op == "<=":
                    op_text = f"{val}以下"
                elif op == "=" or op == "==":
                    op_text = f"{val}"
                elif op == ">":
                    op_text = f"{val}より多い"
                elif op == "<":
                    op_text = f"{val}より少ない"
                cond_text = f"{input_desc}が{op_text}なら"
            elif cond_type == "CIVILIZATION_MATCH":
                cond_text = "マナゾーンに同じ文明があれば"
            elif cond_type == "MANA_CIVILIZATION_COUNT":
                val = cond_detail.get("value", 0)
                op = cond_detail.get("op", ">=")
                op_text = "以上" if op == ">=" else "以下" if op == "<=" else "と同じ" if op == "=" else ""
                cond_text = f"自分のマナゾーンにある文明の数が{val}{op_text}なら"

        if not cond_text and atype != "ELSE":
            cond_text = "もし条件を満たすなら"

        if atype == "IF":
             if_true_cmds = action.get("if_true", [])
             if_true_texts = []
             for cmd in if_true_cmds:
                 if isinstance(cmd, dict):
                     cmd_text = cls._format_command(cmd, is_spell, sample, card_mega_last_burst)
                     if cmd_text:
                         if_true_texts.append(cmd_text)

             if if_true_texts:
                 actions_text = "、".join(if_true_texts)
                 return f"{cond_text}、{actions_text}"
             else:
                 return f"（{cond_text}）"

        elif atype == "IF_ELSE":
            if_true_cmds = action.get("if_true", [])
            if_false_cmds = action.get("if_false", [])

            if_true_texts = []
            for cmd in if_true_cmds:
                if isinstance(cmd, dict):
                    cmd_text = cls._format_command(cmd, is_spell, sample, card_mega_last_burst)
                    if cmd_text:
                        if_true_texts.append(cmd_text)

            if_false_texts = []
            for cmd in if_false_cmds:
                if isinstance(cmd, dict):
                    cmd_text = cls._format_command(cmd, is_spell, sample, card_mega_last_burst)
                    if cmd_text:
                        if_false_texts.append(cmd_text)

            result_parts = []
            if if_true_texts:
                result_parts.append(f"{cond_text}、" + "、".join(if_true_texts))
            if if_false_texts:
                result_parts.append("そうでなければ、" + "、".join(if_false_texts))

            if result_parts:
                return "。".join(result_parts) + "。"
            else:
                return f"（条件分岐: {cond_text}）"

        elif atype == "ELSE":
            return "（そうでなければ）"

        return ""

    @classmethod
    def _format_game_action_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int, val2: int, target_str: str, unit: str, input_key: str, input_usage: str, sample: List[Any]) -> str:
        scope = action.get("scope", action.get('target_group', "NONE"))

        if atype == "SEARCH_DECK":
            dest_zone = action.get("destination_zone", "HAND")
            if not dest_zone: dest_zone = "HAND"
            zone_str = cls._zone_to_japanese(dest_zone)
            count = val1 if val1 > 0 else 1

            if dest_zone == "HAND":
                 action_phrase = "手札に加える"
            elif dest_zone == "MANA_ZONE":
                 action_phrase = "マナゾーンに置く"
            elif dest_zone == "GRAVEYARD":
                 action_phrase = "墓地に置く"
            else:
                 action_phrase = f"{zone_str}に置く"

            template = f"自分の山札を見る。その中から{target_str}を{count}{unit}選び、{action_phrase}。その後、山札をシャッフルする。"
            if count == 1:
                template = f"自分の山札を見る。その中から{target_str}を1{unit}選び、{action_phrase}。その後、山札をシャッフルする。"
            return template

        elif atype == "LOOK_AND_ADD":
             look_count = val1 if val1 > 0 else 3
             add_count = val2 if val2 > 0 else 1
             rest_zone = action.get("rest_zone", "DECK_BOTTOM")

             rest_text = ""
             if rest_zone == "DECK_BOTTOM":
                 rest_text = "残りを好きな順序で山札の下に置く。"
             elif rest_zone == "GRAVEYARD":
                 rest_text = "残りを墓地に置く。"
             else:
                 rest_text = f"残りを{tr(rest_zone)}に置く。"

             filter_text = ""
             if target_str != "カード":
                 filter_text = f"{target_str}を"

             return f"自分の山札の上から{look_count}枚を見る。その中から{filter_text}{add_count}{unit}手札に加え、{rest_text}"

        elif atype == "PUT_CREATURE":
             count = val1 if val1 > 0 else 1
             filter_zones = action.get("filter", {}).get("zones", [])
             src_text = ""

             from_z = action.get("from_zone") or action.get("source_zone")
             if from_z and from_z != "NONE":
                 src_text = cls._zone_to_japanese(from_z) + "から"
             elif filter_zones:
                 znames = [tr(z) for z in filter_zones]
                 src_text = "または".join(znames) + "から"

             return f"{src_text}{target_str}を{count}{unit}バトルゾーンに出す。"

        elif atype == "SHUFFLE_DECK":
             return "山札をシャッフルする。"

        elif atype == "BREAK_SHIELD":
             count = val1 if val1 > 0 else 1
             if not action.get("scope") or action.get("scope") == "NONE":
                 if "相手" not in target_str:
                     target_str = "相手の" + target_str
             return f"{target_str}を{count}つブレイクする。"

        elif atype == "COST_REFERENCE":
             ref_mode = action.get("ref_mode", "")
             return f"（コスト参照: {tr(ref_mode)}）"

        elif atype == "ADD_SHIELD":
             amt = val1 if val1 > 0 else 1
             if "山札" in target_str or target_str == "カード":
                 return f"山札の上から{amt}枚をシールド化する。"
             return f"{target_str}を{amt}つシールド化する。"

        elif atype == "SEND_SHIELD_TO_GRAVE":
             amt = val1 if val1 > 0 else 1
             if scope == "OPPONENT" or scope == "PLAYER_OPPONENT":
                  return f"相手のシールドを{amt}つ選び、墓地に置く。"
             return f"{target_str}を{amt}つ墓地に置く。"

        elif atype == "SHIELD_BURN":
             amt = val1 if val1 > 0 else 1
             return f"相手のシールドを{amt}つ選び、墓地に置く。"

        elif atype == "SEARCH_DECK_BOTTOM":
             amt = val1 if val1 > 0 else 1
             return f"山札の下から{amt}枚を探す。"

        elif atype == "SEND_TO_DECK_BOTTOM":
             amt = val1 if val1 > 0 else 1
             return f"{target_str}を{amt}{unit}山札の下に置く。"

        elif atype == "RESOLVE_BATTLE":
             return f"{target_str}とバトルさせる。"

        elif atype == "MODIFY_POWER":
            val = action.get("value1", 0)
            sign = "+" if val >= 0 else ""
            return f"{target_str}のパワーを{sign}{val}する。"

        elif atype == "SELECT_NUMBER":
            val1 = action.get("value1", 0)
            val2 = action.get("value2", 0)
            if val1 > 0 and val2 > 0:
                 return f"{val1}～{val2}の数字を1つ選ぶ。"

        elif atype == "SELECT_OPTION":
            options = action.get("options", [])
            lines = []
            val1 = action.get("value1", 1)
            optional = action.get("optional", False)
            suffix = "（同じものを選んでもよい）" if optional else ""
            lines.append(f"次の中から{val1}回選ぶ。{suffix}")
            for i, opt_chain in enumerate(options):
                parts = []
                for a in opt_chain:
                    if isinstance(a, dict) and (
                        'amount' in a or 'target_group' in a or 'mutation_kind' in a or 'from_zone' in a or 'to_zone' in a
                    ):
                        parts.append(cls._format_command(a, is_spell=is_spell, sample=sample))
                    else:
                        parts.append(cls._format_action(a, is_spell, sample=sample))
                chain_text = " ".join(parts)
                lines.append(f"> {chain_text}")
            return "\n".join(lines)

        elif atype == "QUERY":
             mode = action.get("query_mode") or action.get("str_param") or action.get("str_val") or ""
             stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(mode), (None, None))
             if stat_name:
                 base = f"{stat_name}{stat_unit}を数える。"
                 if input_key:
                     usage_label = cls._format_input_usage_label(input_usage)
                     if usage_label:
                         base += f"（{usage_label}）"
                 return base

             if str(mode) == "CARDS_MATCHING_FILTER" or str(mode) == "COUNT_CARDS" or not mode:
                 filter_def = action.get("filter", {})
                 zones = filter_def.get("zones", [])
                 if target_str and target_str != "カード":
                     base = f"{target_str}の数を数える。"
                 elif zones:
                     zone_names = [tr(z) for z in zones]
                     if len(zone_names) == 1:
                         base = f"{zone_names[0]}のカードの枚数を数える。"
                     else:
                         base = f"{'または'.join(zone_names)}のカードの枚数を数える。"
                 else:
                     base = "カードの数を数える。"

                 if input_key:
                     usage_label = cls._format_input_usage_label(input_usage)
                     if usage_label:
                         base += f"（{usage_label}）"
                 return base

             base = f"質問: {tr(mode)}"
             if input_key:
                 usage_label = cls._format_input_usage_label(input_usage)
                 if usage_label:
                     base += f"（{usage_label}）"
             return base

        elif atype == "FLOW":
             ftype = action.get("flow_type") or action.get("str_val", "")
             val1 = action.get("value1", 0)

             if ftype == "PHASE_CHANGE":
                 phase_name = CardTextResources.PHASE_MAP.get(val1, str(val1))
                 return f"{phase_name}フェーズへ移行する。"
             elif ftype == "TURN_CHANGE":
                 return f"ターンを終了する。"
             elif ftype == "SET_ACTIVE_PLAYER":
                 return f"手番を変更する。"
             return f"進行制御({tr(ftype)}): {val1}"

        elif atype == "GAME_RESULT":
             res = action.get("result", "")
             if not res:
                  res = action.get("str_val") or action.get("str_param", "")
             return f"ゲームを終了する（{tr(res)}）。"

        elif atype == "ATTACH":
            return f"{target_str}をカードの下に重ねる。"

        elif atype == "MOVE_TO_UNDER_CARD":
             amt = val1 if val1 > 0 else 1
             if amt == 1:
                  return f"{target_str}をカードの下に重ねる。"
             return f"{target_str}を{amt}{unit}カードの下に重ねる。"

        elif atype == "LOCK_SPELL":
             scope = action.get("scope") or action.get("target_group", "NONE")
             target_str_lock = "プレイヤー"
             if scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                  target_str_lock = "相手"
             elif scope in ["PLAYER_SELF", "SELF"]:
                  target_str_lock = "自分"
             elif scope == "ALL_PLAYERS":
                  target_str_lock = "すべてのプレイヤー"
             else:
                  target_str_lock, _ = cls._resolve_target(action, is_spell)

             duration = val1 if val1 > 0 else 1
             return f"{target_str_lock}は{duration}ターンの間、呪文を唱えられない。"

        elif atype in ["SPELL_RESTRICTION", "CANNOT_PUT_CREATURE", "CANNOT_SUMMON_CREATURE", "PLAYER_CANNOT_ATTACK"]:
             scope = action.get("scope") or action.get("target_group", "NONE")
             target_str_lock = "プレイヤー"
             if scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                  target_str_lock = "相手"
             elif scope in ["PLAYER_SELF", "SELF"]:
                  target_str_lock = "自分"
             elif scope == "ALL_PLAYERS":
                  target_str_lock = "すべてのプレイヤー"
             else:
                  target_str_lock, _ = cls._resolve_target(action, is_spell)

             duration = val1 if val1 > 0 else 1

             if atype == "SPELL_RESTRICTION": action_text = "呪文を唱えられない"
             elif atype == "CANNOT_PUT_CREATURE": action_text = "クリーチャーを出せない"
             elif atype == "CANNOT_SUMMON_CREATURE": action_text = "クリーチャーを召喚できない"
             elif atype == "PLAYER_CANNOT_ATTACK": action_text = "攻撃できない"

             return f"{target_str_lock}は{duration}ターンの間、{action_text}。"

        elif atype == "RESET_INSTANCE":
             return f"{target_str}の状態を初期化する（効果を無視する）。"

        elif atype == "DECLARE_NUMBER":
             min_val = action.get("value1", 1)
             max_val = action.get("value2", 10)
             if min_val == 0: min_val = action.get("min_value", 1)
             if max_val == 0: max_val = action.get("amount", 10)
             return f"数字を1つ宣言する（{min_val}～{max_val}）。"

        elif atype == "DECIDE":
            sel = action.get("selected_option_index")
            if isinstance(sel, int) and sel >= 0:
                return f"選択肢{sel}を確定する。"
            indices = action.get("selected_indices") or []
            if isinstance(indices, list) and indices:
                return f"選択（{indices}）を確定する。"
            return "選択を確定する。"

        elif atype == "DECLARE_REACTION":
            if action.get("pass"):
                return "リアクション: パスする。"
            idx = action.get("reaction_index")
            if isinstance(idx, int):
                return f"リアクションを宣言する（インデックス {idx}）。"
            return "リアクションを宣言する。"

        elif atype == "STAT":
            key = action.get('stat') or action.get('str_param') or action.get('str_val')
            amount = action.get('amount', action.get('value1', 0))
            if key:
                stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(key), (None, None))
                if stat_name:
                    return f"統計更新: {stat_name} += {amount}"
            return f"統計更新: {tr(str(key))} += {amount}"

        elif atype == "GET_GAME_STAT":
            key = action.get('str_val') or action.get('result') or ''
            stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(key, (None, None))
            if stat_name:
                if sample is not None:
                    try:
                        val = cls._compute_stat_from_sample(key, sample)
                        if val is not None:
                            return f"{stat_name}（例: {val}{stat_unit}）"
                    except Exception:
                        pass
                return f"{stat_name}"
            return f"（{tr(key)}を参照）"

        elif atype == "REVEAL_CARDS":
            scope = action.get("scope") or action.get("target_group", "NONE")
            deck_owner = "相手の" if scope in ["OPPONENT", "PLAYER_OPPONENT"] else ""

            if input_key:
                return f"{deck_owner}山札の上から、その数だけ表向きにする。"
            return f"{deck_owner}山札の上から{val1}枚を表向きにする。"

        elif atype == "COUNT_CARDS":
            if not target_str or target_str == "カード":
                 return f"({tr('COUNT_CARDS')})"
            return f"{target_str}の数を数える。"

        elif atype == "SELECT_TARGET":
             amt = val1 if val1 > 0 else 1
             return f"{target_str}を{amt}{unit}選ぶ。"

        return ""
