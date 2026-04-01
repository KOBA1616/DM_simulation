from typing import Dict, Any, Tuple
from dm_toolkit.consts import TargetScope, CardType, Zone, MAX_COST_VALUE, MAX_POWER_VALUE
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter


class TargetResolutionService:
    """
    Unified pipeline for resolving TargetScope and TargetFilters into Japanese phrases.
    Replaces disjointed usage of TargetScopeResolver and TargetFormatter.
    """

    @classmethod
    def build_subject(cls, filter_def: Dict[str, Any], **kwargs) -> Tuple[str, str]:
        if not filter_def:
            if kwargs.get("is_modifier"): return "対象", ""
            return "カード", "枚"

        zones = filter_def.get("zones", [])
        types = filter_def.get("types", [])
        atype = kwargs.get("action_type", "")
        zone_noun, type_noun, unit = cls._determine_noun(zones, types, atype)

        is_trigger = kwargs.get("is_trigger", False)
        is_modifier = kwargs.get("is_modifier", False)

        if is_trigger and kwargs.get("default_type"):
            default_type = kwargs["default_type"]
            if default_type == CardType.CREATURE.value and not types:
                type_noun = "クリーチャー"
                unit = "体"
            elif default_type == CardType.SPELL.value and not types:
                type_noun = "呪文"
                unit = "枚"

        attrs = cls.build_attribute_list(
            filter_def,
            omit_cost=kwargs.get("omit_cost", False),
            input_usage=kwargs.get("input_usage", ""),
            has_input_key=kwargs.get("has_input_key", False),
            is_modifier=is_modifier,
            is_trigger=is_trigger
        )

        parts = []


        if is_modifier:
            if zones:
                if len(zones) == 1:
                     parts.append(ZoneFormatter.format_zone_list(zones, context="in"))
                else:
                     parts.append(ZoneFormatter.format_zone_list(zones, context="from", joiner="または"))
        elif is_trigger:
            if zones and Zone.BATTLE_ZONE.value not in zones:
                parts.append(ZoneFormatter.format_zone_list(zones, context="in", joiner="、または"))
        else:
            if zone_noun: parts.append(zone_noun + "の")


        if attrs:
            if is_trigger: parts.append("の".join(attrs) + "の")
            else:
                adjectives = "の".join(attrs)
                parts.append(adjectives + "の")

        parts.append(type_noun)
        target_desc = "".join(parts)

        if is_modifier:
            target_desc = target_desc.replace("のの", "の").replace("、の", "の")
            if not target_desc: target_desc = "対象"

        return target_desc, unit

    @classmethod
    def resolve_effective_owner(cls, scope: str, filter_def: Dict[str, Any]) -> str:
        """Determine the effective owner based on scope and filter definition."""
        if not filter_def:
            return scope
        owner = filter_def.get("owner", "NONE")
        if scope in ["NONE", "ALL"] and owner != "NONE":
            return owner
        return scope

    @classmethod
    def resolve_action_scope(cls, action: Dict[str, Any]) -> str:
        """Resolves the scope from an action dictionary."""
        scope = action.get("target_group") or action.get("scope", "NONE")
        return TargetScope.normalize(scope)

    @classmethod
    def resolve_noun(cls, scope: str, default: str = "") -> str:
        """Resolves the scope to a noun (e.g. "自分", "相手")."""
        if not scope: return default
        normalized = TargetScope.normalize(scope)
        if normalized == TargetScope.SELF: return "自分"
        elif normalized == TargetScope.OPPONENT: return "相手"
        elif normalized == TargetScope.ALL or normalized == "ALL_PLAYERS": return "すべてのプレイヤー"
        elif normalized == "NONE": return default

        prefix = cls.resolve_prefix(normalized)
        if prefix and prefix.endswith("の"):
            return prefix[:-1]
        return default

    @classmethod
    def resolve_prefix(cls, scope: str, default: str = "") -> str:
        """Resolves the scope to a prefix (e.g. "自分の", "相手の")."""
        if not scope or scope == TargetScope.ALL or scope == "NONE":
            return default
        normalized = TargetScope.normalize(scope)
        text = CardTextResources.get_scope_text(normalized)
        return text if text else default

    @classmethod
    def format_target(cls, action: Dict[str, Any], ctx: "TextGenerationContext" = None, omit_cost: bool = False, default_self_noun: str = "") -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        is_spell = getattr(ctx, "is_spell", False) if ctx else False
        scope = cls.resolve_action_scope(action)
        filter_def = action.get('target_filter') or {}
        if not isinstance(filter_def, dict):
            filter_def = {}
        atype = action.get("type", "")

        # Handle Trigger Source targeting
        if filter_def.get('is_trigger_source'):
            types = filter_def.get('types', [])
            if CardType.SPELL.value in types:
                return ("その呪文", "枚")
            elif CardType.CARD.value in types:
                return ("そのカード", "枚")
            return ("そのクリーチャー", "体")

        # Handle Battle Context specific targets
        if ctx and "battle_context_id" in ctx.metadata:
            if filter_def.get("is_battle_loser") is True:
                return ("そのバトルに負けたクリーチャー", "体")
            if filter_def.get("is_battle_winner") is True:
                return ("そのバトルに勝ったクリーチャー", "体")

        if atype == "DISCARD" and scope == "NONE":
            scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
            target = default_self_noun if default_self_noun else ("この呪文" if is_spell else "このクリーチャー")
            return (target, "枚")

        prefix, effective_scope = cls._resolve_scope(scope, filter_def)

        if not filter_def and scope == "NONE":
             # This means target self implicitly
             target_desc = default_self_noun if default_self_noun else "このカード"
             return target_desc, "枚"

        if filter_def:
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            has_input_key = bool(action.get("input_value_key") or action.get("input_link"))
            target_desc, unit = cls.build_subject(filter_def, omit_cost=omit_cost, input_usage=input_usage, has_input_key=has_input_key, action_type=atype)

            from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter

            # Use FilterTextFormatter.format_scope_prefix for standard handling
            # If prefix was custom (like ランダムな), apply it directly instead
            if prefix and prefix not in ["すべてのプレイヤーの", "すべてのプレイヤー"]:
                target_desc = prefix + target_desc
            else:
                target_desc = FilterTextFormatter.format_scope_prefix(effective_scope, target_desc)

            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            if Zone.SHIELD_ZONE.value in zones and (not types or CardType.CARD.value in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"

        else:
            target_desc = ""
            unit = "枚"

            from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
            if atype == "DESTROY":
                if scope == "OPPONENT":
                    target_desc = "相手のクリーチャー"
                    unit = "体"
            elif atype == "TAP" or atype == "UNTAP":
                target_desc = FilterTextFormatter.format_scope_prefix(scope, "クリーチャー")
                if prefix and not target_desc.startswith(prefix):
                    target_desc = prefix + target_desc
                unit = "体"
            elif atype == "DISCARD":
                target_desc = "手札"
            else:
                target_desc = "カード"
                if prefix:
                    if prefix.endswith("の") and target_desc == "カード":
                        target_desc = prefix + target_desc
                    else:
                        target_desc = prefix + target_desc
                else:
                    target_desc = FilterTextFormatter.format_scope_prefix(scope, target_desc)

        if not target_desc: target_desc = "カード"

        if target_desc == "カード" and effective_scope in ["NONE", ""] and default_self_noun:
            target_desc = default_self_noun

        if ctx:
            if ctx.last_target and ctx.last_target == target_desc and target_desc not in ["カード", "手札", "シールド"]:
                from dm_toolkit.gui.editor.formatters.utils import get_command_amount
                amt = get_command_amount(action, default=0)
                if amt == 1:
                    type_noun = ctx.last_target_type_noun or "カード"
                    target_desc = f"その{type_noun}"
                else:
                    target_desc = "それら"
            else:
                ctx.last_target = target_desc
                if filter_def:
                    _, type_noun, _ = cls._determine_noun(filter_def.get("zones", []), filter_def.get("types", []), atype)
                else:
                    if target_desc.endswith("クリーチャー"):
                        type_noun = "クリーチャー"
                    elif target_desc.endswith("呪文"):
                        type_noun = "呪文"
                    else:
                        type_noun = "カード"
                ctx.last_target_type_noun = type_noun

        return target_desc, unit

    @classmethod
    def build_attribute_list(cls, filter_def: Dict[str, Any], omit_cost: bool = False, input_usage: str = "", has_input_key: bool = False, is_modifier: bool = False, is_trigger: bool = False, is_header: bool = False) -> list[str]:
        """
        Build a list of formatted attributes (civilizations, races, cost, power) for a filter.
        """
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter

        adjectives = []

        civs = filter_def.get("civilizations", [])
        if not civs and "civilization" in filter_def:
            single = filter_def.get("civilization")
            if single:
                if isinstance(single, list):
                    civs = single
                else:
                    civs = [single]
        if civs:
            joiner = "/" if is_header else "・"
            adjectives.append(joiner.join([CardTextResources.get_civilization_text(c) for c in civs]))

        races = filter_def.get("races", [])
        if races:
            adjectives.append("/".join(races))

        if not omit_cost:
            min_cost = filter_def.get("min_cost", 0)
            max_cost = filter_def.get("max_cost", MAX_COST_VALUE)
            exact_cost = filter_def.get("exact_cost")
            cost_ref = filter_def.get("cost_ref")

            if cost_ref:
                adjectives.append("選択した数字と同じコスト")
            elif exact_cost is not None:
                adjectives.append(f"コスト{exact_cost}")
            else:
                if InputLinkFormatter.is_input_linked(min_cost, usage="MIN_COST") or (has_input_key and input_usage == "MIN_COST"):
                    adjectives.append("コストその数以上")
                elif InputLinkFormatter.is_input_linked(max_cost, usage="MAX_COST") or (has_input_key and input_usage == "MAX_COST"):
                    adjectives.append("コストその数以下")
                else:
                    cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="その数")
                    if cost_text:
                        adjectives.append(cost_text)

        min_power = filter_def.get("min_power", 0)
        max_power = filter_def.get("max_power", MAX_POWER_VALUE)

        if InputLinkFormatter.is_input_linked(min_power, usage="MIN_POWER") or (has_input_key and input_usage == "MIN_POWER"):
            adjectives.append("パワーその数以上")
        elif InputLinkFormatter.is_input_linked(max_power, usage="MAX_POWER") or (has_input_key and input_usage == "MAX_POWER"):
            adjectives.append("パワーその数以下")
        else:
            power_text = FilterTextFormatter.format_range_text(min_power, max_power, unit="パワー", min_usage="MIN_POWER", max_usage="MAX_POWER", linked_token="その数")
            if power_text:
                adjectives.append(power_text)

        flags_config = {
            "is_tapped": {True: "タップ状態" if is_modifier or is_trigger else "タップされている", False: "アンタップ状態" if is_modifier or is_trigger else "アンタップされている"},
            "is_blocker": {True: "ブロッカー" if is_modifier or is_trigger else "ブロッカーを持つ", False: "ブロッカー以外" if is_modifier or is_trigger else "ブロッカーを持たない"},
            "is_evolution": {True: "進化" if is_trigger else ("進化クリーチャー" if is_modifier else "進化"), False: "進化以外"},
            "is_summoning_sick": {True: "召喚酔い", False: "召喚酔い以外"}
        }

        for flag_key, text_map in flags_config.items():
            val = filter_def.get(flag_key)
            if val is not None:
                bool_val = bool(val) if isinstance(val, bool) else (val == 1)
                if bool_val in text_map:
                    adjectives.append(text_map[bool_val])

        flags = filter_def.get("flags", [])
        for flag in flags:
            if flag == "BLOCKER" and ("ブロッカー" not in adjectives and "ブロッカーを持つ" not in adjectives):
                adjectives.append("ブロッカー")

        if filter_def.get("exclude", False) and adjectives:
            return ["・".join(adjectives) + "以外"]

        return adjectives

    @classmethod
    def format_modifier_target(cls, filter_def: Dict[str, Any], scope: str = "ALL") -> str:
        """Format target description from filter with comprehensive support, including scope."""
        if not filter_def and scope == "NONE":
            return "このクリーチャー"
        elif not filter_def:
             return "対象"

        effective_filter = filter_def.copy() if filter_def else {}
        if scope and scope != "ALL":
             effective_filter["owner"] = scope

        owner = effective_filter.get("owner", "NONE")
        if owner == "NONE" and not effective_filter.get("zones") and not effective_filter.get("types") and not effective_filter.get("races") and not effective_filter.get("civilizations"):
            return "このクリーチャー"

        result, _ = cls.build_subject(effective_filter, is_modifier=True)

        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
        effective_scope = cls.resolve_effective_owner(scope, effective_filter)
        return FilterTextFormatter.format_scope_prefix(effective_scope, result)

    @classmethod
    def _resolve_scope(cls, scope: str, filter_def: Dict[str, Any]) -> Tuple[str, str]:
        prefix = ""
        if scope == "ALL_PLAYERS": prefix = "すべてのプレイヤーの"
        elif scope == "RANDOM": prefix = "ランダムな"

        owner = filter_def.get("owner", "NONE")
        effective_scope = scope
        if scope in ["NONE", "ALL"] and owner != "NONE":
            effective_scope = owner

        return prefix, effective_scope

    @classmethod
    def _format_cost_and_power(cls, filter_def: Dict[str, Any], action: Dict[str, Any], omit_cost: bool = False) -> str:
        # These methods are replaced by build_attribute_list, but we keep them for compatibility
        # if other parts of the code still call them.
        return ""

    @classmethod
    def _format_attributes(cls, filter_def: Dict[str, Any], action: Dict[str, Any]) -> str:
        return ""

    @classmethod
    def compose_subject_from_filter(cls, filter_def: Dict[str, Any], default_type: str) -> str:
        """Compose a subject phrase (noun + adjectives) from a trigger filter."""
        result, _ = cls.build_subject(filter_def, is_trigger=True, default_type=default_type)
        return result

    @classmethod
    def _determine_noun(cls, zones: list, types: list, atype: str) -> Tuple[str, str, str]:
        # Fully rely on consts mapping instead of hardcoding
        from dm_toolkit.consts import CARD_TYPE_UNIT_MAP, Zone, CardType

        zone_noun = ""
        type_noun = "カード"
        unit = CARD_TYPE_UNIT_MAP.get(CardType.CARD.value, "枚")

        # Handle zones with CardTextResources for consistent formatting
        # rather than hardcoding priority lists.
        # But we still need a primary zone for the noun prefix if it's a single zone
        if len(zones) == 1:
            z = zones[0]
            if z == Zone.BATTLE_ZONE.value: zone_noun = "バトルゾーン"
            elif z in [Zone.MANA_ZONE.value, "MANA"]: zone_noun = "マナゾーン"
            elif z == Zone.HAND.value: zone_noun = "手札"
            elif z in [Zone.SHIELD_ZONE.value, "SHIELD"]: zone_noun = "シールドゾーン"
            elif z == Zone.GRAVEYARD.value: zone_noun = "墓地"
            elif z == Zone.DECK.value: zone_noun = "山札"
            else:
                zone_text = CardTextResources.normalize_zone_name(z)
                if zone_text:
                    zone_noun = CardTextResources.get_zone_text(zone_text)

        # Map types directly using translations and standard maps
        # Determine highest priority matching type
        from dm_toolkit.gui.i18n import tr

        # Determine type noun and unit
        if not types:
            if Zone.BATTLE_ZONE.value in zones:
                type_noun = "クリーチャー"
                unit = CARD_TYPE_UNIT_MAP.get(CardType.CREATURE.value, "体")
            elif Zone.SHIELD_ZONE.value in zones:
                type_noun = "シールド"
                unit = CARD_TYPE_UNIT_MAP.get("SHIELD", "つ")
        else:
            if len(types) == 1:
                t = types[0]
                if t == CardType.ELEMENT.value: type_noun = "エレメント"
                elif t == CardType.CREATURE.value: type_noun = "クリーチャー"
                elif t == CardType.SPELL.value: type_noun = "呪文"
                else: type_noun = tr(t) if tr(t) else "カード"
                unit = CARD_TYPE_UNIT_MAP.get(t, "枚")
            else:
                words = []
                units = []
                for t in types:
                    if t == CardType.CREATURE.value: words.append("クリーチャー")
                    elif t == CardType.SPELL.value: words.append("呪文")
                    elif t == CardType.ELEMENT.value: words.append("エレメント")
                    else: words.append(tr(t))
                    units.append(CARD_TYPE_UNIT_MAP.get(t, "枚"))

                type_noun = "/".join(words)
                # If all units are the same, use it, else default to default unit
                if all(u == units[0] for u in units):
                    unit = units[0]

        # Contextual Overrides
        if Zone.SHIELD_ZONE.value in zones and (not types or CardType.CARD.value in types):
            type_noun = "シールド"
            unit = CARD_TYPE_UNIT_MAP.get("SHIELD", "つ")

        if atype == "SEARCH_DECK":
            zone_noun = ""

        return zone_noun, type_noun, unit
