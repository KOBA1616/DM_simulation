from typing import Dict, Any, Tuple
from dm_toolkit.consts import TargetScope, CardType, Zone, MAX_COST_VALUE, MAX_POWER_VALUE
from dm_toolkit.gui.editor.text_resources import CardTextResources

class TargetResolutionService:
    """
    Unified pipeline for resolving TargetScope and TargetFilters into Japanese phrases.
    Replaces disjointed usage of TargetScopeResolver and TargetFormatter.
    """

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

        if atype == "DISCARD" and scope == "NONE":
            scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
            target = default_self_noun if default_self_noun else ("この呪文" if is_spell else "このクリーチャー")
            return (target, "枚")

        prefix, effective_scope = cls._resolve_scope(scope, filter_def)

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])

            input_usage = action.get("input_value_usage") or action.get("input_usage")
            has_input_key = bool(action.get("input_value_key") or action.get("input_link"))

            attrs = cls.build_attribute_list(filter_def, omit_cost=omit_cost, input_usage=input_usage, has_input_key=has_input_key)
            adjectives = "の".join(attrs)
            if adjectives:
                adjectives += "の"

            if filter_def.get("is_tapped", None) is True: adjectives = "タップされている" + adjectives
            elif filter_def.get("is_tapped", None) is False: adjectives = "アンタップされている" + adjectives
            if filter_def.get("is_blocker", None) is True: adjectives = "ブロッカーを持つ" + adjectives
            if filter_def.get("is_evolution", None) is True: adjectives = "進化" + adjectives

            zone_noun, type_noun, unit = cls._determine_noun(zones, types, atype)

            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "の")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)
            target_desc = "".join(parts)

            # Apply standard scope handling
            from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
            if not prefix:
                target_desc = FilterTextFormatter.format_scope_prefix(effective_scope, target_desc)

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

        return target_desc, unit

    @classmethod
    def build_attribute_list(cls, filter_def: Dict[str, Any], omit_cost: bool = False, input_usage: str = "", has_input_key: bool = False) -> list[str]:
        """
        Build a list of formatted attributes (civilizations, races, cost, power) for a filter.
        """
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter

        adjectives = []

        civs = filter_def.get("civilizations", [])
        if not civs and "civilization" in filter_def:
            single = filter_def.get("civilization")
            if single: civs = [single]
        if civs:
            adjectives.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))

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

        return adjectives

    @classmethod
    def format_modifier_target(cls, filter_def: Dict[str, Any]) -> str:
        """Format target description from filter with comprehensive support."""
        if not filter_def:
            return "対象"

        owner = filter_def.get("owner", "")
        if owner == "NONE" and not filter_def.get("zones") and not filter_def.get("types"):
            return "このクリーチャー"

        zones = filter_def.get("zones", [])
        types = filter_def.get("types", [])
        is_tapped = filter_def.get("is_tapped")
        is_blocker = filter_def.get("is_blocker")
        is_evolution = filter_def.get("is_evolution")

        parts = []

        if zones:
            if len(zones) == 1:
                parts.append(CardTextResources.format_zones_list(zones) + "の")
            else:
                parts.append(CardTextResources.format_zones_list(zones, joiner="または") + "から")

        attrs = cls.build_attribute_list(filter_def)
        if attrs:
            parts.append("の".join(attrs) + "の")

        type_noun = "カード"
        if types:
            if len(types) == 1:
                if types[0] == CardType.CREATURE.value:
                    type_noun = "クリーチャー"
                elif types[0] == CardType.SPELL.value:
                    type_noun = "呪文"
                elif types[0] == CardType.ELEMENT.value:
                    type_noun = "エレメント"
            else:
                type_words = []
                if CardType.CREATURE.value in types: type_words.append("クリーチャー")
                if CardType.SPELL.value in types: type_words.append("呪文")
                if CardType.ELEMENT.value in types: type_words.append("エレメント")
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
        f = filter_def or {}
        zones = f.get("zones", [])
        types = f.get("types", [])
        is_tapped = f.get("is_tapped")
        is_blocker = f.get("is_blocker")
        is_evolution = f.get("is_evolution")
        is_summoning_sick = f.get("is_summoning_sick")
        flags = f.get("flags", [])

        # Noun resolution
        noun = "クリーチャー" if default_type == CardType.CREATURE.value else ("呪文" if default_type == CardType.SPELL.value else "カード")
        if types:
            if CardType.ELEMENT.value in types:
                noun = "エレメント"
            elif CardType.SPELL.value in types:
                noun = "呪文"
            elif CardType.CREATURE.value in types:
                noun = "クリーチャー"
            elif CardType.CARD.value in types:
                noun = "カード"

        adjs = []

        # Zone conditions
        if zones and Zone.BATTLE_ZONE.value not in zones:
            zone_names = []
            for z in zones:
                zone_text = CardTextResources.normalize_zone_name(z)
                if zone_text:
                    zone_names.append(zone_text)
            if zone_names:
                adjs.append("/".join(zone_names))

        # Cost, Power, Civs, Races
        adjs.extend(cls.build_attribute_list(f))

        # Flags
        if is_tapped == 1:
            adjs.append("タップ状態")
        elif is_tapped == 0:
            adjs.append("アンタップ状態")
        if is_blocker == 1:
            adjs.append("ブロッカー")
        elif is_blocker == 0:
            adjs.append("ブロッカー以外")
        if is_evolution == 1:
            adjs.append("進化")
        elif is_evolution == 0:
            adjs.append("進化以外")
        if is_summoning_sick == 1:
            adjs.append("召喚酔い")
        elif is_summoning_sick == 0:
            adjs.append("召喚酔い以外")

        # Generic flags
        if flags:
            for flag in flags:
                if flag == "BLOCKER" and "ブロッカー" not in adjs:
                    adjs.append("ブロッカー")

        adj_str = "の".join(adjs)
        if adj_str:
            return f"{adj_str}の{noun}"
        return noun

    @classmethod
    def _determine_noun(cls, zones: list, types: list, atype: str) -> Tuple[str, str, str]:
        # Data-driven priority mapping for zones
        ZONE_PRIORITY = [
            (Zone.BATTLE_ZONE.value, "バトルゾーン"),
            (Zone.MANA_ZONE.value, "マナゾーン"),
            (Zone.HAND.value, "手札"),
            (Zone.SHIELD_ZONE.value, "シールドゾーン"),
            (Zone.GRAVEYARD.value, "墓地"),
            (Zone.DECK.value, "山札")
        ]

        # Data-driven priority mapping for card types -> (noun, unit)
        TYPE_PRIORITY = [
            (CardType.ELEMENT.value, "エレメント", "体"),
            (CardType.CREATURE.value, "クリーチャー", "体"),
            (CardType.SPELL.value, "呪文", "枚"),
            ("CROSS_GEAR", "クロスギア", "枚"),
            (CardType.CARD.value, "カード", "枚")
        ]

        zone_noun = ""
        type_noun = "カード"
        unit = "枚"

        # Resolve primary zone
        for z_val, z_noun in ZONE_PRIORITY:
            if z_val in zones:
                zone_noun = z_noun
                break

        # Resolve primary types
        matched_types = []
        for t_val, t_noun, t_unit in TYPE_PRIORITY:
            if t_val in types:
                matched_types.append((t_noun, t_unit))

        if matched_types:
            if len(types) > 1 and len(matched_types) > 1:
                # E.g. "クリーチャー/呪文"
                type_noun = "/".join([noun for noun, _ in matched_types])
                # Unit defaults to "枚" for mixed unless they are all "体"
                if all(u == "体" for _, u in matched_types):
                    unit = "体"
            else:
                type_noun, unit = matched_types[0]

        # Contextual Overrides
        if Zone.BATTLE_ZONE.value in zones:
            if CardType.CREATURE.value in types or not types:
                type_noun = "クリーチャー"
                unit = "体"
            if CardType.ELEMENT.value in types:
                type_noun = "エレメント"
                unit = "枚"
        elif Zone.SHIELD_ZONE.value in zones:
            type_noun = "カード"
            unit = "つ"
        elif Zone.GRAVEYARD.value in zones:
            if CardType.CREATURE.value in types:
                type_noun = "クリーチャー"
                unit = "体"

        if atype == "SEARCH_DECK":
            zone_noun = ""

        return zone_noun, type_noun, unit
