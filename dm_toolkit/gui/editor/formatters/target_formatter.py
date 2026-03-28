from typing import Dict, Any, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.utils import is_input_linked
from dm_toolkit.gui.editor.formatters.target_scope_resolver import TargetScopeResolver
from dm_toolkit.consts import Zone, CardType, TimingMode, TargetScope, MAX_COST_VALUE, MAX_POWER_VALUE

class TargetFormatter:
    """
    Formatter to extract target resolution logic.
    Handles scopes, filter ranges, attributes, and noun resolution.
    """

    @classmethod
    def format_target(cls, action: Dict[str, Any], is_spell: bool = False, omit_cost: bool = False, default_self_noun: str = "") -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        # Accept either new ('scope'filter') or legacy ('target_group'target_filter') keys

        scope = TargetScopeResolver.resolve_action_scope(action)
        filter_def = action.get('target_filter') or {}
        if not isinstance(filter_def, dict):
            # 再発防止: target_filter が None のカードでもプレビュー更新を継続する。
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

            adjectives = cls._format_attributes(filter_def, action)
            adjectives += cls._format_cost_and_power(filter_def, action, omit_cost=omit_cost)

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

            # Apply standard scope handling, ignoring explicit prefix handling above unless prefix was overridden to non-standard ones
            if not prefix:
                target_desc = FilterTextFormatter.format_scope_prefix(effective_scope, target_desc)

            if Zone.SHIELD_ZONE.value in zones and (not types or CardType.CARD.value in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"

        else:
            target_desc = ""
            unit = "枚"
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
    def _resolve_scope(cls, scope: str, filter_def: Dict[str, Any]) -> Tuple[str, str]:
        prefix = ""
        if scope == "ALL_PLAYERS": prefix = "すべてのプレイヤーの"
        elif scope == "RANDOM": prefix = "ランダムな"

        owner = filter_def.get("owner", "NONE")
        effective_scope = scope
        # Handle explicit owner filter if scope is generic
        if scope in ["NONE", "ALL"] and owner != "NONE":
            effective_scope = owner

        return prefix, effective_scope

    @classmethod
    def _format_cost_and_power(cls, filter_def: Dict[str, Any], action: Dict[str, Any], omit_cost: bool = False) -> str:
        adjectives = ""
        input_usage = action.get("input_value_usage") or action.get("input_usage")
        has_input_key = bool(action.get("input_value_key") or action.get("input_link"))

        # Cost
        if not omit_cost:
            min_cost = filter_def.get("min_cost", 0)
            max_cost = filter_def.get("max_cost", MAX_COST_VALUE)

            if is_input_linked(min_cost, usage="MIN_COST") or (has_input_key and input_usage == "MIN_COST"):
                adjectives += "コストその数以上の"
            elif is_input_linked(max_cost, usage="MAX_COST") or (has_input_key and input_usage == "MAX_COST"):
                adjectives += "コストその数以下の"
            else:
                cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="その数")
                if cost_text:
                    adjectives += cost_text + "の"

        # Power
        min_power = filter_def.get("min_power", 0)
        max_power = filter_def.get("max_power", MAX_POWER_VALUE)

        if is_input_linked(min_power, usage="MIN_POWER") or (has_input_key and input_usage == "MIN_POWER"):
            adjectives += "パワーその数以上の"
        elif is_input_linked(max_power, usage="MAX_POWER") or (has_input_key and input_usage == "MAX_POWER"):
            adjectives += "パワーその数以下の"
        else:
            power_text = FilterTextFormatter.format_range_text(min_power, max_power, unit="パワー", min_usage="MIN_POWER", max_usage="MAX_POWER", linked_token="その数")
            if power_text:
                adjectives += power_text + "の"

        return adjectives

    @classmethod
    def _format_attributes(cls, filter_def: Dict[str, Any], action: Dict[str, Any]) -> str:
        adjectives = ""
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])

        # Robustness: Check for singular 'civilization' if plural is missing/empty
        if not civs and "civilization" in filter_def:
            single = filter_def.get("civilization")
            if single: civs = [single]

        temp_adjs = []
        if civs: temp_adjs.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))
        if races: temp_adjs.append("/".join(races))

        if temp_adjs: adjectives += "/".join(temp_adjs) + "の"
        return adjectives

    @classmethod
    def _determine_noun(cls, zones: list, types: list, atype: str) -> Tuple[str, str, str]:
        zone_noun = ""
        type_noun = "カード"
        unit = "枚"

        # 1. Determine Zone Noun
        if Zone.BATTLE_ZONE.value in zones: zone_noun = "バトルゾーン"
        elif Zone.MANA_ZONE.value in zones: zone_noun = "マナゾーン"
        elif Zone.HAND.value in zones: zone_noun = "手札"
        elif Zone.SHIELD_ZONE.value in zones: zone_noun = "シールドゾーン"
        elif Zone.GRAVEYARD.value in zones: zone_noun = "墓地"
        elif Zone.DECK.value in zones: zone_noun = "山札"

        # 2. Determine Generic Type Noun
        if CardType.ELEMENT.value in types:
            type_noun = "エレメント"
            unit = "体"
        elif CardType.CREATURE.value in types:
            type_noun = "クリーチャー"
            unit = "体"
        elif CardType.SPELL.value in types:
            type_noun = "呪文"
        elif "CROSS_GEAR" in types:
            type_noun = "クロスギア"
        elif CardType.CARD.value in types:
            type_noun = "カード"
            unit = "枚"
        elif len(types) > 1:
            # Join multiple types (e.g., Creature/Spell)
            type_words = []
            if CardType.CREATURE.value in types: type_words.append("クリーチャー")
            if CardType.SPELL.value in types: type_words.append("呪文")
            if CardType.ELEMENT.value in types: type_words.append("エレメント")
            if "CROSS_GEAR" in types: type_words.append("クロスギア")
            if type_words: type_noun = "/".join(type_words)

        # 3. Zone-specific Overrides
        if Zone.BATTLE_ZONE.value in zones:
            if CardType.CREATURE.value in types or not types:
                type_noun = "クリーチャー"
                unit = "体"
            if CardType.ELEMENT.value in types: # Explicit Element override for BZ if needed
                type_noun = "エレメント"
                unit = "枚"
        elif Zone.SHIELD_ZONE.value in zones:
            type_noun = "カード"
            unit = "つ"
        elif Zone.GRAVEYARD.value in zones:
            if CardType.CREATURE.value in types:
                type_noun = "クリーチャー"
                unit = "体"

        # Special case for SEARCH_DECK
        if atype == "SEARCH_DECK":
            zone_noun = ""

        return zone_noun, type_noun, unit

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
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        owner = filter_def.get("owner", "")  # Will NOT apply prefix here (handled in _format_modifier)
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", MAX_COST_VALUE)
        min_power = filter_def.get("min_power", 0)
        max_power = filter_def.get("max_power", MAX_POWER_VALUE)
        is_tapped = filter_def.get("is_tapped")
        is_blocker = filter_def.get("is_blocker")
        is_evolution = filter_def.get("is_evolution")

        parts = []

        # Zone prefix
        if zones:
            if len(zones) == 1:
                parts.append(CardTextResources.format_zones_list(zones) + "の")
            else:
                parts.append(CardTextResources.format_zones_list(zones, joiner="または") + "から")

        # Civilization adjective
        if civs:
            parts.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]) + "の")

        # Race adjective
        if races:
            parts.append("/".join(races) + "の")

        # Cost range
        exact_cost = filter_def.get("exact_cost")
        cost_ref = filter_def.get("cost_ref")

        if cost_ref:
            parts.append("選択した数字と同じコストの")
        elif exact_cost is not None:
            parts.append(f"コスト{exact_cost}の")
        else:
            cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="その数")
            if cost_text:
                parts.append(cost_text + "の")

        # Power range
        power_text = FilterTextFormatter.format_range_text(min_power, max_power, unit="パワー", min_usage="MIN_POWER", max_usage="MAX_POWER", linked_token="その数")
        if power_text:
            parts.append(power_text + "の")

        # Type noun
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
                if CardType.CREATURE.value in types:
                    type_words.append("クリーチャー")
                if CardType.SPELL.value in types:
                    type_words.append("呪文")
                if CardType.ELEMENT.value in types:
                    type_words.append("エレメント")
                if type_words:
                    type_noun = "/".join(type_words)

        # Flags
        flag_parts = []
        if is_tapped == 1:
            flag_parts.append("タップ状態の")
        elif is_tapped == 0:
            flag_parts.append("アンタップ状態の")

        if is_blocker == 1:
            flag_parts.append("ブロッカーの")
        elif is_blocker == 0:
            flag_parts.append("ブロッカー以外の")

        if is_evolution == 1:
            flag_parts.append("進化クリーチャーの")
        elif is_evolution == 0:
            flag_parts.append("進化以外の")

        if flag_parts:
            parts.extend(flag_parts)

        # Combine all parts
        result = "".join(parts) + type_noun

        # Cleanup
        result = result.replace("のの", "の").replace("、の", "の")

        return result if result else "対象"
