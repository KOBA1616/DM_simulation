from typing import Dict, Any, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.utils import is_input_linked
from dm_toolkit.gui.editor.formatters.target_scope_resolver import TargetScopeResolver
from dm_toolkit.consts import MAX_COST_VALUE, MAX_POWER_VALUE

class TargetFormatter:
    """
    Formatter to extract target resolution logic.
    Handles scopes, filter ranges, attributes, and noun resolution.
    """

    @classmethod
    def format_target(cls, action: Dict[str, Any], is_spell: bool = False) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        # Accept either new ('scope'filter') or legacy ('target_group'target_filter') keys
        # 再発防止: target_group 優先。scope は後方互換。 TargetScopeResolver に処理を委譲。
        scope = TargetScopeResolver.resolve_action_scope(action)
        filter_def = action.get("filter") or action.get('target_filter') or {}
        if not isinstance(filter_def, dict):
            # 再発防止: target_filter が None のカードでもプレビュー更新を継続する。
            filter_def = {}
        atype = action.get("type", "")

        # Handle Trigger Source targeting
        if filter_def.get('is_trigger_source'):
            types = filter_def.get('types', [])
            if 'SPELL' in types:
                return ("その呪文", "枚")
            elif 'CARD' in types:
                return ("そのカード", "枚")
            return ("そのクリーチャー", "体")

        if atype == "DISCARD" and scope == "NONE":
            scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
            target = "この呪文" if is_spell else "このクリーチャー"
            return (target, "枚")

        prefix, effective_scope = cls._resolve_scope(scope, filter_def)

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])

            adjectives = cls._format_attributes(filter_def, action)
            adjectives += cls._format_cost_and_power(filter_def, action)

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

            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
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
    def _format_cost_and_power(cls, filter_def: Dict[str, Any], action: Dict[str, Any]) -> str:
        adjectives = ""
        input_usage = action.get("input_value_usage") or action.get("input_usage")
        has_input_key = bool(action.get("input_value_key"))

        # Cost
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
        if "BATTLE_ZONE" in zones: zone_noun = "バトルゾーン"
        elif "MANA_ZONE" in zones: zone_noun = "マナゾーン"
        elif "HAND" in zones: zone_noun = "手札"
        elif "SHIELD_ZONE" in zones: zone_noun = "シールドゾーン"
        elif "GRAVEYARD" in zones: zone_noun = "墓地"
        elif "DECK" in zones: zone_noun = "山札"

        # 2. Determine Generic Type Noun
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
            # Join multiple types (e.g., Creature/Spell)
            type_words = []
            if "CREATURE" in types: type_words.append("クリーチャー")
            if "SPELL" in types: type_words.append("呪文")
            if "ELEMENT" in types: type_words.append("エレメント")
            if "CROSS_GEAR" in types: type_words.append("クロスギア")
            if type_words: type_noun = "/".join(type_words)

        # 3. Zone-specific Overrides
        if "BATTLE_ZONE" in zones:
            if "CREATURE" in types or not types:
                type_noun = "クリーチャー"
                unit = "体"
            if "ELEMENT" in types: # Explicit Element override for BZ if needed
                type_noun = "エレメント"
                unit = "枚"
        elif "SHIELD_ZONE" in zones:
            type_noun = "カード"
            unit = "つ"
        elif "GRAVEYARD" in zones:
            if "CREATURE" in types:
                type_noun = "クリーチャー"
                unit = "体"

        # Special case for SEARCH_DECK
        if atype == "SEARCH_DECK":
            zone_noun = ""

        return zone_noun, type_noun, unit
