# -*- coding: utf-8 -*-
import copy
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext, TextGenerationResult
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordRegistry
from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService as TargetScopeResolver
import dm_toolkit.gui.editor.formatters.special_keywords # Ensure special keywords are registered
from dm_toolkit import consts
from dm_toolkit.consts import Zone, CardType, TimingMode, TargetScope, MAX_COST_VALUE, MAX_POWER_VALUE
from dm_toolkit.gui.editor.data_migration import normalize_card_data

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    @classmethod
    def generate_text(cls, data: Dict[str, Any], sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        Returns a string for the complete output.
        """
        res = cls.generate(data, sample, ctx)
        return res.text

    @classmethod
    def generate(cls, data: Dict[str, Any], sample: List[Any] = None, ctx: TextGenerationContext = None) -> TextGenerationResult:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        Returns a TextGenerationResult struct holding text and semantic metadata.
        """
        if not data:
            return TextGenerationResult("", {})

        # Normalize data to current schema before processing
        data = normalize_card_data(data)

        if ctx is None:
            ctx = TextGenerationContext(data, sample)

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        lines.extend(cls.generate_header_lines(data))

        # 2. Body (Keywords, Effects, etc.)
        lines.append(cls.generate_body_text_lines(data, ctx=ctx))

        return TextGenerationResult("\n".join(lines), ctx.metadata)

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
        from dm_toolkit.gui.editor.formatters.utils import format_civs
        civs = format_civs(civs_data)

        # Use CardTextResources for translation
        raw_type = data.get("type", CardType.CREATURE.value)
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
    def generate_body_text_lines(cls, data: Dict[str, Any], sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        """
        Generates just the body text (keywords, effects, etc.) without the header.
        """
        lines = []
        if ctx is None:
            ctx = TextGenerationContext(data, sample)

        # Body Text (Keywords, Effects, etc.)
        body_text = cls.generate_body_text(data, ctx=ctx)
        if body_text:
            lines.append(body_text)

        return "\n".join(lines)

    @classmethod
    def generate_body_text(cls, data: Dict[str, Any], sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        from dm_toolkit.gui.editor.formatters.reaction_formatter import ReactionFormatter
        """
        Generates only the body text (Keywords, Effects, Reactions) without headers.
        Useful for structured preview and Twinpact separation.
        """
        if not data:
            return ""

        if ctx is None:
            ctx = TextGenerationContext(data, sample)

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

                formatter_cls = SpecialKeywordRegistry.get_formatter(k)
                if formatter_cls:
                    formatted_kw = formatter_cls.format(k, data)
                    if formatted_kw:
                        special_kw_lines.append(f"■ {formatted_kw}")
                else:
                    basic_kw_lines.append(f"■ {kw_str}")

        # Append in required order
        if basic_kw_lines:
            lines.extend(basic_kw_lines)
        if special_kw_lines:
            lines.extend(special_kw_lines)

        # Allow special keywords to provide unbound text if their flags are omitted
        # but they still have an effect command on the card (e.g. Revolution Change)
        for kw_id, formatter_cls in SpecialKeywordRegistry._formatters.items():
            if not keywords.get(kw_id):
                unbound_lines = formatter_cls.get_unbound_text(data)
                for ul in unbound_lines:
                    # Only add if not somehow generated previously
                    if not any(ul in line for line in special_kw_lines):
                        lines.append(f"■ {ul}")

        # 2.5 Cost Reductions
        from dm_toolkit.gui.editor.formatters.cost_modifier_formatter import CostModifierFormatter
        cost_reductions = CostModifierFormatter._normalize_cost_reductions(data.get("cost_reductions", []))
        for cr in cost_reductions:
            text = CostModifierFormatter._format_cost_reduction(cr, ctx=ctx)
            if text:
                lines.append(f"■ {text}")

        # 2.6 Reaction Abilities
        reactions = data.get("reaction_abilities", [])
        for r in reactions:
            text = ReactionFormatter.format(r)
            if text:
                lines.append(f"■ {text}")

        # 3. Effects (Configured actions). Skip effects that only realize special keywords
        effects = data.get("effects", [])

        for i, effect in enumerate(effects):
            if SpecialKeywordRegistry.is_special_only_effect(effect, data):
                continue
            with ctx.error_reporter.path_segment(f"effects[{i}]"):
                text = cls.format_effect(effect, ctx)
                if text:
                    lines.append(f"■ {text}")

        # 3.1 Static Abilities (常在効果)
        # Process static_abilities array which contains Modifier objects
        static_abilities = data.get("static_abilities", [])
        for i, static_ability in enumerate(static_abilities):
            if static_ability and isinstance(static_ability, dict):
                with ctx.error_reporter.path_segment(f"static_abilities[{i}]"):
                    text = cls.format_effect(static_ability, ctx)
                    if text:
                        lines.append(f"■ {text}")

        return "\n".join(lines)

    @classmethod
    def format_modifier(cls, modifier: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        """Format a static ability (Modifier) with comprehensive support for all types and conditions."""
        from dm_toolkit.consts import TargetScope
        
        mtype = modifier.get("type", "NONE")
        condition = modifier.get("condition", {})
        filter_def = modifier.get("filter", {})
        value = modifier.get("value", 0)
        
        # Prefer mutation_kind, fallback to str_val for keywords
        keyword = modifier.get("mutation_kind", "") or modifier.get("str_val", "")
        
        # Normalize scope using TargetScope
        scope = modifier.get("scope", TargetScope.ALL)
        scope = TargetScope.normalize(scope)
        
        # Build condition prefix（条件がある場合）
        if ctx and hasattr(ctx, "error_reporter"):
            with ctx.error_reporter.path_segment("condition"):
                cond_text = cls._format_condition(condition, ctx)
        else:
            cond_text = cls._format_condition(condition, ctx)

        from dm_toolkit.gui.editor.formatters.modifier_formatters import ModifierFormatterRegistry
        if mtype:
            ModifierFormatterRegistry.update_metadata(mtype, modifier, ctx)

        # Delegate fully to TargetResolutionService to build "自分のクリーチャー" etc.
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        full_target = TargetResolutionService.format_modifier_target(filter_def, scope=scope)

        return ModifierFormatterRegistry.format(mtype, cond_text, full_target, value, modifier, ctx)
    
    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        """Get Japanese prefix for scope. Uses TargetScopeResolver."""
        return TargetScopeResolver.resolve_prefix(scope)
    

    @classmethod
    def format_effect(cls, effect: Dict[str, Any], ctx: TextGenerationContext) -> str:
        # Check if this is a Modifier (static ability)
        # Modifiers have a 'type' field with specific values (COST_MODIFIER, POWER_MODIFIER, GRANT_KEYWORD, SET_KEYWORD)
        # and do NOT have a 'trigger' field (or trigger is NONE)
        if isinstance(effect, dict):
            effect_type = effect.get("type", "")
            trigger = effect.get("trigger", "NONE")
            
            # Check if this is a known Modifier type
            if effect_type in ("COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD", "ADD_RESTRICTION"):
                # Verify it's not a triggered effect
                if trigger == "NONE" or trigger not in effect:
                    return cls.format_modifier(effect, ctx=ctx)
        
        trigger = effect.get("trigger", "NONE")
        triggers = effect.get("triggers", [])
        if not triggers and trigger != "NONE":
            triggers = [trigger]

        condition = effect.get("condition", {})
        if condition is None:
            condition = {}

        if ctx and hasattr(ctx, "error_reporter"):
            with ctx.error_reporter.path_segment("condition"):
                cond_text = cls._format_condition(condition, ctx)
        else:
            cond_text = cls._format_condition(condition, ctx)

        has_active_trigger = any(t not in ("NONE", "PASSIVE_CONST") for t in triggers)
        is_passive = any(t == "PASSIVE_CONST" for t in triggers) and not has_active_trigger

        if is_passive or trigger == "PASSIVE_CONST" or not has_active_trigger:
            from dm_toolkit.gui.editor.formatters.abilities.static_ability_formatter import StaticAbilityFormatter
            return StaticAbilityFormatter.format(effect, ctx, cond_text)
        else:
            from dm_toolkit.gui.editor.formatters.abilities.triggered_ability_formatter import TriggeredAbilityFormatter
            return TriggeredAbilityFormatter.format(effect, ctx, cond_text)

    @classmethod
    def _resolve_effect_timing_mode(cls, effect: Dict[str, Any]) -> str:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.resolve_effect_timing_mode(effect)

    @classmethod
    def _to_replacement_trigger_text(cls, trigger_text: str) -> str:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.to_replacement_trigger_text(trigger_text)

    @classmethod
    def is_replacement_effect(cls, effect: Dict[str, Any]) -> bool:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.is_replacement_effect(effect)

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False, effect: Dict[str, Any] = None) -> str:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.trigger_to_japanese(trigger, is_spell, effect)

    @classmethod
    def format_command(cls, command: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        if not command:
            return ""

        # Robustly pick command type from either 'type' or legacy 'name'
        cmd_type = command.get("type") or command.get("name") or "NONE"

        # Use the command dict as read-only via functional overrides
        command_ro = copy.deepcopy(command)

        if cmd_type == "SHIELD_TRIGGER":
            return "S・トリガー"

        # Resolve aliases immediately to hide business logic from generic formatting
        cmd_type = CardTextResources.normalize_command_alias(cmd_type, command_ro)

        # Check the formatter registry
        from dm_toolkit.gui.editor.formatters.command_registry import CommandFormatterRegistry
        import dm_toolkit.gui.editor.formatters # Initialize registry

        formatter_cls = CommandFormatterRegistry.get_formatter(cmd_type)
        if formatter_cls:
            if hasattr(formatter_cls, "update_metadata"):
                formatter_cls.update_metadata(command_ro, ctx)
            # The formatter returns the template or finalized text.
            formatted_text = formatter_cls.format_with_optional(command_ro, ctx)
            return formatted_text

        # 再発防止: REVOLUTION_CHANGE コマンドはカードレベルの革命チェンジテキストで使用されるが、
        # コマンドエディタ等で単独表示する場合のために直接テキストを返す。
        if cmd_type == "REVOLUTION_CHANGE":
            formatter_cls = SpecialKeywordRegistry.get_formatter("revolution_change")
            if formatter_cls and hasattr(formatter_cls, "format_revolution_change_text"):
                tf = command.get("target_filter") or command.get("filter") or {}
                cond_text = formatter_cls.format_revolution_change_text(tf) if tf else "クリーチャー"
                return f"革命チェンジ：{cond_text}"

        # In case the normalized alias is not found, try original cmd_type
        # Some aliases might have specific logic in fallback formatters.
        cmd_type_original = command.get("type") or command.get("name") or "NONE"
        formatter_cls_orig = CommandFormatterRegistry.get_formatter(cmd_type_original)
        if formatter_cls_orig and cmd_type_original != cmd_type:
             if hasattr(formatter_cls_orig, "update_metadata"):
                 formatter_cls_orig.update_metadata(command_ro, ctx)
             formatted_text = formatter_cls_orig.format_with_optional(command_ro, ctx)
             return formatted_text

        # Final fallback
        return f"({tr(cmd_type)})"

    @classmethod
    def _format_condition(cls, condition: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        from dm_toolkit.gui.editor.formatters.condition_formatter import ConditionFormatter
        return ConditionFormatter.format_condition_text(condition, ctx)

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], ctx: "TextGenerationContext" = None) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Delegates to TargetResolutionService for logic extraction.
        Returns (target_description, unit_counter)
        """
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        return TargetResolutionService.format_target(action, ctx)

    @classmethod
    def _merge_action_texts(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        """Post-process sequence of formatted action/command texts to produce
        more natural combined sentences for common patterns.
        """
        from dm_toolkit.gui.editor.formatters.context_merger import ContextMerger
        return ContextMerger.merge(raw_items, formatted_texts)