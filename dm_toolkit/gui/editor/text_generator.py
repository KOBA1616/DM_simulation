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
        civs = cls._format_civs(civs_data)

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
            text = cls._format_reaction(r)
            if text:
                lines.append(f"■ {text}")

        # 3. Effects (Configured actions). Skip effects that only realize special keywords
        effects = data.get("effects", [])

        def _is_special_only_effect(eff: Dict[str, Any]) -> bool:
            # Dynamically fetch formatter once mapped logic or default to iteration
            cmds = eff.get("commands", []) or []
            if not cmds:
                return False

            # Query the dynamically populated reverse map
            reverse_map = SpecialKeywordRegistry.get_special_command_map()

            for cmd in cmds:
                if not isinstance(cmd, dict): continue
                cmd_type = cmd.get("type")

                # Check directly mapped command types
                if cmd_type in reverse_map:
                    for formatter_cls in reverse_map[cmd_type]:
                        if formatter_cls.is_special_only_effect(eff, data):
                            return True

            return False

        for i, effect in enumerate(effects):
            if _is_special_only_effect(effect):
                continue
            with ctx.error_reporter.path_segment(f"effects[{i}]"):
                text = cls._format_effect(effect, ctx)
                if text:
                    lines.append(f"■ {text}")

        # 3.1 Static Abilities (常在効果)
        # Process static_abilities array which contains Modifier objects
        static_abilities = data.get("static_abilities", [])
        for i, static_ability in enumerate(static_abilities):
            if static_ability and isinstance(static_ability, dict):
                with ctx.error_reporter.path_segment(f"static_abilities[{i}]"):
                    text = cls._format_effect(static_ability, ctx)
                    if text:
                        lines.append(f"■ {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("【追加能力】")
            for i, effect in enumerate(metamorphs):
                with ctx.error_reporter.path_segment(f"metamorph_abilities[{i}]"):
                    text = cls._format_effect(effect, ctx)
                    if text:
                        lines.append(f"■ {text}")

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([CardTextResources.get_civilization_text(c) for c in civs])

    @classmethod
    def _format_reaction(cls, reaction: Dict[str, Any]) -> str:
        if not reaction:
            return ""
        rtype = reaction.get("type", "NONE")

        formatter = CardTextResources.REACTION_TEXT_MAP.get(rtype)
        if formatter is not None:
            return formatter(reaction)

        return tr(rtype)

    @classmethod
    def _format_modifier(cls, modifier: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
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

        if cond_text and not cond_text.endswith("、"):
            cond_text += "、"

        from dm_toolkit.gui.editor.formatters.modifier_formatters import ModifierFormatterRegistry
        if mtype:
            ModifierFormatterRegistry.update_metadata(mtype, modifier, ctx)
        
        # Build scope prefix（SCALEが SELF/OPPONENTの場合）
        scope_prefix = cls._get_scope_prefix(scope)
        
        # Build target description（フィルターがある場合）
        # NOTE: フィルターは owner を持つ場合があるが、スコープで上書きする
        effective_filter = copy.deepcopy(filter_def) if filter_def else {}
        if scope and scope != TargetScope.ALL:
            effective_filter["owner"] = scope
        
        if scope == "NONE" and not filter_def:
            target_str = "このクリーチャー"
        else:
            from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
            target_str = TargetResolutionService.format_modifier_target(effective_filter) if effective_filter else "対象"
        
        # Avoid redundancy like "自身のこのクリーチャー"
        if scope == "NONE" and target_str == "このクリーチャー":
            full_target = target_str
        else:
            # Combine: condition + scope + target
            # Final structure: 「条件」「自身の」「対象」「に〜を与える」 or 「自分の」「クリーチャー」
            full_target = FilterTextFormatter.format_scope_prefix(scope, target_str)
        
        from dm_toolkit.gui.editor.formatters.modifier_formatters import ModifierFormatterRegistry

        return ModifierFormatterRegistry.format(mtype, cond_text, full_target, scope_prefix, value, modifier, ctx)
    
    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        """Get Japanese prefix for scope. Uses TargetScopeResolver."""
        return TargetScopeResolver.resolve_prefix(scope)
    

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any], ctx: TextGenerationContext) -> str:
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
                    return cls._format_modifier(effect, ctx=ctx)
        
        trigger = effect.get("trigger", "NONE")
        trigger_scope = effect.get("trigger_scope", "NONE")
        timing_mode = cls._resolve_effect_timing_mode(effect)
        condition = effect.get("condition", {})
        if condition is None:
            condition = {}
        actions = effect.get("actions", [])

        trigger_text = cls.trigger_to_japanese(trigger, ctx.is_spell, effect=effect)

        # Apply trigger scope (NEW: Add prefix based on scope)
        if trigger_scope and trigger_scope != "NONE" and trigger != "PASSIVE_CONST":
            trigger_text = cls._apply_trigger_scope(
                trigger_text,
                trigger_scope,
                trigger,
                effect.get("trigger_filter", {}),
                timing_mode=timing_mode,
            )

        if ctx and hasattr(ctx, "error_reporter"):
            with ctx.error_reporter.path_segment("condition"):
                cond_text = cls._format_condition(condition, ctx)
        else:
            cond_text = cls._format_condition(condition, ctx)
        cond_type = condition.get("type", "NONE")

        # Refined natural language logic
        if trigger != "NONE" and trigger != "PASSIVE_CONST":
            if cond_type == "DURING_YOUR_TURN" or cond_type == "DURING_OPPONENT_TURN":
                base_cond = cond_text.strip("、: ")
                trigger_text = f"{base_cond}、{trigger_text}" # 自分のターン中、このクリーチャーが出た時
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"相手がカードを引いた時、{val}枚目以降なら"
                cond_text = ""

        action_texts = []
        # Keep parallel lists of raw and formatted for merging logic
        raw_items = []

        # Commands-Only Policy:
        # We now expect 'commands' to be the sole source of truth.
        commands = effect.get("commands", [])

        # 連鎖コマンドの入力ラベル推定副作用の分離
        # AST構築パスでラベルを事前推論して独立させる
        from dm_toolkit.gui.editor.formatters.input_link_ast import InputLinkASTBuilder
        commands_with_labels = InputLinkASTBuilder.infer_command_labels(commands)

        for i, command in enumerate(commands_with_labels):
            with ctx.error_reporter.path_segment(f"commands[{i}]"):
                command_for_text = copy.deepcopy(command) if isinstance(command, dict) else command
                raw_items.append(command_for_text)
                action_texts.append(cls._format_command(command_for_text, ctx))

        # Try to merge common sequential patterns for more natural language
        full_action_text = cls._merge_action_texts(raw_items, action_texts)

        # If it's a Spell's main effect (ON_PLAY), we can often omit the trigger text "Played/Cast"
        if ctx.is_spell and trigger == "ON_PLAY":
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
    def _apply_trigger_scope(
        cls,
        trigger_text: str,
        scope: str,
        trigger_type: str,
        trigger_filter: Dict[str, Any] = None,
        timing_mode: str = TimingMode.POST.value,
    ) -> str:
        """
        Apply scope prefix to trigger text (e.g., "ON_CAST_SPELL" + "OPPONENT" -> "相手が呪文を唱えた時").
        """
        if not scope or scope == "NONE" or scope == "ALL":
            return trigger_text

        scope_text = TargetScopeResolver.resolve_noun(scope)
        if not scope_text:
            return trigger_text

        # Uses FilterTextFormatter to avoid scope prefix duplication
        # if the trigger text already has "相手が" or "自分が", etc.
        formatted = FilterTextFormatter.format_scope_prefix(scope, trigger_text)
        # If FilterTextFormatter actually did not prepend anything new (it detected the prefix),
        # return the trigger text as-is to preserve structural replacements correctly,
        # or we could just use its output which is identical.
        if formatted == trigger_text:
            return trigger_text

        # Structured template composition (timing/scope/trigger as variables)
        tmpl_set = CardTextResources.TRIGGER_COMPOSITION_TEMPLATES.get(trigger_type)
        if tmpl_set:
            timing_key = str(timing_mode or "").upper()
            if timing_key not in (TimingMode.PRE.value, TimingMode.POST.value):
                timing_key = TimingMode.PRE.value if cls._looks_like_pre_timing(trigger_text) else TimingMode.POST.value
            tmpl = tmpl_set.get(timing_key) or tmpl_set.get(TimingMode.POST.value)
            if tmpl:
                default_type = CardType.SPELL.value if trigger_type == "ON_CAST_SPELL" else CardType.CREATURE.value
                from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
                subject = TargetResolutionService.compose_subject_from_filter(trigger_filter, default_type)
                return tmpl.format(scope_text=scope_text, subject=subject)

        if trigger_type == "ON_SHIELD_ADD":
             # "カードがシールドゾーンに..." -> replace "シールドゾーン" with "自分の/相手のシールドゾーン"
             if "シールドゾーン" in trigger_text:
                 return trigger_text.replace("シールドゾーン", f"{scope_text}のシールドゾーン")

        # Default fallbacks
        if trigger_text.startswith("この"):
             # "このクリーチャー..." -> "相手のこのクリーチャー..." (Syntactically valid for 'Target's this creature')
             return f"{scope_text}の{trigger_text}"

        # Default to "の" prefix
        return f"{scope_text}の{trigger_text}"

    @classmethod
    def _resolve_effect_timing_mode(cls, effect: Dict[str, Any]) -> str:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.resolve_effect_timing_mode(effect)

    @classmethod
    def _looks_like_pre_timing(cls, trigger_text: str) -> bool:
        from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
        return TriggerFormatter.looks_like_pre_timing(trigger_text)

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
    def _format_keyword_grant_text(cls, target_str: str, key_id: str, display_text: str, duration_text: str, amount: int = None, skip_selection: bool = False) -> str:
        """Helper to format keyword granting text.

        amount=None or 0: apply to all matching targets (no selection).
        amount>0: select N targets (N体選び).
        skip_selection=True: target already determined by input link.
        """
        restriction_keys = [
            'CANNOT_ATTACK', 'CANNOT_BLOCK', 'CANNOT_ATTACK_OR_BLOCK', 'CANNOT_ATTACK_AND_BLOCK'
        ]
        is_restriction = (key_id in restriction_keys) or (str(key_id).upper() in restriction_keys)

        # Normalize duration_text end
        if duration_text and not duration_text.endswith('、'):
            duration_text += "、"

        if is_restriction:
            # 再発防止: skip_selection/amount=0/amount>0 の 3 ケースを明示的に分岐する
            if skip_selection:
                # 入力リンク経由で対象決定済み
                return f"{duration_text}そのクリーチャーは{display_text}。"
            elif isinstance(amount, int) and amount > 0:
                # N体選び
                return f"{target_str}を{amount}体選び、{duration_text}そのクリーチャーは{display_text}。"
            else:
                # amount=0 またの未指定 → 対象すべてに適用
                return f"{duration_text}{target_str}は{display_text}。"

        # 通常キーワード付与
        if skip_selection:
            return f"{duration_text}そのクリーチャーに「{display_text}」を与える。"
        if isinstance(amount, int) and amount > 0:
            # 再発防止: amount>0 の時が遷局テキストがなかった以前のバグを修正
            return f"{duration_text}{target_str}を{amount}体選び、「{display_text}」を与える。"
        # amount=0 またの未指定 → 対象すべてに適用（選択文なし）
        return f"{duration_text}{target_str}に「{display_text}」を与える。"

    @classmethod
    def _format_command(cls, command: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
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
    def _format_selection_quantity(cls, count: Any, unit: str) -> str:
        """Format the number of cards implicitly selected by a filter."""
        if isinstance(count, int) and count > 1:
            return f"{count}{unit}まで"
        return f"1{unit}"

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