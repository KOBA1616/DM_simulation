# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import is_input_linked, get_command_amount
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordRegistry
from dm_toolkit.gui.editor.formatters.target_scope_resolver import TargetScopeResolver
import dm_toolkit.gui.editor.formatters.special_keywords # Ensure special keywords are registered
from dm_toolkit import consts
from dm_toolkit.consts import Zone, CardType, TimingMode, TargetScope, MAX_COST_VALUE, MAX_POWER_VALUE

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    @classmethod
    def generate_text(cls, data: Dict[str, Any], include_twinpact: bool = True, sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        """
        if not data:
            return ""

        if ctx is None:
            ctx = TextGenerationContext(data, sample)

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        lines.extend(cls.generate_header_lines(data))

        # 2. Body (Keywords, Effects, etc.)
        lines.append(cls.generate_body_text_lines(data, include_twinpact=False, ctx=ctx)) # Don't recurse here, handle manually

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            spell_ctx = TextGenerationContext(spell_side, sample)
            lines.append(cls.generate_text(spell_side, ctx=spell_ctx))

        return "\n".join(lines)

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
    def generate_body_text_lines(cls, data: Dict[str, Any], include_twinpact: bool = True, sample: List[Any] = None, ctx: TextGenerationContext = None) -> str:
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

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            spell_ctx = TextGenerationContext(spell_side, ctx.sample)
            lines.append(cls.generate_text(spell_side, ctx=spell_ctx))

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
                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
                    elif k == "hyper_energy":
                        kw_str += "（このクリーチャーを召喚する時、コストが異なる自分のクリーチャーを好きな数タップしてもよい、こうしてタップしたクリーチャー1体につき、このクリーチャーの召喚コストを2少なくする、ただし、コストは0以下にならない。）"
                    elif k == "mega_last_burst":
                        kw_str += "（このクリーチャーが手札、マナゾーン、または墓地に置かれた時、このカードの呪文側をコストを支払わずに唱えてもよい）"
                    elif k == "just_diver":
                        kw_str += "（このクリーチャーが出た時、次の自分のターンのはじめまで、このクリーチャーは相手に選ばれず、攻撃されない）"
                    basic_kw_lines.append(f"■ {kw_str}")

        # Append in required order
        if basic_kw_lines:
            lines.extend(basic_kw_lines)
        if special_kw_lines:
            lines.extend(special_kw_lines)

        # 再発防止: キーワードフラグ未同期でも、RC コマンド条件を本文へ反映する。
        rc_line_exists = any("革命チェンジ" in line for line in special_kw_lines)
        if not rc_line_exists:
            from dm_toolkit.gui.editor.formatters.special_keywords import RevolutionChangeFormatter
            rc_cond = RevolutionChangeFormatter.get_rc_filter_from_effects(data)
            if isinstance(rc_cond, dict) and rc_cond:
                lines.append(f"■ 革命チェンジ：{RevolutionChangeFormatter.format_revolution_change_text(rc_cond)}")

        # 2.5 Cost Reductions
        cost_reductions = cls._normalize_cost_reductions(data.get("cost_reductions", []))
        for cr in cost_reductions:
            text = cls._format_cost_reduction(cr, sample=ctx.sample)
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
            cmds = eff.get("commands", []) or []
            if not cmds:
                return False
            special_seen = False
            from dm_toolkit.gui.editor.formatters.special_keywords import RevolutionChangeFormatter
            for cmd in cmds:
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if RevolutionChangeFormatter.is_revolution_change_command(cmd):
                    special_seen = True
                elif ctype == "MEKRAID" or ctype == "FRIEND_BURST":
                    special_seen = True
                elif ctype == "CAST_SPELL" and cmd.get("str_param") == "SPELL_SIDE":
                    # Use data directly to ensure safety
                    if data.get("keywords", {}).get("mega_last_burst"):
                        special_seen = True
                    else:
                        return False
                else:
                    # Found a non-special command -> not special-only
                    return False
            # All commands were special
            return special_seen

        for effect in effects:
            if _is_special_only_effect(effect):
                continue
            text = cls._format_effect(effect, ctx)
            if text:
                lines.append(f"■ {text}")

        # 3.1 Static Abilities (常在効果)
        # Process static_abilities array which contains Modifier objects
        static_abilities = data.get("static_abilities", [])
        for static_ability in static_abilities:
            if static_ability and isinstance(static_ability, dict):
                text = cls._format_effect(static_ability, ctx)
                if text:
                    lines.append(f"■ {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("【追加能力】")
            for effect in metamorphs:
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
        # Reduce branching by using a mapping for known reaction types.
        REACTION_TEXT_MAP = {
            "NINJA_STRIKE": lambda r: f"ニンジャ・ストライク {r.get('cost', 0)}",
            "STRIKE_BACK": lambda r: "ストライク・バック",
            "COUNTER_ATTACK": lambda r: f"カウンター・アタック {r.get('cost', 0)}",
            "REVOLUTION_0_TRIGGER": lambda r: "革命0トリガー",
            # Additional mappings to reduce branching (added 2026-03-14)
            "SHIELD_TRIGGER": lambda r: "シールド・トリガー",
            "RETURN_ATTACK": lambda r: f"リターン・アタック {r.get('cost', 0)}",
            "ON_DEFEND": lambda r: "守りのトリガー",
        }

        formatter = REACTION_TEXT_MAP.get(rtype)
        if formatter is not None:
            try:
                return formatter(reaction)
            except (KeyError, TypeError, ValueError):
                # Fallback to generic translation if formatting fails
                return tr(rtype)

        return tr(rtype)

    @classmethod
    def _safe_int(cls, value: Any, default: int = 0) -> int:
        """Best-effort int conversion helper used by text formatting paths."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def _format_stat_scaled_cost_text(
        cls,
        target_phrase: str,
        stat_key: Any,
        per_value: Any,
        step_delta: Any,
        min_stat: Any,
        max_reduction: Any,
        prefix: str = "",
    ) -> str:
        """Build unified STAT_SCALED cost text for both cost_reductions and COST_MODIFIER.

        再発防止: cost_reductions と static_abilities(COST_MODIFIER) で別実装にすると
        語尾や単位表現がずれやすいので、STAT_SCALED の本文はこの関数で一元化する。
        """
        raw_stat_key = stat_key if isinstance(stat_key, str) else str(stat_key or "")
        normalized_key = CardTextResources.normalize_stat_key(raw_stat_key) if raw_stat_key else raw_stat_key
        stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(normalized_key, (normalized_key or "統計値", ""))

        per_n = cls._safe_int(per_value, 0)
        step_n = cls._safe_int(step_delta, 1)
        min_n = cls._safe_int(min_stat, 0)
        max_n = cls._safe_int(max_reduction, -1) if max_reduction is not None else -1

        if per_n > 0:
            interval_text = f"{per_n}{stat_unit}につき" if stat_unit else f"{per_n}ごとに"
        else:
            interval_text = "一定量ごとに"

        if step_n > 0:
            delta_text = f"{step_n}軽減"
            max_text = f"（最大{max_n}軽減）" if max_n > 0 else ""
            default_action = "軽減"
        elif step_n < 0:
            delta_text = f"{abs(step_n)}増加"
            max_text = f"（最大{max_n}増加）" if max_n > 0 else ""
            default_action = "増加"
        else:
            # UI 既定値(0)でも文意が消えないよう、軽減として扱う。
            delta_text = "1軽減"
            max_text = f"（最大{max_n}軽減）" if max_n > 0 else ""
            default_action = "軽減"

        threshold_text = ""
        if min_n > 0:
            if stat_unit:
                threshold_text = f"（{stat_name}が{min_n}{stat_unit}以上で適用）"
            else:
                threshold_text = f"（{stat_name}が{min_n}以上で適用）"

        return (
            f"{prefix}{target_phrase}{stat_name}の値に応じて{interval_text}{delta_text}する。"
            f"{threshold_text}{max_text}"
        )

    @classmethod
    def _normalize_cost_reduction_dict(cls, cr: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize cost reduction dictionary to consistently use the new format.

        Extracts values from legacy 'unit_cost' field into top-level properties
        so that downstream rendering only needs to deal with one schema.
        """
        normalized = cr.copy()

        # Merge properties from legacy unit_cost if present
        unit_cost = normalized.get("unit_cost", {})
        if isinstance(unit_cost, dict):
            if "value_mode" not in normalized and "value_mode" in unit_cost:
                normalized["value_mode"] = unit_cost.get("value_mode")
            if "filter" not in normalized and "filter" in unit_cost:
                normalized["filter"] = unit_cost.get("filter")
            if "value" not in normalized and "value" in unit_cost:
                normalized["value"] = unit_cost.get("value")
            if "per_value" not in normalized and "per_value" in unit_cost:
                normalized["per_value"] = unit_cost.get("per_value")
            if "increment_cost" not in normalized and "increment_cost" in unit_cost:
                normalized["increment_cost"] = unit_cost.get("increment_cost")
            if "max_reduction" not in normalized and "max_reduction" in unit_cost:
                normalized["max_reduction"] = unit_cost.get("max_reduction")
            if "stat_key" not in normalized and "stat_key" in unit_cost:
                normalized["stat_key"] = unit_cost.get("stat_key")
            if "min_stat" not in normalized and "min_stat" in unit_cost:
                normalized["min_stat"] = unit_cost.get("min_stat")

        # Resolve common alias fields
        if "value_mode" not in normalized and "mode" in normalized:
            normalized["value_mode"] = normalized.get("mode")
        if "value" not in normalized and "amount" in normalized:
            normalized["value"] = normalized.get("amount")
        if "per_value" not in normalized and "per" in normalized:
            normalized["per_value"] = normalized.get("per")
        if "increment_cost" not in normalized and "increment" in normalized:
            normalized["increment_cost"] = normalized.get("increment")

        return normalized

    @classmethod
    def _format_unified_cost_modifier(cls, mod_dict: Dict[str, Any], prefix: str = "", target_phrase: str = "このカードの召喚コストを", sample: List[Any] = None) -> str:
        """Unified logic for COST_MODIFIER and cost_reductions."""
        vm_raw = mod_dict.get("value_mode")
        if not vm_raw and (mod_dict.get("stat_key") or mod_dict.get("per_value") is not None):
            value_mode = "STAT_SCALED"
        else:
            value_mode = str(vm_raw or "FIXED").upper()

        cond = mod_dict.get('condition') or mod_dict.get('condition_def') or {}
        cond_text = cls._format_condition(cond).replace(": ", "").strip("、") if cond else ""

        filter_def = mod_dict.get("filter", {})
        cond_desc = FilterTextFormatter.describe_simple_filter(filter_def) if filter_def else ""

        if prefix and not prefix.endswith("、"):
             prefix += "、"

        if value_mode in ("FIXED", "FIXED_AMOUNT", "PASSIVE") or mod_dict.get("value") is not None:
             val = mod_dict.get("value")
             if val is None:
                  return f"{prefix}{target_phrase}修正する。"

             verb = "少なくなる" if val > 0 and target_phrase.endswith("コストは") else "多くなる" if target_phrase.endswith("コストは") else "少なくする" if val > 0 else "多くする"

             # Fallback if the user wrote target_phrase as 'XXのコストを' vs 'XXのコストは'
             if target_phrase.endswith("を"):
                  verb = "少なくする" if val > 0 else "多くする"
             elif target_phrase.endswith("は"):
                  verb = "少なくなる" if val > 0 else "多くなる"

             val_abs = abs(val)

             if cond_text:
                  return f"{prefix}{cond_text}、{target_phrase}{val_abs}{verb}。"
             if cond_desc:
                  return f"{prefix}{cond_desc}があれば、{target_phrase}{val_abs}{verb}。"

             if not prefix:
                  return f"{target_phrase}{val_abs}{verb}。"
             else:
                  return f"{prefix}{target_phrase}{val_abs}{verb}。"

        if value_mode == "STAT_SCALED":
            per_value = mod_dict.get("per_value", 0)
            step_delta = mod_dict.get("value")
            if step_delta in (None, 0):
                step_delta = mod_dict.get("increment_cost")
            if step_delta in (None, 0):
                step_delta = 1
            max_reduction = mod_dict.get("max_reduction")
            raw_stat_key = mod_dict.get("stat_key")

            pfx = prefix
            if cond_text:
                 pfx += f"{cond_text}、"
            elif cond_desc:
                 pfx += f"{cond_desc}があると、"

            # Use 'を、' or 'を' properly for STAT_SCALED
            tp = target_phrase
            if tp.endswith("は"):
                 tp = tp[:-1] + "を、"
            elif tp.endswith("を"):
                 tp = tp + "、"

            base = cls._format_stat_scaled_cost_text(
                target_phrase=tp,
                stat_key=raw_stat_key,
                per_value=per_value,
                step_delta=step_delta,
                min_stat=mod_dict.get('min_stat', 1),
                max_reduction=max_reduction,
                prefix=pfx
            )

            try:
                stat_key_normalized = CardTextResources.normalize_stat_key(raw_stat_key) if raw_stat_key else raw_stat_key
                stat_name, _ = CardTextResources.STAT_KEY_MAP.get(stat_key_normalized, (stat_key_normalized or "統計", ""))
                if sample and isinstance(sample, list) and stat_key_normalized:
                    from dm_toolkit.gui.editor.services.preview_evaluator import PreviewEvaluator
                    sval = PreviewEvaluator.compute_stat_from_sample(stat_key_normalized, sample)
                    if sval is None:
                        for s in sample:
                            if isinstance(s, dict) and (stat_key_normalized in s or raw_stat_key in s):
                                try:
                                    sval_raw = s.get(stat_key_normalized)
                                    if sval_raw is None and raw_stat_key:
                                        sval_raw = s.get(raw_stat_key)
                                    sval = int(sval_raw)
                                    break
                                except (ValueError, TypeError):
                                    pass
                    if sval is not None and per_value:
                        calc = max(0, int(sval) - int(mod_dict.get('min_stat', 1)) + 1) * int(per_value)
                        if isinstance(max_reduction, int):
                            calc = min(calc, max_reduction)
                        if calc > 0:
                            base += f" （例: 現在の{stat_name}{sval} → コストを{calc}削減）"
            except (KeyError, ValueError, TypeError):
                pass
            return base

        return f"{prefix}{target_phrase}修正する。"

    @classmethod
    def _format_cost_reduction(cls, cr: Dict[str, Any], sample: List[Any] = None) -> str:
        if not cr:
            return ""

        norm_cr = cls._normalize_cost_reduction_dict(cr)
        name = norm_cr.get("name", "")
        if name:
            return f"{name}"

        cond = norm_cr.get('condition') or norm_cr.get('condition_def') or {}
        if isinstance(cond, dict):
            ctype = cond.get('type')
            if ctype == 'COMPARE_STAT':
                cond_text = cls._format_condition(cond).strip('、: ')
                val = norm_cr.get('value') or norm_cr.get('reduction')
                if val is None:
                    return f"{cond_text}の時、このカードの召喚コストを修正する。"
                return f"{cond_text}の時、このカードの召喚コストを{val}少なくする。"
            elif ctype == 'CARDS_MATCHING_FILTER':
                f = cond.get('filter', {}) or {}
                desc = FilterTextFormatter.describe_simple_filter(f)
                val = cond.get('value') or cond.get('count') or None
                if val:
                    return f"{desc}が{val}体以上いるなら、このカードの召喚コストは{norm_cr.get('value') or 'X'}少なくなる。"
                return f"{desc}がいるなら、このカードの召喚コストを軽減する。"

        return cls._format_unified_cost_modifier(norm_cr, prefix="", target_phrase="このカードの召喚コストは", sample=sample)
    @classmethod
    def _normalize_cost_reductions(cls, crs: Any) -> List[Dict[str, Any]]:
        """Ensure cost_reductions is a list of dicts.

        Accepts None, dict (single item), or list. Filters out non-dict entries.
        """
        if not crs:
            return []
        if isinstance(crs, dict):
            return [crs]
        if isinstance(crs, list):
            out: List[Dict[str, Any]] = []
            for item in crs:
                if isinstance(item, dict):
                    out.append(item)
            return out
        return []

    @classmethod
    def _format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
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
        cond_text = cls._format_condition(condition)
        if cond_text and not cond_text.endswith("、"):
            cond_text += "、"
        
        # Build scope prefix（SCALEが SELF/OPPONENTの場合）
        scope_prefix = cls._get_scope_prefix(scope)
        
        # Build target description（フィルターがある場合）
        # NOTE: フィルターは owner を持つ場合があるが、スコープで上書きする
        effective_filter = filter_def.copy() if filter_def else {}
        if scope and scope != TargetScope.ALL:
            effective_filter["owner"] = scope
        
        if scope == "NONE" and not filter_def:
            target_str = "このクリーチャー"
        else:
            from dm_toolkit.gui.editor.formatters.target_formatter import TargetFormatter
            target_str = TargetFormatter.format_modifier_target(effective_filter) if effective_filter else "対象"
        
        # Avoid redundancy like "自身のこのクリーチャー"
        if scope == "NONE" and target_str == "このクリーチャー":
            full_target = target_str
        else:
            # Combine: condition + scope + target
            # Final structure: 「条件」「自身の」「対象」「に〜を与える」 or 「自分の」「クリーチャー」
            full_target = FilterTextFormatter.format_scope_prefix(scope, target_str)
        
        # Map modifier types to formatter callables to reduce branching
        MODIFIER_FORMATTER_MAP = {
            "COST_MODIFIER": lambda: cls._format_cost_modifier(cond_text, full_target, value, modifier=modifier),
            "POWER_MODIFIER": lambda: cls._format_power_modifier(cond_text, full_target, value),
            "GRANT_KEYWORD": lambda: cls._format_grant_keyword(cond_text, full_target, modifier),
            "SET_KEYWORD": lambda: cls._format_set_keyword(cond_text, full_target, keyword),
            "ADD_RESTRICTION": lambda: f"{cond_text}{scope_prefix}{CardTextResources.get_keyword_text(keyword)}を与える。",
        }

        formatter = MODIFIER_FORMATTER_MAP.get(mtype)
        if formatter is not None:
            try:
                return formatter()
            except (KeyError, TypeError, ValueError):
                return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"

        return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"
    
    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        """Get Japanese prefix for scope. Uses TargetScopeResolver."""
        return TargetScopeResolver.resolve_prefix(scope)
    
    @classmethod
    def _format_cost_modifier(cls, cond: str, target: str, value: int, modifier: Dict[str, Any] = None) -> str:
        """Format COST_MODIFIER modifier. Delegates to _format_unified_cost_modifier."""
        if not isinstance(modifier, dict):
             if value > 0:
                  return f"{cond}{target}のコストを{value}少なくする。"
             elif value < 0:
                  return f"{cond}{target}のコストを{abs(value)}多くする。"
             return f"{cond}{target}のコストを修正する。"

        norm_mod = modifier.copy()
        if "value" not in norm_mod:
             norm_mod["value"] = value

        return cls._format_unified_cost_modifier(norm_mod, prefix=cond, target_phrase=f"{target}のコストを")

    @classmethod
    def _format_power_modifier(cls, cond: str, target: str, value: int) -> str:
        """Format POWER_MODIFIER modifier."""
        sign = "+" if value >= 0 else ""
        if value == 0:
            return f"{cond}{target}のパワーは不変。"
        return f"{cond}{target}のパワーを{sign}{value}する。"
    
    @classmethod
    def _format_grant_keyword(cls, cond: str, target: str, modifier: Dict[str, Any]) -> str:
        """Format GRANT_KEYWORD modifier generically using modifier settings.

        Uses `modifier` fields such as `mutation_kind`str_val`, `value`, `duration`,
        and the provided `target` (which already includes scope/filter) to build
        a natural Japanese sentence. Handles restriction-style keywords specially
        but in a generic way driven by the modifier values.
        """
        # Resolve keyword id from modifier
        str_val = modifier.get('mutation_kind') or modifier.get('str_val', '')

        if not str_val:
            return f"{cond}{target}に能力を与える。"

        keyword = CardTextResources.get_keyword_text(str_val)

        # Duration text
        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = ""
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + "、"
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

        # 再発防止: GRANT_KEYWORD は modifier_form では value、コマンド互換データでは amount に数量が入る。
        # どちらの経路でも 0=全体、N>0=N体選択として同じ文面に正規化する。
        # Amount / value (how many to select)
        amt = modifier.get('value') if modifier.get('value') not in (None, 0) else modifier.get('amount', 0)
        if not isinstance(amt, int) or amt <= 0:
            amt = None

        subject = f"{cond}{target}"
        return cls._format_keyword_grant_text(subject, str_val, keyword, duration_text, amount=amt)
    
    @classmethod
    def _format_set_keyword(cls, cond: str, target: str, str_val: str) -> str:
        """Format SET_KEYWORD modifier. Uses CardTextResources."""
        if str_val:
            # Use CardTextResources for keyword translation
            keyword = CardTextResources.get_keyword_text(str_val)
            result = f"{cond}{target}は「{keyword}」を得る。"
            return result
        # Fallback: if str_val is empty, show a more helpful message
        return f"{cond}{target}は能力を得る。"
    

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
                    return cls._format_modifier(effect, sample=ctx.sample)
        
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

        cond_text = cls._format_condition(condition)
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
        output_label_map: Dict[str, str] = {}
        for command in commands:
            command_for_text = command.copy() if isinstance(command, dict) else command
            if isinstance(command_for_text, dict):
                in_key = str(command_for_text.get("input_value_key") or "")
                saved_label = str(command_for_text.get("_input_value_label") or "").strip()
                mapped_label = output_label_map.get(in_key, "") if in_key else ""
                # 再発防止: 連鎖コマンドの入力ラベルは generic "クエリ結果" より
                # 推論済みのクエリ/出力ラベルを優先して自然文を生成する。
                if mapped_label and (not saved_label or "クエリ結果" in saved_label or saved_label.startswith("Step ")):
                    command_for_text["_input_value_label"] = mapped_label

                out_key = str(command_for_text.get("output_value_key") or "")
                if out_key:
                    inferred_label = InputLinkFormatter.infer_output_value_label(command_for_text)
                    if inferred_label:
                        output_label_map[out_key] = inferred_label

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

        # Helper to compose subject phrase from filter
        def _compose_subject_from_filter(default_type: str) -> str:
            f = trigger_filter or {}
            civs = f.get("civilizations", [])
            races = f.get("races", [])
            types = f.get("types", [])
            zones = f.get("zones", [])
            min_cost = f.get("min_cost", 0)
            if min_cost is None:
                min_cost = 0
            max_cost = f.get("max_cost", MAX_COST_VALUE)
            if max_cost is None:
                max_cost = MAX_COST_VALUE
            exact_cost = f.get("exact_cost")
            cost_ref = f.get("cost_ref")
            min_power = f.get("min_power", 0)
            if min_power is None:
                min_power = 0
            max_power = f.get("max_power", MAX_POWER_VALUE)
            if max_power is None:
                max_power = MAX_POWER_VALUE
            power_max_ref = f.get("power_max_ref")
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

            adjs: List[str] = []
            
            # Zone conditions (if specified, generally not mentioned in trigger text but may appear)
            if zones and Zone.BATTLE_ZONE.value not in zones:
                zone_names = []
                for z in zones:
                    zone_text = CardTextResources.normalize_zone_name(z)
                    if zone_text:
                        zone_names.append(zone_text)
                if zone_names:
                    adjs.append("/".join(zone_names))
            
            if civs:
                adjs.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))
            if races:
                adjs.append("/".join(races))

            # Cost conditions
            if cost_ref:
                adjs.append("選択した数字と同じコスト")
            elif exact_cost is not None:
                adjs.append(f"コスト{exact_cost}")
            else:
                if is_input_linked(min_cost, usage="MIN_COST"):
                    adjs.append("コストその数以上")
                elif is_input_linked(max_cost, usage="MAX_COST"):
                    adjs.append("コストその数以下")
                else:
                    if min_cost > 0 and max_cost < MAX_COST_VALUE:
                        adjs.append(f"コスト{min_cost}～{max_cost}")
                    elif min_cost > 0:
                        adjs.append(f"コスト{min_cost}以上")
                    elif max_cost < MAX_COST_VALUE:
                        adjs.append(f"コスト{max_cost}以下")

            # Power conditions
            if power_max_ref:
                adjs.append("パワーその数以下")
            elif is_input_linked(min_power, usage="MIN_POWER"):
                adjs.append("パワーその数以上")
            elif is_input_linked(max_power, usage="MAX_POWER"):
                adjs.append("パワーその数以下")
            else:
                if min_power > 0 and max_power < MAX_POWER_VALUE:
                    adjs.append(f"パワー{min_power}～{max_power}")
                elif min_power > 0:
                    adjs.append(f"パワー{min_power}以上")
                elif max_power < MAX_POWER_VALUE:
                    adjs.append(f"パワー{max_power}以下")

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
                    if flag == "BLOCKER":
                        if "ブロッカー" not in adjs:
                            adjs.append("ブロッカー")

            adj_str = "の".join(adjs)
            if adj_str:
                return f"{adj_str}の{noun}"
            return noun

        # Structured template composition (timing/scope/trigger as variables)
        tmpl_set = CardTextResources.TRIGGER_COMPOSITION_TEMPLATES.get(trigger_type)
        if tmpl_set:
            timing_key = str(timing_mode or "").upper()
            if timing_key not in (TimingMode.PRE.value, TimingMode.POST.value):
                timing_key = TimingMode.PRE.value if cls._looks_like_pre_timing(trigger_text) else TimingMode.POST.value
            tmpl = tmpl_set.get(timing_key) or tmpl_set.get(TimingMode.POST.value)
            if tmpl:
                default_type = CardType.SPELL.value if trigger_type == "ON_CAST_SPELL" else CardType.CREATURE.value
                subject = _compose_subject_from_filter(default_type)
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
        """Normalize effect timing mode for text composition."""
        if not isinstance(effect, dict):
            return TimingMode.POST.value
        mode = str(effect.get("timing_mode", "") or "").upper()
        if mode in (TimingMode.PRE.value, TimingMode.POST.value):
            return mode
        return TimingMode.PRE.value if cls.is_replacement_effect(effect) else TimingMode.POST.value

    @classmethod
    def _looks_like_pre_timing(cls, trigger_text: str) -> bool:
        """Best-effort check whether trigger text is already in PRE/replacement tone."""
        if not trigger_text:
            return False
        return any(token in trigger_text for token in ("出る時", "唱える時", "引く時", "置かれる時", "される時", "する時"))

    @classmethod
    def _to_replacement_trigger_text(cls, trigger_text: str) -> str:
        """Convert post-event trigger text (〜た時) into replacement tone (〜る時)."""
        text = trigger_text
        replacements = [
            ("された時", "される時"),
            ("置かれた時", "置かれる時"),
            ("唱えた時", "唱える時"),
            ("引いた時", "引く時"),
            ("出た時", "出る時"),
            ("した時", "する時"),
            ("った時", "る時"),
        ]
        for src, dst in replacements:
            if src in text:
                return text.replace(src, dst)
        return text

    @classmethod
    def is_replacement_effect(cls, effect: Dict[str, Any]) -> bool:
        """Return True if the effect should be rendered as PRE/replacement timing."""
        if not isinstance(effect, dict):
            return False
        return effect.get("mode") == "REPLACEMENT" or effect.get("timing_mode") == TimingMode.PRE.value

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False, effect: Dict[str, Any] = None) -> str:
        """Get Japanese trigger text, applying replacement phrasing when needed."""
        base = CardTextResources.get_trigger_text(trigger, is_spell=is_spell)
        if effect is not None and cls.is_replacement_effect(effect):
            return cls._to_replacement_trigger_text(base)
        return base

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
    def _format_command(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        if not command:
            return ""

        # Robustly pick command type from either 'type' or legacy 'name'
        cmd_type = command.get("type") or command.get("name") or "NONE"

        # Use a copy of command to avoid modifying the input dictionary in place
        command_copy = command.copy()

        # Handle aliases mappings like legacy original_cmd_type logic
        if cmd_type == "MANA_CHARGE":
            if command_copy.get("target_group", "NONE") == "NONE" and command_copy.get("scope", "NONE") == "NONE":
                cmd_type = "ADD_MANA"
            else:
                cmd_type = "SEND_TO_MANA"
        elif cmd_type == "MEASURE_COUNT":
            cmd_type = "COUNT_CARDS"
        elif cmd_type == "SHIELD_TRIGGER":
            return "S・トリガー"

        # Check the formatter registry
        from dm_toolkit.gui.editor.formatters.command_registry import CommandFormatterRegistry
        import dm_toolkit.gui.editor.formatters # Initialize registry

        formatter_cls = CommandFormatterRegistry.get_formatter(cmd_type)
        if formatter_cls:
            # The formatter returns the template or finalized text.
            formatted_text = formatter_cls.format(command_copy, ctx)
            return formatted_text

        # 再発防止: REVOLUTION_CHANGE コマンドはカードレベルの革命チェンジテキストで使用されるが、
        # コマンドエディタ等で単独表示する場合のために直接テキストを返す。
        if cmd_type == "REVOLUTION_CHANGE":
            from dm_toolkit.gui.editor.formatters.special_keywords import RevolutionChangeFormatter
            tf = command.get("target_filter") or command.get("filter") or {}
            cond_text = RevolutionChangeFormatter.format_revolution_change_text(tf) if tf else "クリーチャー"
            return f"革命チェンジ：{cond_text}"

        cmd_type = CardTextResources.normalize_command_alias(cmd_type)
        formatter_cls = CommandFormatterRegistry.get_formatter(cmd_type)
        if formatter_cls:
            # Re-try with normalized alias.
            formatted_text = formatter_cls.format(command_copy, ctx)
            return formatted_text

        # Final fallback
        return f"({tr(cmd_type)})"

    @classmethod
    def _format_condition(cls, condition: Dict[str, Any]) -> str:
        from dm_toolkit.gui.editor.formatters.condition_formatter import ConditionFormatter
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        text = ConditionFormatter.format_condition_text(condition)
        if text:
            if condition.get("type", "NONE") == "MANA_ARMED":
                 return text + ": "
            elif "なら" in text:
                 return text + ": "
            elif condition.get("type") == "OPPONENT_PLAYED_WITHOUT_MANA":
                 return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた時: "
            elif condition.get("type") == "DURING_YOUR_TURN":
                 return CardTextResources.get_condition_text("DURING_YOUR_TURN")
            elif condition.get("type") == "DURING_OPPONENT_TURN":
                 return CardTextResources.get_condition_text("DURING_OPPONENT_TURN")
            else:
                 return text + ": "

        if condition.get("type") == "OPPONENT_PLAYED_WITHOUT_MANA":
            return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた時: "
        if condition.get("type") == "DURING_YOUR_TURN":
            return CardTextResources.get_condition_text("DURING_YOUR_TURN")
        if condition.get("type") == "DURING_OPPONENT_TURN":
            return CardTextResources.get_condition_text("DURING_OPPONENT_TURN")

        return ""

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False, effect: Dict[str, Any] = None) -> str:
        """Get Japanese trigger text, applying replacement phrasing when needed."""
        base = CardTextResources.get_trigger_text(trigger, is_spell=is_spell)
        if effect is not None and cls.is_replacement_effect(effect):
            return cls._to_replacement_trigger_text(base)
        return base

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
    def _format_cast_spell_cost_phrase(cls, action: Dict[str, Any]) -> str:
        """Return the cost phrase used by CAST_SPELL preview text."""
        play_flags = action.get("play_flags")
        explicit_cost = action.get("cost")

        is_free = True
        if isinstance(play_flags, bool):
            is_free = play_flags
        elif isinstance(play_flags, list):
            is_free = "FREE" in play_flags or "COST_FREE" in play_flags
        elif explicit_cost not in (None, 0):
            is_free = False

        if is_free:
            return "コストを支払わずに唱える"
        if isinstance(explicit_cost, int) and explicit_cost > 0:
            return f"コスト{explicit_cost}を支払って唱える"
        return "コストを支払って唱える"

    @classmethod
    def _format_selection_quantity(cls, count: Any, unit: str) -> str:
        """Format the number of cards implicitly selected by a filter."""
        if isinstance(count, int) and count > 1:
            return f"{count}{unit}まで"
        return f"1{unit}"

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Delegates to TargetFormatter for logic extraction.
        Returns (target_description, unit_counter)
        """
        return TargetFormatter.format_target(action, is_spell)

    @classmethod
    def _merge_action_texts(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        """Post-process sequence of formatted action/command texts to produce
        more natural combined sentences for common patterns.
        """
        from dm_toolkit.gui.editor.formatters.context_merger import ContextMerger
        return ContextMerger.merge(raw_items, formatted_texts)