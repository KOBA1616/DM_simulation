# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.utils import get_command_amount

from dataclasses import dataclass

@dataclass
class TargetPhrase:
    """Represents a target phrase for cost modifiers, separating noun and particle."""
    noun: str
    particle: str

    def to_string(self) -> str:
        return f"{self.noun}{self.particle}"

from dm_toolkit.gui.editor.formatters.condition_formatter import ConditionFormatter

class CostModifierStrategy:
    @classmethod
    def _safe_int(cls, value: Any, default: int = 0) -> int:
        """Best-effort int conversion helper."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def format(cls, mod_dict: Dict[str, Any], prefix: str, target: TargetPhrase, ctx: TextGenerationContext = None, cond_text: str = "", cond_desc: str = "") -> str:
        raise NotImplementedError

class FixedCostModifier(CostModifierStrategy):
    @classmethod
    def format(cls, mod_dict: Dict[str, Any], prefix: str, target: TargetPhrase, ctx: TextGenerationContext = None, cond_text: str = "", cond_desc: str = "") -> str:
        val = mod_dict.get("value")
        if val is None:
            return f"{prefix}{target.to_string()}修正する。"

        # Preserve exact legacy phrasing for Fixed Modifiers unless customized
        verb = "軽減"
        if val < 0:
            verb = "増加"

        # When specifically applying condition text, Duel Masters often uses "少なくなる"
        # However, for legacy compatibility where prefix doesn't exist, we fall back to 軽減

        target_str = target.to_string()
        if target_str.endswith("は"):
            verb = "少なくなる" if val > 0 else "多くなる"
        elif cond_text or cond_desc or prefix:
            if target.particle in ("を、", "を"):
                verb = "少なくする" if val > 0 else "多くする"
            else:
                verb = "少なくする" if val > 0 else "多くする"

        val_abs = abs(val)

        min_cost = mod_dict.get("min_cost")
        min_cost_text = ""
        if min_cost is not None:
            if min_cost <= 0:
                min_cost_text = f"ただし、コストは{min_cost}以下にならない。"
            else:
                min_cost_text = f"ただし、コストは{min_cost}より少なくならない。"

        if cond_text:
            result = f"{prefix}{cond_text}、{target_str}{val_abs}{verb}。"
        elif cond_desc:
            result = f"{prefix}{cond_desc}があれば、{target_str}{val_abs}{verb}。"
        else:
            # Preserve generic legacy output without particles
            if verb in ("軽減", "増加"):
                 # Legacy outputs usually format like: "コストをX軽減"
                 target_adj = target_str if target_str else "コストを"
                 if not target_adj.endswith("を"):
                      if target_adj.endswith("の"):
                           target_adj += "コストを"
                 result = f"{prefix}{target_adj}{val_abs}{verb}"
                 if result.endswith("コストを"):
                      result = result[:-4] + "軽減"
            else:
                 result = f"{prefix}{target_str}{val_abs}{verb}。"

        if min_cost_text:
            result += min_cost_text

        return result

class StatScaledCostModifier(CostModifierStrategy):
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
        """Build unified STAT_SCALED cost text."""
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
    def format(cls, mod_dict: Dict[str, Any], prefix: str, target: TargetPhrase, ctx: TextGenerationContext = None, cond_text: str = "", cond_desc: str = "") -> str:
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

        # STAT_SCALED requires "を、" or "を" instead of "は"
        if target.particle == "は":
            adj_target = f"{target.noun}を、"
        elif target.particle == "を":
            adj_target = f"{target.noun}を、"
        else:
            adj_target = target.to_string()

        base = cls._format_stat_scaled_cost_text(
            target_phrase=adj_target,
            stat_key=raw_stat_key,
            per_value=per_value,
            step_delta=step_delta,
            min_stat=mod_dict.get('min_stat', 1),
            max_reduction=max_reduction,
            prefix=pfx
        )

        sample = ctx.sample if ctx else None
        try:
            stat_key_normalized = CardTextResources.normalize_stat_key(raw_stat_key) if raw_stat_key else raw_stat_key
            stat_name, _ = CardTextResources.STAT_KEY_MAP.get(stat_key_normalized, (stat_key_normalized or "統計", ""))
            if sample and isinstance(sample, list) and stat_key_normalized:
                sval = ctx.evaluated_stats.get(stat_key_normalized) if ctx and hasattr(ctx, 'evaluated_stats') else None
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

        min_cost = mod_dict.get("min_cost")
        if min_cost is not None:
            if min_cost <= 0:
                base += f"ただし、コストは{min_cost}以下にならない。"
            else:
                base += f"ただし、コストは{min_cost}より少なくならない。"

        return base

class CostModifierFormatter:
    @classmethod
    def _safe_int(cls, value: Any, default: int = 0) -> int:
        """Best-effort int conversion helper used by text formatting paths."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def _parse_target_phrase(cls, phrase: str) -> TargetPhrase:
        if phrase.endswith("は"):
            return TargetPhrase(noun=phrase[:-1], particle="は")
        elif phrase.endswith("を"):
            return TargetPhrase(noun=phrase[:-1], particle="を")
        elif phrase.endswith("を、"):
            return TargetPhrase(noun=phrase[:-2], particle="を")
        return TargetPhrase(noun=phrase, particle="")

    @classmethod
    def _format_unified_cost_modifier(cls, mod_dict: Dict[str, Any], prefix: str = "", target_phrase: str = "このカードの召喚コストを", ctx: TextGenerationContext = None) -> str:
        """Unified logic for COST_MODIFIER and cost_reductions."""
        vm_raw = mod_dict.get("value_mode")
        if not vm_raw and (mod_dict.get("stat_key") or mod_dict.get("per_value") is not None):
            value_mode = "STAT_SCALED"
        else:
            value_mode = str(vm_raw or "FIXED").upper()

        cond = mod_dict.get('condition') or mod_dict.get('condition_def') or {}
        cond_text = ConditionFormatter.format_condition_text(cond).replace(": ", "").strip("、") if cond else ""

        filter_def = mod_dict.get("filter", {})
        cond_desc = FilterTextFormatter.describe_simple_filter(filter_def) if filter_def else ""

        if prefix and not prefix.endswith("、"):
             prefix += "、"

        target = cls._parse_target_phrase(target_phrase)

        if value_mode == "STAT_SCALED":
             return StatScaledCostModifier.format(mod_dict, prefix, target, ctx, cond_text, cond_desc)
        elif value_mode in ("FIXED", "FIXED_AMOUNT", "PASSIVE") or mod_dict.get("value") is not None:
             return FixedCostModifier.format(mod_dict, prefix, target, ctx, cond_text, cond_desc)

        return f"{prefix}{target.to_string()}修正する。"

    @classmethod
    def _format_cost_reduction(cls, cr: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        if not cr:
            return ""

        from dm_toolkit.gui.editor.services.data_migration import DataMigration
        norm_cr = DataMigration.normalize_cost_reduction_dict(cr)
        name = norm_cr.get("name", "")
        if name:
            return f"{name}"

        cond = norm_cr.get('condition') or norm_cr.get('condition_def') or {}
        if isinstance(cond, dict):
            ctype = cond.get('type')
            if ctype == 'COMPARE_STAT':
                cond_text = ConditionFormatter.format_condition_text(cond).strip('、: ')
                val = norm_cr.get('value') or norm_cr.get('reduction')
                if val is None:
                    return f"{cond_text}の時、このカードの召喚コストを修正する。"
                return f"{cond_text}、このカードの召喚コストは{val}少なくなる。"
            elif ctype == 'CARDS_MATCHING_FILTER':
                f = cond.get('filter', {}) or cond.get('target_filter', {}) or {}
                desc = FilterTextFormatter.describe_simple_filter(f)
                val = cond.get('value') or cond.get('count') or None
                unit = "枚" if "文明" in desc else "体"
                verb = "ある" if "文明" in desc else "いる"
                if norm_cr.get('value_mode') == 'STAT_SCALED':
                    pass
                elif val:
                    return f"{desc}が{val}{unit}以上{verb}なら、このカードの召喚コストは{norm_cr.get('value') or 'X'}少なくなる。"
                else:
                    return f"{desc}が{verb}なら、このカードの召喚コストを軽減する。"

        return cls._format_unified_cost_modifier(norm_cr, prefix="", target_phrase="このカードの召喚コストは", ctx=ctx)
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
