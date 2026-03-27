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
from dm_toolkit.consts import MAX_COST_VALUE, MAX_POWER_VALUE

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
        raw_type = data.get("type", "CREATURE")
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
            rc_cond = cls._get_rc_filter_from_effects(data)
            if isinstance(rc_cond, dict) and rc_cond:
                lines.append(f"■ 革命チェンジ：{cls._format_revolution_change_text(rc_cond)}")

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
            for cmd in cmds:
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if cls._is_revolution_change_command(cmd):
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
    def _format_revolution_change_text(cls, cond: Dict[str, Any]) -> str:
        """Format REVOLUTION_CHANGE condition summary text from filter definition."""
        parts: List[str] = []

        civs = cond.get("civilizations", []) or []
        if civs:
            parts.append(cls._format_civs(civs))

        min_cost = cond.get("min_cost", 0)
        max_cost = cond.get("max_cost", MAX_COST_VALUE)
        if is_input_linked(min_cost):
            parts.append("コストその数以上")
        elif is_input_linked(max_cost):
            parts.append("コストその数以下")
        else:
            if isinstance(min_cost, int) and isinstance(max_cost, int):
                has_min = min_cost > 0
                has_max = max_cost > 0 and max_cost not in (MAX_COST_VALUE,)
                if has_min and has_max and min_cost != max_cost:
                    parts.append(f"コスト{min_cost}～{max_cost}")
                elif has_min:
                    parts.append(f"コスト{min_cost}以上")
                elif has_max:
                    parts.append(f"コスト{max_cost}以下")

        races = cond.get("races", []) or []
        noun = "/".join(races) if races else "クリーチャー"

        is_evo = cond.get("is_evolution")
        if is_evo is True:
            noun = "進化" + noun
        elif is_evo is False:
            parts.append("進化以外の")

        adjs = "の".join(parts)
        return f"{adjs}の{noun}" if adjs else noun

    @classmethod
    def _compute_stat_from_sample(cls, key: str, sample: List[Any]) -> Any:
        """Compute a concrete example value for a given stat key from a sample list.

        `sample` is typically a list of civilization strings or card dicts.
        Returns an int or None if not computable.
        """
        if not sample:
            return None

        # Normalize sample to list of civ strings when possible
        if key == "MANA_CIVILIZATION_COUNT":
            civs = set()
            for entry in sample:
                if isinstance(entry, str):
                    civs.add(entry)
                elif isinstance(entry, dict):
                    for c in entry.get('civilizations', []):
                        civs.add(c)
            return len(civs)

        # For simple count-based stats, return the number of entries
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
    def _get_rc_filter_from_effects(cls, data: dict) -> dict:
        """REVOLUTION_CHANGE コマンドの target_filter を効果ノードから探して返す。
        再発防止: 最新仕様では target_filter を単一の正規入力として扱う。"""
        for eff in data.get("effects", []):
            for cmd in (eff.get("commands", []) if isinstance(eff, dict) else []):
                if not isinstance(cmd, dict):
                    continue
                if cls._is_revolution_change_command(cmd):
                    tf = cmd.get("target_filter")
                    if tf and isinstance(tf, dict):
                        return tf
        return {}

    @classmethod
    def _is_revolution_change_command(cls, cmd: Dict[str, Any]) -> bool:
        """Return True only for the current REVOLUTION_CHANGE command type."""
        return cmd.get("type") == "REVOLUTION_CHANGE"

    @classmethod
    def _describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        types = filter_def.get("types", [])
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", MAX_COST_VALUE)
        exact_cost = filter_def.get("exact_cost")
        cost_ref = filter_def.get("cost_ref")

        adjectives = []
        if civs:
            adjectives.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))

        # Handle cost filtering
        if cost_ref:
            adjectives.append("選択した数字と同じコスト")
        elif exact_cost is not None:
            adjectives.append(f"コスト{exact_cost}")
        else:
            cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="その数")
            if cost_text:
                adjectives.append(cost_text)

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        # 再発防止: types が空のときに「クリーチャー」をデフォルトにしない。
        #   フィルターでタイプ未指定は「カード」(全タイプ対象)。
        #   CREATURE のみ指定時だけ「クリーチャー」、SPELL のみなら「呪文」、
        #   複数タイプ指定時は "/" 区切り、races 指定があればそれを優先する。
        if "ELEMENT" in types:
            noun_str = "エレメント"
        elif "SPELL" in types and "CREATURE" not in types:
            noun_str = "呪文"
        elif "CREATURE" in types:
            noun_str = "クリーチャー"
        elif types:
            noun_str = "/".join(tr(t) for t in types if t)
        else:
            noun_str = "カード"  # タイプ未指定は全タイプ対象

        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str

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
            except Exception:
                # Fallback to generic translation if formatting fails
                return tr(rtype)

        return tr(rtype)

    @classmethod
    def _safe_int(cls, value: Any, default: int = 0) -> int:
        """Best-effort int conversion helper used by text formatting paths."""
        try:
            return int(value)
        except Exception:
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
    def _format_cost_reduction(cls, cr: Dict[str, Any], sample: List[Any] = None) -> str:
        if not cr:
            return ""

        # Pre-normalize to the new schema
        norm_cr = cls._normalize_cost_reduction_dict(cr)

        ctype = norm_cr.get("type", "PASSIVE")
        name = norm_cr.get("name", "")
        if name:
            return f"{name}"

        value_mode = norm_cr.get("value_mode")
        filter_def = norm_cr.get("filter", {})

        # Describe condition (civilization, zones, races etc.)
        cond_desc = cls._describe_simple_filter(filter_def) if filter_def else ""

        # FIXED / PASSIVE simple reductions
        if (value_mode and str(value_mode).upper() in ("FIXED", "FIXED_AMOUNT", "PASSIVE")) or norm_cr.get("value") is not None:
            val = norm_cr.get("value")
            if val is None:
                # generic message
                return tr("コスト軽減: {cond}").format(cond=cond_desc)

            condition_text = ""
            if "condition" in norm_cr:
                condition_text = cls._format_condition(norm_cr["condition"]).replace(": ", "")

            # Example: "マナゾーンに闇の文明が2枚以上あれば、このカードの召喚コストは2少なくなる。"
            if condition_text:
                return f"{condition_text}、このカードの召喚コストは{val}少なくなる。"

            if cond_desc:
                return f"{cond_desc}があれば、このカードの召喚コストは{val}少なくなる。"
            return f"このカードの召喚コストは{val}少なくなる。"

        # STAT_SCALED style reductions
        if value_mode and str(value_mode).upper() == "STAT_SCALED":
            per_value = norm_cr.get("per_value")
            increment_cost = norm_cr.get("increment_cost")
            max_reduction = norm_cr.get("max_reduction")
            raw_stat_key = norm_cr.get("stat_key")

            condition_text = ""
            if "condition" in norm_cr:
                condition_text = cls._format_condition(norm_cr["condition"]).replace(": ", "")

            prefix = ""
            if condition_text:
                prefix = f"{condition_text}、"
            elif cond_desc:
                prefix = f"{cond_desc}があると、"

            stat_key_normalized = CardTextResources.normalize_stat_key(raw_stat_key) if raw_stat_key else raw_stat_key
            stat_name, unit = CardTextResources.STAT_KEY_MAP.get(stat_key_normalized, (raw_stat_key, ""))

            base = cls._format_stat_scaled_cost_text(
                target_phrase="このカードの召喚コストを、",
                stat_key=raw_stat_key,
                per_value=per_value,
                step_delta=increment_cost,
                min_stat=norm_cr.get('min_stat', 1),
                max_reduction=max_reduction,
                prefix=prefix
            )

            # If sample provided, attempt to show example computed reduction/cost
            try:
                stat_key = stat_key_normalized
                stat_name, _ = CardTextResources.STAT_KEY_MAP.get(stat_key, (stat_key or "統計", ""))
                if sample and isinstance(sample, list) and stat_key:
                    sval = None
                    # Prefer numeric entries via _compute_stat_from_sample
                    sval = cls._compute_stat_from_sample(stat_key, sample)
                    if sval is None:
                        # if sample contains dicts, try key lookup
                        for s in sample:
                            if isinstance(s, dict) and (stat_key in s or raw_stat_key in s):
                                try:
                                    sval_raw = s.get(stat_key)
                                    if sval_raw is None and raw_stat_key:
                                        sval_raw = s.get(raw_stat_key)
                                    sval = int(sval_raw)
                                    break
                                except Exception:
                                    pass
                    if sval is not None and per_value:
                        calc = max(0, int(sval) - int(norm_cr.get('min_stat', 1)) + 1) * int(per_value)
                        if isinstance(max_reduction, int):
                            calc = min(calc, max_reduction)
                        if calc > 0:
                            base += f" （例: 現在の{stat_name}{sval} → コストを{calc}削減）"
            except Exception:
                pass
            return base

        # Condition-based reductions (COMPARE_STAT, CARDS_MATCHING_FILTER)
        cond = norm_cr.get('condition') or norm_cr.get('condition_def') or {}
        if isinstance(cond, dict):
            ctype = cond.get('type')
            if ctype == 'COMPARE_STAT':
                # Use existing condition formatter to get "自分のXがYなら: " style then adapt
                cond_text = cls._format_condition(cond).strip('、: ')
                val = norm_cr.get('value') or norm_cr.get('reduction')
                if val is None:
                    return f"{cond_text}の時、このカードの召喚コストを修正する。"
                return f"{cond_text}の時、このカードの召喚コストを{val}少なくする。"
            elif ctype == 'CARDS_MATCHING_FILTER':
                f = cond.get('filter', {}) or {}
                desc = cls._describe_simple_filter(f)
                val = cond.get('value') or cond.get('count') or None
                if val:
                    return f"{desc}が{val}体以上いるなら、このカードの召喚コストは{norm_cr.get('value') or 'X'}少なくなる。"
                return f"{desc}がいるなら、このカードの召喚コストを軽減する。"

        # Fallback: show filter description
        if filter_def:
            desc = cls._describe_simple_filter(filter_def)
            return tr("コスト軽減: {desc}").format(desc=desc)

        return tr("コスト軽減")

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
            target_str = cls._format_modifier_target(effective_filter) if effective_filter else "対象"
        
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
            except Exception:
                return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"

        return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"
    
    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        """Get Japanese prefix for scope. Uses TargetScopeResolver."""
        return TargetScopeResolver.resolve_prefix(scope)
    
    @classmethod
    def _format_cost_modifier(cls, cond: str, target: str, value: int, modifier: Dict[str, Any] = None) -> str:
        """Format COST_MODIFIER modifier."""
        # 再発防止: COST_MODIFIER は FIXED/STAT_SCALED の両モードを持つ。
        # value のみで表現すると stat_key 情報が失われ、プレビュー文面が「修正する」だけになる。
        if isinstance(modifier, dict):
            vm_raw = modifier.get("value_mode")
            # 再発防止: 旧データでは value_mode が欠落しても stat_key/per_value が保存されている場合がある。
            # その場合は STAT_SCALED として扱わないと「コストを修正する。」に退化する。
            if not vm_raw and (modifier.get("stat_key") or modifier.get("per_value") is not None):
                value_mode = "STAT_SCALED"
            else:
                value_mode = str(vm_raw or "FIXED").upper()
            if value_mode == "STAT_SCALED":
                step_delta = modifier.get("value")
                if step_delta in (None, 0):
                    step_delta = modifier.get("increment_cost")
                if step_delta in (None, 0):
                    step_delta = 1
                return cls._format_stat_scaled_cost_text(
                    target_phrase=f"{cond}{target}のコストを、",
                    stat_key=modifier.get("stat_key"),
                    per_value=modifier.get("per_value", 0),
                    step_delta=step_delta,
                    min_stat=modifier.get("min_stat", 1),
                    max_reduction=modifier.get("max_reduction"),
                )

        if value > 0:
            return f"{cond}{target}のコストを{value}軽減する。"
        elif value < 0:
            return f"{cond}{target}のコストを{abs(value)}増やす。"
        return f"{cond}{target}のコストを修正する。"
    
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
    def _format_modifier_target(cls, filter_def: Dict[str, Any]) -> str:
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
            zone_names = []
            for z in zones:
                if z == "BATTLE_ZONE":
                    zone_names.append("バトルゾーン")
                elif z == "MANA_ZONE":
                    zone_names.append("マナゾーン")
                elif z == "HAND":
                    zone_names.append("手札")
                elif z == "GRAVEYARD":
                    zone_names.append("墓地")
                else:
                    zone_names.append(tr(z))
            
            if len(zone_names) == 1:
                # Single zone: "手札の" or "バトルゾーンの"
                parts.append(zone_names[0] + "の")
            else:
                # Multiple zones: "手札または墓地から"
                parts.append("または".join(zone_names) + "から")
        
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
                if types[0] == "CREATURE":
                    type_noun = "クリーチャー"
                elif types[0] == "SPELL":
                    type_noun = "呪文"
                elif types[0] == "ELEMENT":
                    type_noun = "エレメント"
            else:
                type_words = []
                if "CREATURE" in types:
                    type_words.append("クリーチャー")
                if "SPELL" in types:
                    type_words.append("呪文")
                if "ELEMENT" in types:
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
        timing_mode: str = "POST",
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
            noun = "クリーチャー" if default_type == "CREATURE" else ("呪文" if default_type == "SPELL" else "カード")
            if types:
                if "ELEMENT" in types:
                    noun = "エレメント"
                elif "SPELL" in types:
                    noun = "呪文"
                elif "CREATURE" in types:
                    noun = "クリーチャー"
                elif "CARD" in types:
                    noun = "カード"

            adjs: List[str] = []
            
            # Zone conditions (if specified, generally not mentioned in trigger text but may appear)
            if zones and "BATTLE_ZONE" not in zones:
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
            if timing_key not in ("PRE", "POST"):
                timing_key = "PRE" if cls._looks_like_pre_timing(trigger_text) else "POST"
            tmpl = tmpl_set.get(timing_key) or tmpl_set.get("POST")
            if tmpl:
                default_type = "SPELL" if trigger_type == "ON_CAST_SPELL" else "CREATURE"
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
            return "POST"
        mode = str(effect.get("timing_mode", "") or "").upper()
        if mode in ("PRE", "POST"):
            return mode
        return "PRE" if cls.is_replacement_effect(effect) else "POST"

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
        return effect.get("mode") == "REPLACEMENT" or effect.get("timing_mode") == "PRE"

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

    # --- MUTATE handlers moved to class methods for reuse and reduced branching ---

    @classmethod
    def _mutate_tap(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if val1 == 0:
            return f"{target_str}をすべてタップする。"
        return f"{target_str}を{val1}{unit}選び、タップする。"

    @classmethod
    def _mutate_untap(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if val1 == 0:
            return f"{target_str}をすべてアンタップする。"
        return f"{target_str}を{val1}{unit}選び、アンタップする。"

    @classmethod
    def _mutate_power(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        sign = "+" if val1 >= 0 else ""
        return f"{duration_text}{target_str}のパワーを{sign}{val1}する。"

    @classmethod
    def _mutate_add_keyword(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        keyword = CardTextResources.get_keyword_text(str_param)
        return cls._format_keyword_grant_text(target_str, str_param, keyword, duration_text, val1, skip_selection=is_target_linked)

    @classmethod
    def _mutate_remove_keyword(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        keyword = CardTextResources.get_keyword_text(str_param)
        return f"{duration_text}{target_str}の「{keyword}」を無視する。"

    @classmethod
    def _mutate_add_passive(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if str_param:
            kw = CardTextResources.get_keyword_text(str_param)
            return f"{duration_text}{target_str}に「{kw}」を与える。"
        return f"{duration_text}{target_str}にパッシブ効果を与える。"

    @classmethod
    def _mutate_add_cost(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        return f"{duration_text}{target_str}にコスト修正を追加する。"

    # Mapping table for mutation kinds -> handler methods
    MUTATE_KIND_HANDLERS = {
        consts.MutationKind.TAP: _mutate_tap.__func__,
        consts.MutationKind.UNTAP: _mutate_untap.__func__,
        consts.MutationKind.POWER_MOD: _mutate_power.__func__,
        consts.MutationKind.GIVE_POWER: _mutate_power.__func__,
        consts.MutationKind.ADD_KEYWORD: _mutate_add_keyword.__func__,
        consts.MutationKind.GIVE_ABILITY: _mutate_add_keyword.__func__,
        consts.MutationKind.REMOVE_KEYWORD: _mutate_remove_keyword.__func__,
        consts.MutationKind.ADD_PASSIVE_EFFECT: _mutate_add_passive.__func__,
        consts.MutationKind.ADD_MODIFIER: _mutate_add_passive.__func__,
        consts.MutationKind.ADD_COST_MODIFIER: _mutate_add_cost.__func__,
    }

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
            tf = command.get("target_filter") or command.get("filter") or {}
            cond_text = cls._format_revolution_change_text(tf) if tf else "クリーチャー"
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
    def _format_zone_move_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int, target_str: str) -> str:
        """Handle zone movement commands (TRANSITION, MOVE_CARD, REPLACE_CARD_MOVE)."""
        input_key = action.get("input_value_key", "")
        # input_usage = action.get("input_value_usage") or action.get("input_usage")
        is_generic_selection = atype in ["MOVE_CARD", "TRANSITION"]

        if atype == "TRANSITION":
            from_z = CardTextResources.normalize_zone_name(action.get("from_zone", ""))
            to_z = CardTextResources.normalize_zone_name(action.get("to_zone", ""))
            amount = val1 # Use value1 which maps to amount
            up_to_flag = bool(action.get('up_to', False))

            template_key = (from_z, to_z)
            if template_key in CardTextResources.ZONE_MOVE_TEMPLATES:
                template = CardTextResources.ZONE_MOVE_TEMPLATES[template_key]

                # Further refine template based on up_to flag, amount and input_key
                if template_key == ("BATTLE_ZONE", "GRAVEYARD") or template_key == ("BATTLE_ZONE", "MANA_ZONE"):
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                    elif amount == 0 and not input_key:
                        template = "{from_z}の{target}をすべて{to_z}に置く。"
                elif template_key == ("BATTLE_ZONE", "HAND"):
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に戻す。"
                    elif amount == 0 and not input_key:
                        template = "{from_z}の{target}をすべて{to_z}に戻す。"
                elif template_key == ("HAND", "MANA_ZONE"):
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                elif template_key == ("DECK", "HAND"):
                    if up_to_flag:
                        if target_str != "カード":
                            template = "{from_z}から{target}を{amount}{unit}まで選び、{to_z}に加える。"
                        else:
                            template = "山札からカードを最大{amount}枚まで選び、手札に加える。"
                    else:
                        if target_str != "カード":
                            template = "{from_z}から{target}を{amount}{unit}選び、{to_z}に加える。"
                elif template_key == ("GRAVEYARD", "HAND"):
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に戻す。"
                elif template_key == ("GRAVEYARD", "BATTLE_ZONE"):
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に出す。"
            else:
                if to_z == "GRAVEYARD":
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                    else:
                        template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
                elif to_z == "DECK_BOTTOM":
                    if input_key:
                        normalized_from = CardTextResources.normalize_zone_name(from_z)
                        scope = action.get("target_group") or action.get("scope", "NONE")
                        if normalized_from == "HAND":
                            to_zone_text = CardTextResources.get_zone_text(to_z)
                            linked_count = InputLinkFormatter.format_linked_count_token(action, "その同じ数")
                            owner = ""
                            if scope in ["PLAYER_SELF", "SELF"]:
                                owner = "自分の"
                            elif scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                                owner = "相手の"
                            elif scope == "ALL_PLAYERS":
                                owner = "各プレイヤーの"
                            if up_to_flag:
                                template = f"{owner}手札から{{target}}を{linked_count}だけまで選び、{to_zone_text}に置く。"
                            else:
                                template = f"{owner}手札から{{target}}を{linked_count}だけ選び、{to_zone_text}に置く。"
                        elif up_to_flag:
                            template = "{from_z}の{target}をその同じ数だけまで選び、{to_z}に置く。"
                        else:
                            template = "{from_z}の{target}をその同じ数だけ選び、{to_z}に置く。"
                    else:
                        if up_to_flag and amount > 0:
                            template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                        else:
                            template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
                else:
                    template = CardTextResources.ZONE_MOVE_TEMPLATES.get("DEFAULT", "{target}を{from_z}から{to_z}へ移動する。")

            # input_value_keyがある場合は「その同じ枚数」と表示
            # Note: We need to return the modified template to be formatted by _format_action's loop?
            # No, _format_zone_move_command is called by _format_command and returns the final string usually,
            # BUT _format_command applies substitutions at the end.
            # We should return the template and let _format_command do substitutions?
            # Or we do them here. _format_command calls _format_action.
            # Wait, `_format_command` structure:
            # 1. Map to proxy
            # 2. Call `_format_action(proxy)`
            # 3. `_format_action` calls `_format_zone_move_command`? No, we intercepted it in `_format_action`.
            # Let's verify where I put the call.
            # I put it in `_format_command` (search block above).

            # Zone name localization when placeholders are present
            if "{from_z}" in template:
                template = template.replace("{from_z}", CardTextResources.get_zone_text(from_z))
            if "{to_z}" in template:
                template = template.replace("{to_z}", CardTextResources.get_zone_text(to_z))

            return template
            # But since we extracted it, we need to handle substitutions OR return a template
            # that _format_command can handle?
            # Actually, `_format_command` calls `_format_action`.
            # The extraction happened inside `_format_action` logic (the big elif block).
            # So `val1` is already available.
            # We just need to ensure `template` is returned and `_format_action` continues to substitution?
            # The `_format_command` calls `_format_action`.
            # My Search/Replace block removed the logic from `_format_action`.
            # So `_format_zone_move_command` is called from `_format_action`.
            # Wait, look at the previous `replace_with_git_merge_diff`.
            # I replaced:
            # elif atype == "TRANSITION" ...
            # with:
            # elif atype == "TRANSITION" or ...: text = cls._format_zone_move_command(...)
            # This is inside `_format_action`.
            # So `_format_zone_move_command` returns the TEMPLATE.
            # Then `_format_action` proceeds to: `text = template.replace(...)`
            # YES.

            return template

        elif atype == "REPLACE_CARD_MOVE":
            dest_zone = action.get("destination_zone", "")
            if not dest_zone:
                dest_zone = action.get("to_zone", "DECK_BOTTOM")

            src_zone = action.get("source_zone", "")
            if not src_zone:
                src_zone = action.get("from_zone", "GRAVEYARD")

            zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
            orig_zone_str = CardTextResources.get_zone_text(src_zone) if src_zone else "元のゾーン"
            up_to_flag = bool(action.get('up_to', False))

            # 再発防止: target_group 優先。scope は後方互換。
            scope = action.get("target_group") or action.get("scope", "NONE")
            is_self_ref = scope == "SELF"

            if input_key:
                input_usage = str(action.get("input_value_usage") or action.get("input_usage") or "").upper()
                link_suffix = InputLinkFormatter.format_input_link_context_suffix(action)
                linked_target = "そのカード"
                # 再発防止: EVENT_SOURCE を対象参照として扱う場合は明示選択文を出さない。
                if input_key == "EVENT_SOURCE" and CardTextResources.normalize_zone_name(src_zone) == "BATTLE_ZONE":
                    linked_target = "そのクリーチャー"

                if input_usage in ("", "TARGET"):
                    template = f"{linked_target}を{orig_zone_str}に置くかわりに、{zone_str}に置く。"
                else:
                    up_to_text = "まで" if up_to_flag else ""
                    template = f"{linked_target}をその同じ数だけ{up_to_text}選び、{orig_zone_str}に置くかわりに、{zone_str}に置く。"

                if link_suffix:
                    template += link_suffix
            else:
                if is_self_ref:
                    # target_str = "このカード" # Handled by caller or substitution
                    # unit = ""
                    # Caller (_format_action) handles target resolution usually, but for REPLACE_CARD_MOVE
                    # specific target strings might be needed.
                    # In the original code:
                    # if is_self_ref: target_str="このカード"; unit=""
                    # BUT `target_str` is passed in as argument to this helper.
                    # We might need to override it?
                    # The original code did: `template = ...{target}...`
                    # `_format_action` calculates `target_str` BEFORE calling this block.
                    # If `is_self_ref`, `_resolve_target` might return "自分のカード" or similar.
                    # Let's rely on standard template param {target} for now,
                    # but maybe we should return a modified target string too?
                    # The helper returns `str`.
                    # Let's just return the template and hope `target_str` from `_resolve_target` is good enough.
                    # Or we hardcode "このカード" in the template if self ref.
                    template = f"このカードを{orig_zone_str}に置くかわりに、{zone_str}に置く。"
                elif val1 > 0:
                    up_to_text = "まで" if up_to_flag else ""
                    template = f"{{target}}を{{value1}}{{unit}}{up_to_text}選び、{orig_zone_str}に置くかわりに、{zone_str}に置く。"
                else:
                    template = f"{{target}}をすべて{orig_zone_str}に置くかわりに、{zone_str}に置く。"
            return template

        elif atype == "MOVE_CARD":
            dest_zone = action.get("destination_zone") or action.get("to_zone", "")
            is_all = (val1 == 0 and not input_key)
            up_to_flag = bool(action.get('up_to', False))

            src_zone = action.get("source_zone", "")
            src_str = tr(src_zone) if src_zone else ""
            zone_str = tr(dest_zone) if dest_zone else "どこか"

            if dest_zone == "HAND":
                if up_to_flag and val1 > 0:
                    template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に戻す。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に戻す。")
                else:
                    template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に戻す。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に戻す。")
            elif dest_zone == "MANA_ZONE":
                if up_to_flag and val1 > 0:
                    template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
                else:
                    template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
            elif dest_zone == "GRAVEYARD":
                if up_to_flag and val1 > 0:
                    template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
                else:
                    template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
            elif dest_zone == "DECK_BOTTOM":
                if up_to_flag and val1 > 0:
                    template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
                else:
                    template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")

            return template or ""

        return ""

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

    # --- MUTATE handlers moved to class methods for reuse and reduced branching ---


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
    @classmethod
    def generate_trigger_filter_description(cls, trigger_filter: Dict[str, Any]) -> str:
        """
        Generate detailed Japanese description of trigger filter conditions.
        
        Args:
            trigger_filter: Filter definition dictionary
        
        Returns:
            Detailed Japanese description of filter (可: "コスト3以上の呪文" など)
        """
        if not trigger_filter:
            return ""
        
        descriptions = []
        
        # Type conditions
        types = trigger_filter.get("types", [])
        if types:
            type_names = []
            for t in types:
                if t == "CREATURE":
                    type_names.append("クリーチャー")
                elif t == "SPELL":
                    type_names.append("呪文")
                elif t == "ELEMENT":
                    type_names.append("エレメント")
                elif t == "CARD":
                    type_names.append("カード")
            if type_names:
                descriptions.append("/".join(type_names))
        
        # Civilization conditions
        civs = trigger_filter.get("civilizations", [])
        if civs:
            civ_names = [CardTextResources.get_civilization_text(c) for c in civs]
            descriptions.append("/".join([c for c in civ_names if c]))
        
        # Race conditions
        races = trigger_filter.get("races", [])
        if races:
            descriptions.append("/".join(races))
        
        # Cost conditions
        exact_cost = trigger_filter.get("exact_cost")
        cost_ref = trigger_filter.get("cost_ref")
        min_cost = trigger_filter.get("min_cost", 0)
        max_cost = trigger_filter.get("max_cost", MAX_COST_VALUE)
        
        if cost_ref:
            descriptions.append("コスト【選択数字】")
        elif exact_cost is not None:
            descriptions.append(f"コスト{exact_cost}")
        else:
            cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="【入力値】")
            if cost_text:
                descriptions.append(cost_text)
        
        # Power conditions
        min_power = trigger_filter.get("min_power", 0)
        max_power = trigger_filter.get("max_power", MAX_POWER_VALUE)
        power_max_ref = trigger_filter.get("power_max_ref")
        
        if power_max_ref:
            descriptions.append("パワー【入力値】以下")
        else:
            power_text = FilterTextFormatter.format_range_text(min_power, max_power, unit="パワー", min_usage="MIN_POWER", max_usage="MAX_POWER", linked_token="【入力値】")
            if power_text:
                descriptions.append(power_text)
        
        # Tapped/Untapped state
        is_tapped = trigger_filter.get("is_tapped")
        if is_tapped == 1:
            descriptions.append("(タップ状態)")
        elif is_tapped == 0:
            descriptions.append("(アンタップ状態)")
        
        # Blocker status
        is_blocker = trigger_filter.get("is_blocker")
        if is_blocker == 1:
            descriptions.append("(ブロッカー)")
        elif is_blocker == 0:
            descriptions.append("(ブロッカー以外)")
        
        # Evolution status
        is_evolution = trigger_filter.get("is_evolution")
        if is_evolution == 1:
            descriptions.append("(進化)")
        elif is_evolution == 0:
            descriptions.append("(進化以外)")
        
        # Summoning sickness
        is_summoning_sick = trigger_filter.get("is_summoning_sick")
        if is_summoning_sick == 1:
            descriptions.append("(召喚酔い)")
        elif is_summoning_sick == 0:
            descriptions.append("(召喚酔い解除)")
        
        # Zone conditions (if not BATTLE_ZONE)
        zones = trigger_filter.get("zones", [])
        if zones and "BATTLE_ZONE" not in zones:
            zone_names = []
            for z in zones:
                zone_text = CardTextResources.normalize_zone_name(z)
                if zone_text:
                    zone_names.append(zone_text)
            if zone_names:
                descriptions.append("[" + "/".join(zone_names) + "]")
        
        return "、".join(descriptions) if descriptions else ""