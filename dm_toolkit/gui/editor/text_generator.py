# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import is_input_linked, get_command_amount
from dm_toolkit.gui.editor.formatters.variable_link_formatter import VariableLinkTextFormatter
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordRegistry
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
        """Get Japanese prefix for scope. Uses CardTextResources."""
        return CardTextResources.get_scope_text(scope)
    
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
                    inferred_label = VariableLinkTextFormatter.infer_output_value_label(command_for_text)
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

        scope_text = CardTextResources.get_scope_text(scope)
        if not scope_text:
            return trigger_text

        # Strip trailing "の" for consistent composition
        scope_text = scope_text.rstrip("の")

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

        # Map CommandDef fields to Action-like dict to reuse _format_action logic where possible
        # Robustly pick command type from either 'type' or legacy 'name'
        cmd_type = command.get("type") or command.get("name") or "NONE"

        # Use a copy of command to avoid modifying the input dictionary in place
        command_copy = command.copy()

        # Mapping CommandType to ActionType logic where applicable
        original_cmd_type = cmd_type

        # Check the new formatter registry first to bypass the legacy proxy conversion
        from dm_toolkit.gui.editor.formatters.command_registry import CommandFormatterRegistry
        import dm_toolkit.gui.editor.formatters.draw_discard_formatters # Initialize registry for these types
        import dm_toolkit.gui.editor.formatters.zone_formatters # Initialize registry for zone types
        import dm_toolkit.gui.editor.formatters.magic_summon_formatters # Initialize registry for magic and summon types

        formatter_cls = CommandFormatterRegistry.get_formatter(original_cmd_type)
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
        if cmd_type == "ADD_KEYWORD":
            # Ensure mutation_kind is mapped to str_val for text generation
            if not command_copy.get("str_val") and command_copy.get("mutation_kind"):
                command_copy["str_val"] = command_copy["mutation_kind"]

        # Construct proxy action object
        # Normalize common input-link fields from various sources
        input_value_key = command.get("input_value_key") or command.get("input_link") or ""
        input_value_usage = command.get("input_value_usage") or command.get("input_usage") or ""

        action_proxy = {
            "type": cmd_type,
            "scope": command_copy.get("target_group", "NONE"),
            "filter": command_copy.get("target_filter") or command_copy.get("filter") or {},
            "value1": get_command_amount(command_copy, default=0),
            "value2": command_copy.get("val2") or command_copy.get("value2", 0),
            "optional": command_copy.get("optional", False),
            "up_to": command_copy.get("up_to", False),
            # Prefer the normalized key, but accept legacy key if present.
            # IMPORTANT: str_param (from UI) is always mapped to str_val for logic usage.
            "str_val": command_copy.get("str_param") or command_copy.get("str_val", ""),
            "input_value_key": input_value_key,
            "input_value_usage": input_value_usage,
            "from_zone": command_copy.get("from_zone", ""),
            "to_zone": command_copy.get("to_zone", ""),
            "original_to_zone": command_copy.get("original_to_zone", ""),
            "mutation_kind": command_copy.get("mutation_kind", ""),
            "destination_zone": command_copy.get("to_zone") or command_copy.get("destination_zone", ""), # For MOVE_CARD mapping
            "result": command_copy.get("result") or command_copy.get("str_param", ""), # For GAME_RESULT, map properly
            "is_mega_last_burst": ctx.has_mega_last_burst,  # Pass mega_last_burst flag for CAST_SPELL detection
            "duration": command_copy.get("duration", ""),
            "cost": command_copy.get("cost"),
            "use_mana_from": command_copy.get("use_mana_from"),
        }

        # Extra passthrough fields for complex/structured commands
        if "options" in command_copy:
            action_proxy["options"] = command_copy.get("options")
        if "flags" in command_copy:
            action_proxy["flags"] = command_copy.get("flags")
        if "look_count" in command_copy:
            action_proxy["look_count"] = command_copy.get("look_count")
        if "add_count" in command_copy:
            action_proxy["add_count"] = command_copy.get("add_count")
        if "rest_zone" in command_copy:
            action_proxy["rest_zone"] = command_copy.get("rest_zone")
        if "max_cost" in command_copy:
            action_proxy["max_cost"] = command_copy.get("max_cost")
        if "token_id" in command_copy:
            action_proxy["token_id"] = command_copy.get("token_id")
        if "play_flags" in command_copy:
            action_proxy["play_flags"] = command_copy.get("play_flags")
        if "select_count" in command_copy:
            action_proxy["select_count"] = command_copy.get("select_count")
        # 再発防止: explicit_self は ADD_KEYWORD でターゲットを「このカード」に固定するため passthrough 必須
        if "explicit_self" in command_copy:
            action_proxy["explicit_self"] = command_copy.get("explicit_self")
        if "_input_value_label" in command_copy:
            action_proxy["_input_value_label"] = command_copy.get("_input_value_label")

        # Pass through IF/IF_ELSE/ELSE control flow fields
        if "if_true" in command_copy:
            action_proxy["if_true"] = command_copy.get("if_true")
        if "if_false" in command_copy:
            action_proxy["if_false"] = command_copy.get("if_false")
        if "condition" in command_copy:
            action_proxy["condition"] = command_copy.get("condition")
        # IF commands may use target_filter for condition
        if cmd_type == "IF" and "target_filter" in command_copy:
            if "condition" not in action_proxy or not action_proxy["condition"]:
                action_proxy["target_filter"] = command_copy.get("target_filter")

        # Some templates expect source_zone rather than from_zone
        action_proxy["source_zone"] = command_copy.get("from_zone") or command_copy.get("source_zone", "")

        # Specific Adjustments
        # Specific Adjustments: handle legacy original_cmd_type values via a mapping
        def _handle_original_cmd(cmd_type: str) -> Any:
            # Handlers may return a string to short-circuit, or None to continue
            if cmd_type == "MANA_CHARGE":
                if action_proxy["scope"] == "NONE":
                    action_proxy["type"] = "ADD_MANA"
                else:
                    action_proxy["type"] = "SEND_TO_MANA"
                return None

            if cmd_type == "MEASURE_COUNT":
                action_proxy["type"] = "COUNT_CARDS"
                return None

            if cmd_type == "SHIELD_TRIGGER":
                return "S・トリガー"

            if cmd_type == "QUERY":
                query_mode = command_copy.get("str_param") or command_copy.get("query_mode") or ""
                action_proxy["query_mode"] = query_mode
                if query_mode and query_mode != "CARDS_MATCHING_FILTER":
                    action_proxy["str_param"] = query_mode
                    action_proxy["str_val"] = query_mode
                return None

            if cmd_type == "LOOK_AND_ADD":
                if "look_count" in command_copy and command_copy.get("look_count") is not None:
                    action_proxy["value1"] = command_copy.get("look_count")
                if "add_count" in command_copy and command_copy.get("add_count") is not None:
                    action_proxy["value2"] = command_copy.get("add_count")
                rz = command_copy.get("rest_zone") or command_copy.get("destination_zone") or command_copy.get("to_zone")
                if rz:
                    action_proxy["rest_zone"] = rz
                    action_proxy["destination_zone"] = rz
                return None

            if cmd_type == "MEKRAID":
                max_cost_src = command_copy.get("max_cost")
                if max_cost_src is None and "target_filter" in command_copy:
                    # 再発防止: target_filter が明示的に None の場合も .get() が呼べるよう or {} でガード
                    max_cost_src = (command_copy.get("target_filter") or {}).get("max_cost")
                if max_cost_src is not None and not isinstance(max_cost_src, dict):
                    action_proxy["value1"] = max_cost_src
                if "look_count" in command_copy and command_copy.get("look_count") is not None:
                    action_proxy["look_count"] = command_copy.get("look_count")
                if "rest_zone" in command_copy and command_copy.get("rest_zone") is not None:
                    action_proxy["rest_zone"] = command_copy.get("rest_zone")
                return None

            if cmd_type == "SUMMON_TOKEN":
                if "token_id" in command_copy and command_copy.get("token_id") is not None:
                    action_proxy["str_val"] = command_copy.get("token_id")
                return None

            if cmd_type == "PLAY_FROM_ZONE":
                if not action_proxy["source_zone"]:
                    action_proxy["source_zone"] = command_copy.get("from_zone", "")
                max_cost = command_copy.get("max_cost")
                if max_cost is None and "target_filter" in command_copy:
                    # 再発防止: target_filter が明示的に None の場合も .get() が呼べるよう or {} でガード
                    max_cost = (command_copy.get("target_filter") or {}).get("max_cost")
                if max_cost is not None and not isinstance(max_cost, dict):
                    action_proxy["value1"] = max_cost
                return None

            if cmd_type == "SELECT_NUMBER" or cmd_type == "DECLARE_NUMBER":
                action_proxy["value1"] = command_copy.get("min_value", 1)
                action_proxy["value2"] = command_copy.get("amount", 6)
                return None

            if cmd_type == "CHOICE":
                flags = command_copy.get("flags", []) or []
                if isinstance(flags, list) and "ALLOW_DUPLICATES" in flags:
                    action_proxy["optional"] = True
                action_proxy["value1"] = command_copy.get("amount", 1)
                if command_copy.get("target_filter"):
                    action_proxy["target_filter"] = command_copy["target_filter"]
                return None

            if cmd_type == "SHUFFLE_DECK":
                return None

            if cmd_type == "REGISTER_DELAYED_EFFECT":
                action_proxy["str_val"] = command_copy.get("str_param") or command_copy.get("str_val", "")
                return None

            if cmd_type == "COST_REFERENCE":
                action_proxy["ref_mode"] = command_copy.get("ref_mode")
                return None

            return None

        # Invoke handler for legacy original command types
        short_circuit = _handle_original_cmd(original_cmd_type)
        if isinstance(short_circuit, str):
            return short_circuit

        return cls._format_action(action_proxy, ctx)

    @classmethod
    def _format_logic_command(cls, atype: str, action: Dict[str, Any], ctx: TextGenerationContext) -> str:
        """Handle conditional logic commands (IF, IF_ELSE, ELSE)."""
        cond_detail = action.get("condition", {}) or action.get("target_filter", {})
        cond_text = ""

        # Use a mapping of cond_type -> handler to reduce branching and centralize formatting.
        def _handle_opponent_draw_count(d):
            val = d.get("value", 0)
            return f"相手がカードを{val}枚目以上引いたなら"

        def _handle_compare_stat(d):
            key = d.get("stat_key", "")
            op = d.get("op", "=")
            val = d.get("value", 0)
            stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ""))
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
            else:
                op_text = f"{val}{unit}"
            return f"自分の{stat_name}が{op_text}なら"

        def _handle_shield_count(d):
            val = d.get("value", 0)
            op = d.get("op", ">=")
            op_text = "以上" if op == ">=" else "以下" if op == "<=" else ""
            if op == "=":
                op_text = ""
            return f"自分のシールドが{val}つ{op_text}なら"

        def _handle_compare_input(d, action_local):
            val = d.get("value", 0)
            op = d.get("op", ">=")
            input_key = action_local.get("input_value_key", "")
            input_desc_map = {
                "spell_count": "墓地の呪文の数",
                "card_count": "カードの数",
                "creature_count": "クリーチャーの数",
                "element_count": "エレメントの数"
            }
            input_desc = input_desc_map.get(input_key, input_key if input_key else "入力値")
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
            else:
                op_text = f"{val}"
            return f"{input_desc}が{op_text}なら"

        def _handle_civ_match(d):
            return "マナゾーンに同じ文明があれば"

        def _handle_played_without_mana(d):
            return "指定した対象をコストを支払わずに出していれば"

        def _handle_mana_civ_count(d):
            val = d.get("value", 0)
            op = d.get("op", ">=")
            op_text = "以上" if op == ">=" else "以下" if op == "<=" else "と同じ" if op == "=" else ""
            return f"自分のマナゾーンにある文明の数が{val}{op_text}なら"

        COND_HANDLERS = {
            "OPPONENT_DRAW_COUNT": lambda d: _handle_opponent_draw_count(d),
            "COMPARE_STAT": lambda d: _handle_compare_stat(d),
            "SHIELD_COUNT": lambda d: _handle_shield_count(d),
            "COMPARE_INPUT": lambda d: _handle_compare_input(d, action),
            "CIVILIZATION_MATCH": lambda d: _handle_civ_match(d),
            "PLAYED_WITHOUT_MANA_TARGET": lambda d: _handle_played_without_mana(d),
            "MANA_CIVILIZATION_COUNT": lambda d: _handle_mana_civ_count(d),
        }

        if isinstance(cond_detail, dict):
            cond_type = cond_detail.get("type", "NONE")
            handler = COND_HANDLERS.get(cond_type)
            if handler:
                try:
                    cond_text = handler(cond_detail)
                except Exception:
                    cond_text = ""

        if not cond_text and atype != "ELSE":
            cond_text = "もし条件を満たすなら"

        if atype == "IF":
             if_true_cmds = action.get("if_true", [])
             if_true_texts = []
             for cmd in if_true_cmds:
                 if isinstance(cmd, dict):
                     cmd_text = cls._format_command(cmd, ctx)
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
                    cmd_text = cls._format_command(cmd, ctx)
                    if cmd_text:
                        if_true_texts.append(cmd_text)

            if_false_texts = []
            for cmd in if_false_cmds:
                if isinstance(cmd, dict):
                    cmd_text = cls._format_command(cmd, ctx)
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
    def _format_buffer_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int) -> str:
        """Handle buffer-related commands."""
        # 再発防止: val1 が文字列や float で渡される場合がある。
        #   int 比較で TypeError を起こさないよう先頭で安全に変換する。
        try:
            val1 = int(val1)
        except (TypeError, ValueError):
            val1 = 0
        if atype == "LOOK_TO_BUFFER":
             src_zone = tr(action.get("from_zone", "DECK"))
             amt = val1 if val1 > 0 else 1
             return f"{src_zone}から{amt}枚を見る。"

        elif atype == "REVEAL_TO_BUFFER":
             src_zone = tr(action.get("from_zone", "DECK"))
             amt = val1 if val1 > 0 else 1
             return f"{src_zone}から{amt}枚を表向きにしてバッファに置く。"

        elif atype == "SELECT_FROM_BUFFER":
             # 再発防止: action_proxy は target_filter を "filter" キーにマッピングする。
             #   テンプレート: "見た_{文明}の_{タイプ}_{量}を選ぶ。"
             #   各パーツはフィルターと量から自律的に許決される。
             filter_def = action.get("filter") or action.get("target_filter") or {}
             civs = filter_def.get("civilizations", []) if filter_def else []
             types = filter_def.get("types", []) if filter_def else []
             races = filter_def.get("races", []) if filter_def else []
             # 文明部分: "水の"
             civ_part = ""
             if civs:
                 civ_part = "/".join(CardTextResources.get_civilization_text(c) for c in civs) + "の"
             # タイプ部分: "クリーチャー" / "呪文" / "カード"
             if races:
                 type_part = "/".join(races)
             elif "ELEMENT" in types:
                 type_part = "エレメント"
             elif "SPELL" in types and "CREATURE" not in types:
                 type_part = "呪文"
             elif "CREATURE" in types:
                 type_part = "クリーチャー"
             elif types:
                 type_part = "/".join(tr(t) for t in types if t)
             else:
                 type_part = "カード"
             # 量部分: "すべて" / "N枚"
             if val1 <= 0:
                 qty_part = "すべて"
             else:
                 qty_part = f"{val1}枚"
             return f"見た{civ_part}{type_part}{qty_part}を選ぶ。"

        elif atype == "PLAY_FROM_BUFFER":
             target_str, unit = cls._resolve_target(action, ctx.is_spell)
             return f"選んだカード（{target_str}）を使う。"

        elif atype == "MOVE_BUFFER_TO_ZONE":
             # 再発防止: target_filter 有無に関わらず「選び」を明示して移動文面を統一する。
             #   フィルターなし=SELECT_FROM_BUFFER で設定した $buffer_select を使う。
             to_zone = tr(action.get("to_zone", "HAND"))
             filter_def = action.get("filter") or action.get("target_filter") or {}
             civs = filter_def.get("civilizations", []) if filter_def else []
             types = filter_def.get("types", []) if filter_def else []
             races = filter_def.get("races", []) if filter_def else []
             has_filter = bool(civs or types or races)
             if has_filter:
                 # 暗黙抽出でも文面は明示選択で揃える
                 civ_part = ""
                 if civs:
                     civ_part = "/".join(CardTextResources.get_civilization_text(c) for c in civs) + "の"
                 if races:
                     type_part = "/".join(races)
                 elif "ELEMENT" in types:
                     type_part = "エレメント"
                 elif "SPELL" in types and "CREATURE" not in types:
                     type_part = "呪文"
                 elif "CREATURE" in types:
                     type_part = "クリーチャー"
                 elif types:
                     type_part = "/".join(tr(t) for t in types if t)
                 else:
                     type_part = "カード"
                 qty = f"{val1}枚" if val1 > 0 else "すべて"
                 return f"見た{civ_part}{type_part}{qty}を選び、{to_zone}に置く。"
             # インタラクティブ選択テキスト（SELECT_FROM_BUFFER と組み合わせ）
             if val1 > 0:
                 return f"選んだカードを{val1}枚{to_zone}に置く。"
             return f"選んだカードをすべて{to_zone}に置く。"

        elif atype == "MOVE_BUFFER_REMAIN_TO_ZONE":
             to_zone = tr(action.get("to_zone", "DECK_BOTTOM"))
             return f"残りを{to_zone}に置く。"

        return ""

    @classmethod
    def _format_special_effect_command(cls, atype: str, action: Dict[str, Any], is_spell: bool, val1: int, target_str: str, unit: str) -> str:
        """Handle special effect commands (MEKRAID, FRIEND_BURST, MUTATE, etc)."""

        if atype == "MEKRAID":
            val2 = action.get("value2", 3) # Look count
            select_count = action.get("select_count", 1) # Number to summon
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
            # Fallback: try to extract race from filter if str_val is missing
            if not str_val:
                races = action.get("filter", {}).get("races", [])
                if races:
                    str_val = races[0]

            return f"＜{str_val}＞のフレンド・バースト（このクリーチャーが出た時、自分の他の{str_val}・クリーチャーを1体タップしてもよい。そうしたら、このクリーチャーの呪文側をバトルゾーンに置いたまま、コストを支払わずに唱える。）"

        elif atype == "APPLY_MODIFIER":
            # APPLY_MODIFIER uses str_param to indicate effect id (from schema)
            str_param = action.get('str_param') or action.get('str_val') or action.get('mutation_kind') or ''
            duration_key = action.get('duration') or action.get('input_value_key', '')

            # Determine if target is linked via input
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

            # Effect application is not keyword granting.
            # Build generic structure: 「(対象)を(数)体は、(期間)まで、そのクリーチャーに(効果)を与える。」
            effect_text = CardTextResources.get_keyword_text(str_param) if str_param else "（効果）"
            if isinstance(effect_text, str):
                effect_text = effect_text.strip() or "（効果）"

            # Amount comes from command 'amount' (value1 in proxy) in most cases.
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
            str_val = action.get("str_val") or action.get("str_param", "")
            duration_key = action.get("duration") or action.get("input_value_key", "")

            # Determine if target is linked via input
            input_key = action.get("input_value_key", "")
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            is_target_linked = bool(input_key) and (not input_usage or input_usage == "TARGET")

            duration_text = ""
            if duration_key:
                # Verify it is a valid duration key
                trans = CardTextResources.get_duration_text(duration_key)
                if trans and trans != duration_key:
                    duration_text = trans + "、"
                elif duration_key in CardTextResources.DURATION_TRANSLATION:
                    duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

            keyword = CardTextResources.get_keyword_text(str_val)

            # Explicit "this card" target override
            if action.get("explicit_self"):
                target_str = "このカード"

            # If the filter explicitly targets shields, prefer generic "カード" phrasing
            # to match expected UI wording (e.g. シールドに付与 -> カードに付与)
            filt = action.get("filter") or action.get("target_filter") or {}
            if isinstance(filt, dict) and "zones" in filt and filt.get("zones"):
                if "SHIELD_ZONE" in filt.get("zones") or "SHIELD" in filt.get("zones"):
                    target_str = "カード"

            amt = action.get('amount')
            # 再発防止: action_proxy に 'amount' キーはなく 'value1' に格納されるため、
            # val1（=action["value1"]）を優先する。amount=0 は「すべて」、amount>0 は「N体選び」。
            if amt is None:
                amt = val1 if isinstance(val1, int) else 0
            return cls._format_keyword_grant_text(target_str, str_val, keyword, duration_text, amt, skip_selection=is_target_linked)

        elif atype == "MUTATE":
             mkind = action.get("mutation_kind", "")
             # val1 is amount
             str_param = action.get("str_param") or action.get("str_val", "")

             # Determine if target is linked via input
             input_key = action.get("input_value_key", "")
             input_usage = action.get("input_value_usage") or action.get("input_usage")
             is_target_linked = bool(input_key) and (not input_usage or input_usage == "TARGET")

             # Duration handling
             duration_key = action.get("duration") or action.get("input_value_key", "")

             duration_text = ""
             if duration_key:
                 # Verify it is a valid duration key
                 trans = CardTextResources.get_duration_text(duration_key)
                 if trans and trans != duration_key:
                     duration_text = trans + "、"
                 elif duration_key in CardTextResources.DURATION_TRANSLATION:
                     duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"

             # Use centralized class-level MUTATE_KIND_HANDLERS to reduce branching
             # Coerce legacy string mutation_kind to Enum when possible, then lookup.
             lookup_key = mkind
             if isinstance(mkind, str):
                 if hasattr(consts.MutationKind, mkind):
                     try:
                         lookup_key = getattr(consts.MutationKind, mkind)
                     except Exception:
                         lookup_key = mkind
             handler = cls.MUTATE_KIND_HANDLERS.get(lookup_key)
             if handler:
                 try:
                    return handler(cls, target_str, val1, unit, duration_text, str_param, is_target_linked)
                 except Exception:
                    return f"状態変更({tr(mkind)}): {target_str} (値:{val1})"

             # Default fallback
             return f"状態変更({tr(mkind)}): {target_str} (値:{val1})"

        elif atype == "SUMMON_TOKEN":
             token_id = action.get("str_val", "")
             count = val1 if val1 > 0 else 1

             # Try to resolve token name if possible (assuming token_id is a key or name)
             token_name = "トークン"
             if token_id:
                 translated = tr(token_id)
                 # Heuristic: If translation returns the key (same as input) and it looks like a system ID
                 # (uppercase with underscores), fallback to generic "Token".
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
    def _format_game_action_command(cls, atype: str, action: Dict[str, Any], val1: int, val2: int, target_str: str, unit: str, input_key: str, input_usage: str, ctx: TextGenerationContext) -> str:
        """Handle general game action commands (SEARCH, SHIELD, BATTLE, etc)."""

        # 再発防止: target_group (editor新形式) を scope (legacy) より優先する。同一関数内で混在させないこと。
        scope = action.get("target_group") or action.get("scope", "NONE")

        # Map of specific action handlers to reduce branching and make extensions easier.
        def _handle_look_and_add():
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

        def _handle_search_deck():
            dest_zone = action.get("destination_zone", "HAND")
            if not dest_zone:
                dest_zone = "HAND"
            zone_str = CardTextResources.get_zone_text(dest_zone)
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

        ACTION_HANDLER_MAP = {
            "LOOK_AND_ADD": _handle_look_and_add,
            "SEARCH_DECK": _handle_search_deck,
            "APPLY_MODIFIER": (lambda: cls._format_special_effect_command("APPLY_MODIFIER", action, is_spell, val1, target_str, unit)),
            "MUTATE": (lambda: cls._format_special_effect_command("MUTATE", action, is_spell, val1, target_str, unit)),
            "FRIEND_BURST": (lambda: cls._format_special_effect_command("FRIEND_BURST", action, is_spell, val1, target_str, unit)),
            "SUMMON_TOKEN": (lambda: cls._format_special_effect_command("SUMMON_TOKEN", action, is_spell, val1, target_str, unit)),
            "REGISTER_DELAYED_EFFECT": (lambda: cls._format_special_effect_command("REGISTER_DELAYED_EFFECT", action, is_spell, val1, target_str, unit)),
            "PUT_CREATURE": None,  # placeholder, actual handler defined below
            # Additional handlers registered to reduce branching
            "SHUFFLE_DECK": None,
            "BOOST_MANA": None,
            "BREAK_SHIELD": None,
            "ADD_SHIELD": None,
            "DRAW_CARD": None,
            "DISCARD": None,
            "REVEAL_CARDS": None,
            "COUNT_CARDS": None,
            "TRANSITION": (lambda: (
                (lambda tpl: tpl.replace("{target}", target_str).replace("{amount}", str(val1)).replace("{unit}", unit))
                (cls._format_zone_move_command("TRANSITION", action, is_spell, val1, target_str))
            )),
            "MOVE_CARD": (lambda: (
                (lambda tpl: tpl.replace("{target}", target_str).replace("{amount}", str(val1)).replace("{unit}", unit))
                (cls._format_zone_move_command("MOVE_CARD", action, is_spell, val1, target_str))
            )),
        }

        handler = ACTION_HANDLER_MAP.get(atype)
        if handler:
            try:
                return handler()
            except Exception:
                # Fall through to legacy handlers on failure
                pass

        # Define PUT_CREATURE handler and register it to the map to reduce branching.
        def _handle_put_creature():
            # 再発防止: from_zone/source_zone を文頭で出す場合、filter.zones を残すと
            # 「手札から手札の〜」のように重複するため、ターゲット解決前に取り除く。
            action_local = action.copy()
            filter_local = (action.get("filter") or action.get("target_filter") or {}).copy()
            count = val1 if val1 > 0 else 1
            filter_zones = filter_local.get("zones", [])
            src_text = ""

            # Prioritize explicit from_zone
            from_z = action.get("from_zone") or action.get("source_zone")
            if from_z and from_z != "NONE":
                src_text = CardTextResources.get_zone_text(from_z) + "から"
                filter_local.pop("zones", None)
            elif filter_zones:
                znames = [CardTextResources.get_zone_text(z) for z in filter_zones]
                src_text = "または".join(znames) + "から"
                filter_local["zones"] = []

            action_local["filter"] = filter_local
            action_local["target_filter"] = filter_local
            put_target_str, put_unit = cls._resolve_target(action_local)

            return f"{src_text}{put_target_str}を{count}{put_unit}バトルゾーンに出す。"

        # Register handler into the action map for future fast-path lookups.
        ACTION_HANDLER_MAP["PUT_CREATURE"] = _handle_put_creature

        # Define additional fast-path handlers mirroring the legacy elif blocks
        def _handle_shuffle_deck():
            return "山札をシャッフルする。"

        def _handle_boost_mana():
            count = val1 if val1 > 0 else 1
            return f"自分のマナを{count}つ増やす。"

        def _handle_break_shield():
            count = val1 if val1 > 0 else 1
            tgt = target_str
            if tgt in ("", "カード", "自分のカード", "それ"):
                tgt = "シールド"
            if not action.get("target_group") and not action.get("scope") or (action.get("target_group") or action.get("scope")) == "NONE":
                if "相手" not in tgt:
                    tgt = "相手の" + tgt
            return f"{tgt}を{count}つブレイクする。"

        def _handle_add_shield():
            amt = val1 if val1 > 0 else 1
            if "山札" in target_str or target_str == "カード":
                return f"山札の上から{amt}枚をシールド化する。"
            return f"{target_str}を{amt}つシールド化する。"

        def _handle_shield_burn():
            amt = val1 if val1 > 0 else 1
            return f"相手のシールドを{amt}つ選び、墓地に置く。"

        ACTION_HANDLER_MAP["SHUFFLE_DECK"] = _handle_shuffle_deck
        ACTION_HANDLER_MAP["BOOST_MANA"] = _handle_boost_mana
        ACTION_HANDLER_MAP["BREAK_SHIELD"] = _handle_break_shield
        ACTION_HANDLER_MAP["ADD_SHIELD"] = _handle_add_shield
        ACTION_HANDLER_MAP["SHIELD_BURN"] = _handle_shield_burn
        # Draw / Discard handlers
        def _handle_draw_card():
            # Prefer explicit up_to flag; default count fallback
            up_to = bool(action.get('up_to', False))
            has_input_key = bool(action.get("input_value_key") or action.get("input_link") or input_key)
            optional_draw = bool(action.get('optional', False))
            cnt = val1 if val1 > 0 else 1
            if has_input_key:
                linked_count = VariableLinkTextFormatter.format_linked_count_token(action, "その同じ枚数")
                if up_to:
                    text = f"カードを{linked_count}まで引く。"
                else:
                    text = f"カードを{linked_count}引く。"
                if optional_draw:
                    return text[:-2] + "いてもよい。"
                return text
            if up_to:
                text = f"最大{cnt}枚引く。"
            else:
                text = f"{cnt}枚引く。"
            if optional_draw:
                return text[:-2] + "いてもよい。"
            return text

        def _handle_discard():
            up_to = bool(action.get('up_to', False))
            cnt = val1 if val1 > 0 else 1
            # Prefer explicit target if provided, otherwise assume hand
            tgt = target_str
            if not tgt or tgt == "カード":
                tgt = "手札"
            if up_to:
                return f"{tgt}を最大{cnt}{unit}捨てる。"
            return f"{tgt}を{cnt}{unit}捨てる。"

        ACTION_HANDLER_MAP["DRAW_CARD"] = _handle_draw_card
        ACTION_HANDLER_MAP["DISCARD"] = _handle_discard
        # Reveal / Count handlers
        def _handle_reveal_cards():
            deck_owner = "相手の" if (action.get("target_group") or action.get("scope", "NONE")) in ["OPPONENT", "PLAYER_OPPONENT"] else ""
            # Prefer explicit action input key when present, fallback to the function param
            has_input_key = bool(action.get("input_value_key") or action.get("input_link") or input_key)
            if has_input_key:
                return f"{deck_owner}山札の上から、その数だけ表向きにする。"
            return f"{deck_owner}山札の上から{val1}枚を表向きにする。"

        def _handle_count_cards():
            if not target_str or target_str == "カード":
                return f"({tr('COUNT_CARDS')})"
            return f"{target_str}の数を数える。"

        ACTION_HANDLER_MAP["REVEAL_CARDS"] = _handle_reveal_cards
        ACTION_HANDLER_MAP["COUNT_CARDS"] = _handle_count_cards
        # LOCK / restriction / stat handlers
        def _resolve_player_scope_text() -> str:
            # 再発防止: target_group 優先で対象プレイヤー文言を解決し、LOCK/制限系で表記ゆれを防ぐ。
            if scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                return "相手"
            if scope in ["PLAYER_SELF", "SELF"]:
                return "自分"
            if scope == "ALL_PLAYERS":
                return "すべてのプレイヤー"
            resolved, _ = cls._resolve_target(action, is_spell)
            return resolved

        def _resolve_duration_text() -> str:
            # 再発防止: duration は文字列キー優先で扱い、value1 による誤表示を避ける。
            duration_key = action.get("duration", "") or ""
            if duration_key:
                return CardTextResources.get_duration_text(duration_key)
            return f"{val1}ターン" if val1 > 0 else "このターン"

        def _handle_lock_spell():
            return f"{_resolve_player_scope_text()}は{_resolve_duration_text()}の間、呪文を唱えられない。"

        def _handle_spell_and_action_restrictions():
            if atype == "SPELL_RESTRICTION":
                filt = action.get("filter", {}) or {}
                exact_cost = filt.get("exact_cost") if isinstance(filt, dict) else None
                if input_key:
                    action_text = "入力値と同じコストの呪文を唱えられない"
                elif exact_cost is not None:
                    action_text = f"コスト{exact_cost}の呪文を唱えられない"
                else:
                    action_text = "呪文を唱えられない"
            elif atype == "CANNOT_PUT_CREATURE":
                action_text = "クリーチャーを出せない"
            elif atype == "CANNOT_SUMMON_CREATURE":
                action_text = "クリーチャーを召喚できない"
            else:
                action_text = "攻撃できない"

            return f"{_resolve_player_scope_text()}は{_resolve_duration_text()}の間、{action_text}。"

        def _handle_stat():
            key = action.get('stat') or action.get('str_param') or action.get('str_val')
            amount = action.get('amount', action.get('value1', 0))
            if key:
                stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(key), (None, None))
                if stat_name:
                    return f"統計更新: {stat_name} += {amount}"
            return f"統計更新: {tr(str(key))} += {amount}"

        def _handle_get_game_stat():
            key = action.get('str_val') or action.get('result') or ''
            stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(key, (None, None))
            if stat_name:
                if ctx.sample is not None:
                    try:
                        val = cls._compute_stat_from_sample(key, ctx.sample)
                        if val is not None:
                            return f"{stat_name}（例: {val}{stat_unit}）"
                    except Exception:
                        pass
                return f"{stat_name}"
            return f"（{tr(key)}を参照）"

        def _handle_flow():
            ftype = action.get("flow_type") or action.get("str_val", "")
            flow_value = action.get("value1", 0)

            if ftype == "PHASE_CHANGE":
                phase_name = CardTextResources.PHASE_MAP.get(flow_value, str(flow_value))
                return f"{phase_name}フェーズへ移行する。"
            if ftype == "TURN_CHANGE":
                return "ターンを終了する。"
            if ftype == "SET_ACTIVE_PLAYER":
                return "手番を変更する。"
            return f"進行制御({tr(ftype)}): {flow_value}"

        def _handle_game_result():
            result = action.get("result", "") or action.get("str_val") or action.get("str_param", "")
            return f"ゲームを終了する（{tr(result)}）。"

        def _handle_declare_number():
            min_val = action.get("value1", 1)
            max_val = action.get("value2", 10)
            if min_val == 0:
                min_val = action.get("min_value", 1)
            if max_val == 0:
                max_val = action.get("amount", 10)
            return f"数字を1つ宣言する（{min_val}～{max_val}）。"

        def _handle_decide():
            selected_option = action.get("selected_option_index")
            if isinstance(selected_option, int) and selected_option >= 0:
                return f"選択肢{selected_option}を確定する。"
            indices = action.get("selected_indices") or []
            if isinstance(indices, list) and indices:
                return f"選択（{indices}）を確定する。"
            return "選択を確定する。"

        def _handle_declare_reaction():
            if action.get("pass"):
                return "リアクション: パスする。"
            reaction_index = action.get("reaction_index")
            if isinstance(reaction_index, int):
                return f"リアクションを宣言する（インデックス {reaction_index}）。"
            return "リアクションを宣言する。"

        def _handle_attach():
            return f"{target_str}をカードの下に重ねる。"

        def _handle_move_to_under_card():
            amount = val1 if val1 > 0 else 1
            if amount == 1:
                return f"{target_str}をカードの下に重ねる。"
            return f"{target_str}を{amount}{unit}カードの下に重ねる。"

        def _handle_reset_instance():
            return f"{target_str}の状態を初期化する（効果を無視する）。"

        def _handle_select_target():
            amount = val1 if val1 > 0 else 1
            return f"{target_str}を{amount}{unit}選ぶ。"

        ACTION_HANDLER_MAP["LOCK_SPELL"] = _handle_lock_spell
        ACTION_HANDLER_MAP["SPELL_RESTRICTION"] = _handle_spell_and_action_restrictions
        ACTION_HANDLER_MAP["CANNOT_PUT_CREATURE"] = _handle_spell_and_action_restrictions
        ACTION_HANDLER_MAP["CANNOT_SUMMON_CREATURE"] = _handle_spell_and_action_restrictions
        ACTION_HANDLER_MAP["PLAYER_CANNOT_ATTACK"] = _handle_spell_and_action_restrictions
        ACTION_HANDLER_MAP["STAT"] = _handle_stat
        ACTION_HANDLER_MAP["GET_GAME_STAT"] = _handle_get_game_stat
        ACTION_HANDLER_MAP["FLOW"] = _handle_flow
        ACTION_HANDLER_MAP["GAME_RESULT"] = _handle_game_result
        ACTION_HANDLER_MAP["DECLARE_NUMBER"] = _handle_declare_number
        ACTION_HANDLER_MAP["DECIDE"] = _handle_decide
        ACTION_HANDLER_MAP["DECLARE_REACTION"] = _handle_declare_reaction
        ACTION_HANDLER_MAP["ATTACH"] = _handle_attach
        ACTION_HANDLER_MAP["MOVE_TO_UNDER_CARD"] = _handle_move_to_under_card
        ACTION_HANDLER_MAP["RESET_INSTANCE"] = _handle_reset_instance
        ACTION_HANDLER_MAP["SELECT_TARGET"] = _handle_select_target
        # Zone movement handlers
        def _handle_transition():
            return cls._format_zone_move_command("TRANSITION", action, ctx.is_spell, val1, target_str)

        def _handle_move_card():
            return cls._format_zone_move_command("MOVE_CARD", action, ctx.is_spell, val1, target_str)

        ACTION_HANDLER_MAP["TRANSITION"] = _handle_transition
        ACTION_HANDLER_MAP["MOVE_CARD"] = _handle_move_card

        # 再発防止: 後段で追加したハンドラもここで再ディスパッチする。
        # これを省くと map 化済みコマンドでも legacy if/elif 経路に落ちて分岐削減が進まない。
        handler = ACTION_HANDLER_MAP.get(atype)
        if handler:
            try:
                return handler()
            except Exception:
                pass

        elif atype == "COST_REFERENCE":
             ref_mode = action.get("ref_mode", "")
             return f"（コスト参照: {tr(ref_mode)}）"

        elif atype == "SEND_SHIELD_TO_GRAVE":
             amt = val1 if val1 > 0 else 1
             if scope == "OPPONENT" or scope == "PLAYER_OPPONENT":
                  return f"相手のシールドを{amt}つ選び、墓地に置く。"
             return f"{target_str}を{amt}つ墓地に置く。"


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
                        parts.append(cls._format_command(a, ctx))
                    else:
                        parts.append(cls._format_action(a, ctx))
                chain_text = " ".join(parts)
                lines.append(f"> {chain_text}")
            return "\n".join(lines)

        elif atype == "QUERY":
             mode = action.get("query_mode") or action.get("str_param") or action.get("str_val") or ""
             stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(mode), (None, None))
             if stat_name:
                 base = f"{stat_name}{stat_unit}を数える。"
                 if input_key:
                     usage_label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
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
                     usage_label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                     if usage_label:
                         base += f"（{usage_label}）"
                 return base

             if str(mode) == "SELECT_OPTION":
                 sel_count = action.get("value1", action.get("amount", 1))
                 filter_txt = cls._format_filter(action.get("filter", {}))
                 if input_key:
                     usage_label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                     cnt_txt = "指定数"
                     if usage_label:
                         cnt_txt = f"入力値（{usage_label}）"
                     if filter_txt:
                         return f"{filter_txt}から{cnt_txt}選ぶ。"
                     return f"条件に合うカードから{cnt_txt}選ぶ。"

                 if filter_txt:
                     return f"{filter_txt}から{sel_count}枚選ぶ。"
                 return f"条件に合うカードから{sel_count}枚選ぶ。"

             base = f"質問: {tr(mode)}"
             if input_key:
                 usage_label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                 if usage_label:
                     base += f"（{usage_label}）"
             return base

        elif atype == "IGNORE_ABILITY":
             scope = action.get("target_group") or action.get("scope", "NONE")
             if scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                 target_str_lock = "相手"
             elif scope in ["PLAYER_SELF", "SELF"]:
                 target_str_lock = "自分"
             elif scope == "ALL_PLAYERS":
                 target_str_lock = "すべてのプレイヤー"
             else:
                 target_str_lock, _ = cls._resolve_target(action, is_spell)

             duration_key = action.get("duration", "") or ""
             duration_text = CardTextResources.get_duration_text(duration_key) if duration_key else (f"{val1}ターン" if val1 > 0 else "このターン")

             filt = action.get("filter", {}) or {}
             types = []
             if isinstance(filt, dict):
                 types = filt.get("types", []) or []
             type_txt = "カード"
             if types:
                 type_txt = "・".join([tr(t) for t in types])

             if input_key:
                 return f"{target_str_lock}のコスト入力値と同じ{type_txt}の能力は{duration_text}の間、無視される。"
             if isinstance(filt, dict) and filt.get("exact_cost") is not None:
                 return f"{target_str_lock}のコスト{filt.get('exact_cost')}の{type_txt}の能力は{duration_text}の間、無視される。"
             return f"{target_str_lock}の{type_txt}の能力は{duration_text}の間、無視される。"

        return ""

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
                            linked_count = VariableLinkTextFormatter.format_linked_count_token(action, "その同じ数")
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
                link_suffix = VariableLinkTextFormatter.format_input_link_context_suffix(action)
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
    def _format_action(cls, action: Dict[str, Any], ctx: TextGenerationContext) -> str:
        """
        INTERNAL: Format action-like dictionary to Japanese text.

        This method is now primarily used internally by _format_command to handle
        the action_proxy representation. Direct calls to this method for legacy
        Action formatting should be replaced with ActionConverter + _format_command.

        Legacy Actions are automatically converted to Commands at load time.
        """
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = CardTextResources.ACTION_MAP.get(atype, "")

        # Up-to drawing: adjust template for explicit DRAW_CARD
        if atype == 'DRAW_CARD':
            if bool(action.get('up_to', False)):
                template = "最大{value1}枚まで引く。"

        # Special-case: treat TRANSITION from DECK->HAND as DRAW_CARD for natural language
        if atype == 'TRANSITION':
            from_zone = CardTextResources.normalize_zone_name(action.get('from_zone') or action.get('fromZone') or '')
            to_zone = CardTextResources.normalize_zone_name(action.get('to_zone') or action.get('toZone') or '')
            amt = action.get('amount') or action.get('value1') or 0
            up_to = bool(action.get('up_to', False))

            # Use short alias if available (e.g., "破壊" for BATTLE->GRAVEYARD)
            alias = CardTextResources.TRANSITION_ALIASES.get((from_zone, to_zone))
            if alias:
                 # Reconstruct natural sentences based on known aliases
                 if alias == "破壊":
                      return f"{{target}}を{amt}体破壊する。" if amt > 0 else f"{{target}}をすべて破壊する。"
                 elif alias == "捨てる":
                      return f"手札を{amt}枚捨てる。" if amt > 0 else "手札をすべて捨てる。"
                 elif alias == "手札に戻す":
                      # Manually resolve vars to ensure correctness and return immediately
                      target_str, unit = cls._resolve_target(action, ctx.is_spell)
                      if up_to and amt > 0:
                          t = f"{target_str}を{amt}{unit}まで選び、手札に戻す。"
                      elif amt == 0:
                          t = f"{target_str}をすべて手札に戻す。"
                      else:
                          t = f"{target_str}を{amt}{unit}手札に戻す。"

                      # Optional conjugation
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

            # If transition represents drawing from deck to hand
            if (from_zone == 'DECK' or from_zone == '') and to_zone == 'HAND':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                if up_to:
                    return f"カードを最大{amt}枚まで引く。"
                else:
                    return f"カードを{amt}枚引く。"
            # If transition represents moving to mana zone, render as ADD_MANA
            if (from_zone == 'DECK' or from_zone == '') and to_zone == 'MANA_ZONE':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                return f"自分の山札の上から{amt}枚をマナゾーンに置く。"

        # Determine verb form (standard or optional)
        optional = action.get("optional", False)

        # Resolve dynamic target strings
        target_str, unit = cls._resolve_target(action, ctx.is_spell)

        # Parameter Substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")
        input_key = action.get("input_value_key", "")
        input_usage = action.get("input_value_usage") or action.get("input_usage")

        is_generic_selection = atype in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA", "MOVE_CARD", "TRANSITION", "DISCARD"]

        # 1. Handle Input Variable Linking (Contextual substitution)
        if input_key:
            # Usage label for linked inputs
            usage_label_suffix = ""
            if input_usage:
                label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"

            # 前のアクションの出力を参照する場合
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
                # TRANSITION用の汎用的な参照表現
                val1 = "その同じ枚数"
                # 「まで」フラグがある場合は追加
                if bool(action.get('up_to', False)):
                    val1 = "その同じ枚数まで"
                # Add usage label at the end after template is fully formed
            elif atype == "MOVE_CARD":
                # MOVE_CARDの入力リンク対応（行き先に応じた自然文）
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
                # 前回の出力枚数と同じ枚数を捨てる
                up_to_discard = bool(action.get('up_to', False))
                if up_to_discard:
                    template = f"手札をその同じ枚数まで捨てる。{usage_label_suffix}"
                else:
                    template = f"手札をその同じ枚数捨てる。{usage_label_suffix}"
            else:
                val1 = "その数"
        elif (val1 == 0 or (atype == "TRANSITION" and action.get("amount", 0) == 0)) and is_generic_selection:
             # Logic for "All"
             if atype == "DESTROY": template = "{target}をすべて破壊する。"
             elif atype == "TAP": template = "{target}をすべてタップする。"
             elif atype == "UNTAP": template = "{target}をすべてアンタップする。"
             elif atype == "RETURN_TO_HAND": template = "{target}をすべて手札に戻す。"
             elif atype == "SEND_TO_MANA": template = "{target}をすべてマナゾーンに置く。"
             elif atype == "MOVE_CARD": pass # Handled below
             elif atype == "TRANSITION": pass # Handled below
             elif atype == "DISCARD": template = "手札をすべて捨てる。"

        # Complex Action Logic
        if atype == "DISCARD":
            # Standard discard with amount
            amt = get_command_amount(action, default=val1 if val1 else 1)
            up_to_discard = bool(action.get('up_to', False))
            if amt == 0:
                template = "手札をすべて捨てる。"
            elif up_to_discard:
                template = f"手札を{amt}枚まで捨てる。"
            else:
                template = f"手札を{amt}枚捨てる。"
            return template


        elif atype == "MEKRAID" or atype == "FRIEND_BURST" or atype == "APPLY_MODIFIER" or atype == "ADD_KEYWORD" or atype == "MUTATE" or atype == "REGISTER_DELAYED_EFFECT" or atype == "SUMMON_TOKEN":
             text = cls._format_special_effect_command(atype, action, ctx.is_spell, val1, target_str, unit)
             if text: return text

        elif atype == "TRANSITION" or atype == "MOVE_CARD" or atype == "REPLACE_CARD_MOVE":
             # Zone move commands return a template that needs variable substitution
             # so we assign to template and let execution proceed.
             t = cls._format_zone_move_command(atype, action, ctx.is_spell, val1, target_str)
             if t:
                 template = t

        elif atype == "IF" or atype == "IF_ELSE" or atype == "ELSE":
            text = cls._format_logic_command(atype, action, ctx)
            if text: return text

        # Attempt to format as a general game action command
        text = cls._format_game_action_command(atype, action, val1, val2, target_str, unit, input_key, input_usage, ctx)
        if text: return text

        # Check Buffer Command (Override template if specific logic exists)
        buf = cls._format_buffer_command(atype, action, ctx.is_spell, val1)
        if buf:
            template = buf

        if not template:
            return f"({tr(atype)})"

        if atype == "GRANT_KEYWORD" or atype == "ADD_KEYWORD":
            # キーワードの翻訳を適用
            keyword = CardTextResources.get_keyword_text(str_val)
            str_val = keyword


        elif atype == "CAST_SPELL":
            # CAST_SPELL: フィルタの詳細を反映した呪文テキスト生成
            action = action.copy()
            temp_filter = action.get("filter") or action.get("target_filter") or {}
            if not isinstance(temp_filter, dict):
                temp_filter = {}
            temp_filter = temp_filter.copy()
            action["filter"] = temp_filter
            
            # Mega Last Burst detection: check for mega_last_burst flag in context or action
            is_mega_last_burst = action.get("is_mega_last_burst", False) or action.get("mega_last_burst", False)
            mega_burst_prefix = ""
            if is_mega_last_burst:
                mega_burst_prefix = "このクリーチャーがバトルゾーンから離れて、"
            cast_phrase = cls._format_cast_spell_cost_phrase(action)
            
            # Input Usage label
            usage_label_suffix = ""
            if input_key and input_usage:
                label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"
            
            # フィルタでSPELLタイプが指定されている場合、詳細なターゲット文字列を生成
            types = temp_filter.get("types", [])
            if "SPELL" in types or not types:
                # ゾーン表現（複数ゾーンは「または」で連結して『〜から』を生成）
                zones = temp_filter.get("zones", [])
                linked_cost_phrase = ""
                max_cost_def = temp_filter.get("max_cost")
                if is_input_linked(max_cost_def, usage="MAX_COST"):
                    source_token = VariableLinkTextFormatter.format_linked_count_token(action, "その数")
                    source_token = VariableLinkTextFormatter.normalize_linked_count_label(source_token)
                    linked_cost_phrase = f"{source_token}以下のコストの"
                zone_phrase = ""
                if zones:
                    zone_names = []
                    for z in zones:
                        if z == "HAND":
                            zone_names.append("手札")
                        elif z == "GRAVEYARD":
                            zone_names.append("墓地")
                        elif z == "MANA_ZONE":
                            zone_names.append("マナゾーン")
                        elif z == "BATTLE_ZONE":
                            zone_names.append("バトルゾーン")
                        else:
                            zone_names.append(tr(z))
                    if len(zone_names) == 1:
                        zone_phrase = zone_names[0] + "から"
                    else:
                        zone_phrase = "または".join(zone_names) + "から"

                # ターゲット形容（文明/コスト/パワー等）はゾーンを除いて生成して重複を避ける
                tf_no_zones = temp_filter.copy()
                if "zones" in tf_no_zones:
                    tf_no_zones["zones"] = []
                if linked_cost_phrase and "max_cost" in tf_no_zones:
                    # 再発防止: 入力リンク由来の最大コストは専用文言へ寄せ、重複表現を防ぐ。
                    del tf_no_zones["max_cost"]
                action_no_zone = action.copy()
                action_no_zone["filter"] = tf_no_zones
                target_str, unit = cls._resolve_target(action_no_zone)
                if zone_phrase and target_str.startswith("自分の"):
                    target_str = target_str[len("自分の"):]
                elif zone_phrase and target_str.startswith("相手の"):
                    target_str = target_str[len("相手の"):]

                # 最終テンプレート
                if target_str.endswith("呪文"):
                    subject_text = target_str
                elif target_str == "カード" or target_str == "":
                    subject_text = "呪文"
                elif target_str.endswith("カード"):
                    # 再発防止: 呪文タイプ指定済みの対象に「カードの呪文」を重ねない。
                    subject_text = target_str[:-3] + "呪文"
                else:
                    subject_text = f"{target_str}の呪文"
                if linked_cost_phrase and not subject_text.startswith(linked_cost_phrase):
                    subject_text = linked_cost_phrase + subject_text

                selection_target = f"{zone_phrase}{subject_text}" if zone_phrase else subject_text
                if temp_filter and not temp_filter.get("is_trigger_source"):
                    count_text = cls._format_selection_quantity(temp_filter.get("count"), unit)
                    template = f"{mega_burst_prefix}{selection_target}を{count_text}選び、{cast_phrase}。{usage_label_suffix}"
                else:
                    template = f"{mega_burst_prefix}{selection_target}を{cast_phrase}。{usage_label_suffix}"
            else:
                # SPELLタイプ以外の場合は通常のターゲット文字列
                target_str, unit = cls._resolve_target(action)
                if target_str == "" or target_str == "カード":
                    template = f"{mega_burst_prefix}カードを{cast_phrase}。{usage_label_suffix}"
                else:
                    template = f"{mega_burst_prefix}{target_str}を{cast_phrase}。{usage_label_suffix}"

        elif atype == "PLAY_FROM_ZONE":
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter

            # Input Usage label
            usage_label_suffix = ""
            if input_key and input_usage:
                label = VariableLinkTextFormatter.format_input_usage_label(input_usage)
                if label:
                    usage_label_suffix = f"（{label}）"

            if not action.get("source_zone") and "zones" in temp_filter:
                zones = temp_filter["zones"]
                if len(zones) == 1:
                    action["source_zone"] = zones[0]

            if action.get("value1", 0) == 0:
                max_cost = temp_filter.get("max_cost", MAX_COST_VALUE)
                # Handle max_cost that might be int or dict with input_link
                if is_input_linked(max_cost):
                    # If it's input-linked, don't extract a numeric value
                    # Keep max_cost in filter so _resolve_target can process it
                    pass
                elif max_cost < MAX_COST_VALUE:
                    action["value1"] = max_cost
                    if not input_key: val1 = max_cost
                    if "max_cost" in temp_filter: del temp_filter["max_cost"]
            else:
                 # If value1 is already set (e.g. from schema param), remove it from filter to avoid duplication in target_str
                 if "max_cost" in temp_filter:
                      del temp_filter["max_cost"]

            if "zones" in temp_filter: temp_filter["zones"] = []
            # 再発防止: target_group 優先。scope は後方互換。
            scope = action.get("target_group") or action.get("scope", "NONE")
            if scope in ["PLAYER_SELF", "SELF"]: action["scope"] = "NONE"

            target_str, unit = cls._resolve_target(action)
            count = temp_filter.get("count")
            count_text = ""
            if isinstance(count, int) and count > 0:
                count_text = f"{count}{unit}選び、"
            verb = "プレイする"
            types = temp_filter.get("types", [])
            if "SPELL" in types and "CREATURE" not in types:
                verb = "唱える"
            elif "CREATURE" in types:
                verb = "召喚する"

            # Check for play_flags (Play for Free)
            play_flags = action.get("play_flags")
            is_free = False
            if isinstance(play_flags, bool) and play_flags:
                is_free = True
            elif isinstance(play_flags, list) and ("FREE" in play_flags or "COST_FREE" in play_flags):
                is_free = True

            if is_free:
                verb = f"コストを支払わずに{verb}"

            # Check if using input-linked cost to avoid duplicate/unnatural cost text.
            use_linked_cost = False
            max_cost = temp_filter.get("max_cost")
            linked_cost_phrase = ""
            if is_input_linked(max_cost, usage="MAX_COST"):
                use_linked_cost = True
                source_token = VariableLinkTextFormatter.format_linked_count_token(action, "その数")
                source_token = VariableLinkTextFormatter.normalize_linked_count_label(source_token)
                linked_cost_phrase = f"{source_token}以下のコストの"

            if use_linked_cost:
                # 再発防止: 「コストその数以下の」を入力元由来の自然文へ統一する。
                if target_str.startswith("コストその数以下の"):
                    target_str = target_str.replace("コストその数以下の", "", 1)
                if linked_cost_phrase and not target_str.startswith(linked_cost_phrase):
                    target_str = linked_cost_phrase + target_str

                if action.get("source_zone"):
                    template = "{source_zone}から{target}を" + count_text + verb + f"。{usage_label_suffix}"
                else:
                    template = "{target}を" + count_text + verb + f"。{usage_label_suffix}"
            else:
                if action.get("source_zone"):
                    template = "{source_zone}からコスト{value1}以下の{target}を" + count_text + verb + f"。{usage_label_suffix}"
                else:
                    template = "コスト{value1}以下の{target}を" + count_text + verb + f"。{usage_label_suffix}"

        # Override template for DESTROY if targeting trigger source to avoid "All" or "1 body"
        if atype == "DESTROY" and action.get('filter', {}).get('is_trigger_source'):
            template = "{target}を破壊する。"

        # Destination/Source Resolution
        dest_zone = action.get("destination_zone", "")
        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        src_zone = action.get("source_zone", "")
        src_str = CardTextResources.get_zone_text(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        # Handle PLAY_FROM_ZONE with input-linked max_cost: remove "コスト{value1}以下の" since target_str includes cost info
        if atype == "PLAY_FROM_ZONE" and action.get("value1") == 0:
            max_cost = action.get("filter", {}).get("max_cost")
            if isinstance(max_cost, dict) and max_cost.get("input_value_usage") == "MAX_COST":
                # Remove the "コスト{value1}以下の" part since _resolve_target already includes it
                text = text.replace("コスト0以下の", "")

        # Handle specific replacements for TRANSITION/MUTATE
        if atype in ["TRANSITION", "MUTATE"]:
            text = text.replace("{amount}", str(val1))

        if "{filter}" in text:
             text = text.replace("{filter}", target_str)

        if "{result}" in text:
             # Handle result replacement if not done yet
             res = action.get("result", "")
             text = text.replace("{result}", tr(res))

        if atype == "COST_REDUCTION":
            if target_str == "カード" or target_str == "自分のカード":
                replacement = "この呪文" if ctx.is_spell else "このクリーチャー"
                text = text.replace("カード", replacement)
                text = text.replace("自分のカード", replacement)
            cond = action.get("condition", {})
            if cond:
                cond_text = cls._format_condition(cond)
                text = f"{cond_text}{text}"

        # Verb Conjugation for Optional Actions
        if optional:
            if text.endswith("する。"):
                text = text[:-3] + "してもよい。"
            elif text.endswith("く。"): # 引く、置く
                text = text[:-2] + "いてもよい。" # 引いてもよい
            elif text.endswith("す。"): # 戻す、出す
                text = text[:-2] + "してもよい。" # 戻してもよい
            elif text.endswith("る。"): # 見る、捨てる、唱える
                text = text[:-2] + "てもよい。"
            elif text.endswith("う。"): # 支払う
                text = text[:-2] + "ってもよい。"
            else:
                if not text.endswith("てもよい。"):
                    text = text[:-1] + "てもよい。"

        return text

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