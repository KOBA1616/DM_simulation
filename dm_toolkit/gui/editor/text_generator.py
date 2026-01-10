# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.localization import tr

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    # Maps kept for reverse lookups or specific logic, but tr() should be preferred for output
    CIVILIZATION_MAP = {
        "LIGHT": "光",
        "WATER": "水",
        "DARKNESS": "闇",
        "FIRE": "火",
        "NATURE": "自然",
        "ZERO": "ゼロ"
    }

    TYPE_MAP = {
        "CREATURE": "クリーチャー",
        "SPELL": "呪文",
        "CROSS_GEAR": "クロスギア",
        "CASTLE": "城",
        "EVOLUTION_CREATURE": "進化クリーチャー",
        "NEO_CREATURE": "NEOクリーチャー",
        "PSYCHIC_CREATURE": "サイキック・クリーチャー",
        "PSYCHIC_SUPER_CREATURE": "サイキック・スーパー・クリーチャー",
        "DRAGHEART_CREATURE": "ドラグハート・クリーチャー",
        "DRAGHEART_WEAPON": "ドラグハート・ウエポン",
        "DRAGHEART_FORTRESS": "ドラグハート・フォートレス",
        "AURA": "オーラ",
        "FIELD": "フィールド",
        "D2_FIELD": "D2フィールド",
    }

    # Backup map if tr() returns the key itself
    KEYWORD_TRANSLATION = {
        # Lowercase versions (original)
        "speed_attacker": "スピードアタッカー",
        "blocker": "ブロッカー",
        "slayer": "スレイヤー",
        "double_breaker": "W・ブレイカー",
        "triple_breaker": "T・ブレイカー",
        "world_breaker": "ワールド・ブレイカー",
        "shield_trigger": "S・トリガー",
        "evolution": "進化",
        "just_diver": "ジャストダイバー",
        "mach_fighter": "マッハファイター",
        "g_strike": "G・ストライク",
        "hyper_energy": "ハイパーエナジー",
        "shield_burn": "シールド焼却",
        "revolution_change": "革命チェンジ",
        "mekraid": "メクレイド",
        "friend_burst": "フレンド・バースト",
        "untap_in": "タップして出る",
        "meta_counter_play": "メタカウンター",
        "power_attacker": "パワーアタッカー",
        "g_zero": "G・ゼロ",
        "ex_life": "EXライフ",
        "mega_last_burst": "メガ・ラスト・バースト",
        "unblockable": "ブロックされない",
        "no_choice": "選ばれない",
        # Uppercase versions (for compatibility)
        "SPEED_ATTACKER": "スピードアタッカー",
        "BLOCKER": "ブロッカー",
        "SLAYER": "スレイヤー",
        "DOUBLE_BREAKER": "W・ブレイカー",
        "TRIPLE_BREAKER": "T・ブレイカー",
        "WORLD_BREAKER": "ワールド・ブレイカー",
        "SHIELD_TRIGGER": "S・トリガー",
        "EVOLUTION": "進化",
        "JUST_DIVER": "ジャストダイバー",
        "MACH_FIGHTER": "マッハファイター",
        "G_STRIKE": "G・ストライク",
        "HYPER_ENERGY": "ハイパーエナジー",
        "SHIELD_BURN": "シールド焼却",
        "REVOLUTION_CHANGE": "革命チェンジ",
        "MEKRAID": "メクレイド",
        "FRIEND_BURST": "フレンド・バースト",
        "UNTAP_IN": "タップして出る",
        "META_COUNTER_PLAY": "メタカウンター",
        "POWER_ATTACKER": "パワーアタッカー",
        "G_ZERO": "G・ゼロ",
        "EX_LIFE": "EXライフ",
        "MEGA_LAST_BURST": "メガ・ラスト・バースト",
        "UNBLOCKABLE": "ブロックされない",
        "NO_CHOICE": "選ばれない",
        "ATTACKER": "アタッカー",
        # Additional keywords (uppercase)
        "CANNOT_ATTACK": "攻撃できない",
        "CANNOT_BLOCK": "ブロックできない",
        "CANNOT_ATTACK_OR_BLOCK": "攻撃またはブロックできない",
        # Lowercase versions of additional keywords
        "attacker": "アタッカー"
    }

    PHASE_MAP = {
        0: "ターン開始",
        1: "ドロー",
        2: "マナ",
        3: "メイン",
        4: "攻撃",
        5: "ブロック",
        6: "ターン終了"
    }

    ACTION_MAP = {
        "DRAW_CARD": "カードを{value1}枚引く。",
        "ADD_MANA": "自分の山札の上から{value1}枚をマナゾーンに置く。",
        "DESTROY": "{target}を{value1}{unit}破壊する。",
        "TAP": "{target}を{value1}{unit}選び、タップする。",
        "UNTAP": "{target}を{value1}{unit}選び、アンタップする。",
        "RETURN_TO_HAND": "{target}を{value1}{unit}選び、手札に戻す。",
        "SEND_TO_MANA": "{target}を{value1}{unit}選び、マナゾーンに置く。",
        "MODIFY_POWER": "{target}のパワーを{value1}する。",
        "BREAK_SHIELD": "相手のシールドを{value1}つブレイクする。",
        "LOOK_AND_ADD": "自分の山札の上から{value1}枚を見る。その中から{value2}枚を手札に加え、残りを{zone}に置く。",
        "SEARCH_DECK_BOTTOM": "自分の山札の下から{value1}枚を見る。",
        "SEARCH_DECK": "自分の山札を見る。その中から{filter}を1枚選び、{zone}に置く。その後、山札をシャッフルする。",
        "MEKRAID": "メクレイド{value1}",
        "DISCARD": "手札を{value1}枚捨てる。",
        "PLAY_FROM_ZONE": "{source_zone}からコスト{value1}以下の{target}をプレイしてもよい。",
        "COUNT_CARDS": "{filter}の数を数える。",
        "GET_GAME_STAT": "（{str_val}を参照）",
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "CAST_SPELL": "コストを支払わずに唱える。",
        "PUT_CREATURE": "{target}をバトルゾーンに出す。",
        "GRANT_KEYWORD": "{target}に「{str_val}」を与える。",
        "MOVE_CARD": "{target}を{zone}に置く。",
        "COST_REFERENCE": "",
        "SUMMON_TOKEN": "「{str_val}」を{value1}体出す。",
        "RESET_INSTANCE": "{target}の状態をリセットする（アンタップ等）。",
        "REGISTER_DELAYED_EFFECT": "「{str_val}」の効果を{value1}ターン登録する。",
        "FRIEND_BURST": "{str_val}のフレンド・バースト",
        "MOVE_TO_UNDER_CARD": "{target}を{value1}{unit}選び、カードの下に置く。",
        "SELECT_NUMBER": "数字を1つ選ぶ。",
        "DECLARE_NUMBER": "{value1}～{value2}の数字を1つ宣言する。",
        "COST_REDUCTION": "{target}のコストを{value1}少なくする。ただし、コストは0以下にはならない。",
        "LOOK_TO_BUFFER": "{source_zone}から{value1}枚を見る（バッファへ）。",
        "SELECT_FROM_BUFFER": "バッファから{value1}枚選ぶ（{filter}）。",
        "PLAY_FROM_BUFFER": "バッファからプレイする。",
        "MOVE_BUFFER_TO_ZONE": "バッファから{zone}に置く。",
        "SELECT_OPTION": "次の中から選ぶ。",
        "LOCK_SPELL": "相手は呪文を唱えられない。",
        "APPLY_MODIFIER": "効果を付与する。",

        # --- Generalized Commands (Mapped to natural text if encountered in Card Data) ---
        "TRANSITION": "{target}を{from_zone}から{to_zone}へ移動する。", # Fallback
        "MUTATE": "{target}の状態を変更する。", # Fallback
        "FLOW": "進行制御: {str_param}",
        "QUERY": "クエリ発行: {query_mode}",
        "ATTACH": "{target}を{base_target}の下に重ねる。",
        "GAME_RESULT": "ゲームを終了する（{result}）。",
    }

    STAT_KEY_MAP = {
        "MANA_COUNT": ("マナゾーンのカード", "枚"),
        "CREATURE_COUNT": ("クリーチャー", "体"),
        "SHIELD_COUNT": ("シールド", "つ"),
        "HAND_COUNT": ("手札", "枚"),
        "GRAVEYARD_COUNT": ("墓地のカード", "枚"),
        "BATTLE_ZONE_COUNT": ("バトルゾーンのカード", "枚"),
        "OPPONENT_MANA_COUNT": ("相手のマナゾーンのカード", "枚"),
        "OPPONENT_CREATURE_COUNT": ("相手のクリーチャー", "体"),
        "OPPONENT_SHIELD_COUNT": ("相手のシールド", "つ"),
        "OPPONENT_HAND_COUNT": ("相手の手札", "枚"),
        "OPPONENT_GRAVEYARD_COUNT": ("相手の墓地のカード", "枚"),
        "OPPONENT_BATTLE_ZONE_COUNT": ("相手のバトルゾーンのカード", "枚"),
        "CARDS_DRAWN_THIS_TURN": ("このターンに引いたカード", "枚"),
        "MANA_CIVILIZATION_COUNT": ("マナゾーンの文明数", ""),
    }

    # Short aliases for natural language rendering of common zone transitions
    TRANSITION_ALIASES = {
        ("BATTLE_ZONE", "GRAVEYARD"): "破壊",
        ("HAND", "GRAVEYARD"): "捨てる",
        ("BATTLE_ZONE", "HAND"): "手札に戻す",
        ("DECK", "MANA_ZONE"): "マナチャージ",
        ("SHIELD_ZONE", "GRAVEYARD"): "シールド焼却",
        ("BATTLE_ZONE", "DECK"): "山札に戻す"
    }

    # Hint labels for how a linked input value is consumed by a subsequent step
    INPUT_USAGE_LABELS = {
        "COST": "コストとして使用",
        "AMOUNT": "枚数として使用",
        "COUNT": "枚数として使用",
        "SELECTION": "選択数として使用",
        "TARGET_COUNT": "選択数として使用",
        "POWER": "パワーとして使用",
    }

    @classmethod
    def generate_text(cls, data: Dict[str, Any], include_twinpact: bool = True) -> str:
        """
        Generate the full text for a card including name, cost, type, keywords, and effects.
        """
        if not data:
            return ""

        lines = []

        # 1. Header (Name / Cost / Civ / Race)
        lines.extend(cls.generate_header_lines(data))

        # 2. Body (Keywords, Effects, etc.)
        lines.append(cls.generate_body_text_lines(data, include_twinpact=False)) # Don't recurse here, handle manually

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

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

        # Use TYPE_MAP for translation
        raw_type = data.get("type", "CREATURE")
        type_str = cls.TYPE_MAP.get(raw_type, tr(raw_type))
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
    def generate_body_text_lines(cls, data: Dict[str, Any], include_twinpact: bool = True) -> str:
        """
        Generates just the body text (keywords, effects, etc.) without the header.
        """
        lines = []

        # Body Text (Keywords, Effects, etc.)
        body_text = cls.generate_body_text(data)
        if body_text:
            lines.append(body_text)

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " 呪文側 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def generate_body_text(cls, data: Dict[str, Any], sample: List[Any] = None) -> str:
        """
        Generates only the body text (Keywords, Effects, Reactions) without headers.
        Useful for structured preview and Twinpact separation.
        """
        if not data:
            return ""

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
                kw_str = cls.KEYWORD_TRANSLATION.get(k, tr(k))

                # Basic keywords: everything except the special set
                if k not in ("revolution_change", "mekraid", "friend_burst"):
                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
                    elif k == "hyper_energy":
                        kw_str += "（このクリーチャーを召喚する時、コストが異なる自分のクリーチャーを好きな数タップしてもよい、こうしてタップしたクリーチャー1体につき、このクリーチャーの召喚コストを2少なくする、ただし、コストは0以下にならない。）"
                    elif k == "just_diver":
                        kw_str += "（このクリーチャーが出た時、次の自分のターンのはじめまで、このクリーチャーは相手に選ばれず、攻撃されない）"
                    basic_kw_lines.append(f"■ {kw_str}")
                else:
                    # Special keywords: single-line concise style. Show selected tribe/civ as requested.
                    if k == "revolution_change":
                        cond = data.get("revolution_change_condition", {})
                        if cond and isinstance(cond, dict):
                            civs = cond.get("civilizations", []) or []
                            races = cond.get("races", []) or []
                            pieces = []
                            if civs:
                                pieces.append(cls._format_civs(civs))
                            # Add race names without extra noun when provided
                            if races:
                                if pieces:
                                    # Civs exist -> join with の
                                    pieces[-1] = pieces[-1] + "の" + "/".join(races)
                                else:
                                    pieces.append("/".join(races))
                            if pieces:
                                kw_str += f"：{''.join(pieces)}"
                    elif k == "friend_burst":
                        cond = data.get("friend_burst_condition", {})
                        if cond and isinstance(cond, dict):
                            races = cond.get("races", []) or []
                            if races:
                                kw_str += f"：{'/'.join(races)}"
                    special_kw_lines.append(f"■ {kw_str}")

        # Append in required order
        if basic_kw_lines:
            lines.extend(basic_kw_lines)
        if special_kw_lines:
            lines.extend(special_kw_lines)

        # 2.5 Cost Reductions
        cost_reductions = data.get("cost_reductions", [])
        for cr in cost_reductions:
            text = cls._format_cost_reduction(cr)
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
        is_spell = data.get("type", "CREATURE") == "SPELL"

        def _is_special_only_effect(eff: Dict[str, Any]) -> bool:
            cmds = eff.get("commands", []) or []
            if not cmds:
                return False
            special_seen = False
            for cmd in cmds:
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if ctype == "MUTATE" and cmd.get("mutation_kind") == "REVOLUTION_CHANGE":
                    special_seen = True
                elif ctype == "MEKRAID" or ctype == "FRIEND_BURST":
                    special_seen = True
                else:
                    # Found a non-special command -> not special-only
                    return False
            # All commands were special
            return special_seen

        for effect in effects:
            if _is_special_only_effect(effect):
                continue
            text = cls._format_effect(effect, is_spell, sample=sample)
            if text:
                lines.append(f"■ {text}")

        # 3.1 Static Abilities (常在効果)
        # Process static_abilities array which contains Modifier objects
        static_abilities = data.get("static_abilities", [])
        for static_ability in static_abilities:
            if static_ability and isinstance(static_ability, dict):
                text = cls._format_effect(static_ability, is_spell, sample=sample)
                if text:
                    lines.append(f"■ {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("【追加能力】")
            for effect in metamorphs:
                text = cls._format_effect(effect, is_spell, sample=sample)
                if text:
                    lines.append(f"■ {text}")

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "無色"
        return "/".join([tr(c) for c in civs])

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
    def _describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", 999)

        adjectives = []
        if civs:
            adjectives.append("/".join([tr(c) for c in civs]))

        if min_cost > 0:
            adjectives.append(f"コスト{min_cost}以上")
        if max_cost < 999:
            adjectives.append(f"コスト{max_cost}以下")

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        noun_str = "クリーチャー"
        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str

    @classmethod
    def _format_reaction(cls, reaction: Dict[str, Any]) -> str:
        if not reaction:
            return ""
        rtype = reaction.get("type", "NONE")
        if rtype == "NINJA_STRIKE":
             cost = reaction.get("cost", 0)
             return f"ニンジャ・ストライク {cost}"
        elif rtype == "STRIKE_BACK":
             return "ストライク・バック"
        elif rtype == "REVOLUTION_0_TRIGGER":
             return "革命0トリガー"
        return tr(rtype)

    @classmethod
    def _format_cost_reduction(cls, cr: Dict[str, Any]) -> str:
        if not cr:
            return ""
        ctype = cr.get("type", "PASSIVE")
        name = cr.get("name", "")
        if name:
            return f"{name}"

        unit_cost = cr.get("unit_cost", {})
        filter_def = unit_cost.get("filter", {})
        desc = cls._describe_simple_filter(filter_def)
        return f"コスト軽減: {desc}"

    @classmethod
    def _format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
        """Format a static ability (Modifier) with comprehensive support for all types and conditions."""
        mtype = modifier.get("type", "NONE")
        condition = modifier.get("condition", {})
        filter_def = modifier.get("filter", {})
        value = modifier.get("value", 0)
        str_val = modifier.get("str_val", "")
        scope = modifier.get("scope", "ALL")
        
        print(f"[TextGen._format_modifier] START: mtype={mtype}, str_val='{str_val}', scope='{scope}'")
        
        # Build condition prefix（条件がある場合）
        cond_text = cls._format_condition(condition)
        if cond_text and not cond_text.endswith("、"):
            cond_text += "、"
        
        # Build scope prefix（SCALEが SELF/OPPONENTの場合）
        scope_prefix = cls._get_scope_prefix(scope)
        print(f"[TextGen._format_modifier] scope_prefix='{scope_prefix}'")
        
        # Build target description（フィルターがある場合）
        # NOTE: フィルターは owner を持つ場合があるが、スコープで上書きする
        effective_filter = filter_def.copy() if filter_def else {}
        if scope and scope != "ALL":
            effective_filter["owner"] = scope
        
        target_str = cls._format_modifier_target(effective_filter) if effective_filter else "対象"
        print(f"[TextGen._format_modifier] target_str='{target_str}'")
        
        # Combine: condition + scope + target
        # Final structure: 「条件」「自分の」「カード」「に〜を与える」
        full_target = scope_prefix + target_str if scope_prefix else target_str
        
        # Format based on modifier type
        if mtype == "COST_MODIFIER":
            return cls._format_cost_modifier(cond_text, full_target, value)
        
        elif mtype == "POWER_MODIFIER":
            return cls._format_power_modifier(cond_text, full_target, value)
        
        elif mtype == "GRANT_KEYWORD":
            return cls._format_grant_keyword(cond_text, full_target, str_val)
        
        elif mtype == "SET_KEYWORD":
            return cls._format_set_keyword(cond_text, full_target, str_val)
        
        else:
            return f"{cond_text}{scope_prefix}常在効果: {tr(mtype)}"
    
    @classmethod
    def _get_scope_prefix(cls, scope: str) -> str:
        """Get Japanese prefix for scope."""
        if scope == "SELF":
            return "自分の"
        elif scope == "OPPONENT":
            return "相手の"
        else:  # ALL or None
            return ""
    
    @classmethod
    def _format_cost_modifier(cls, cond: str, target: str, value: int) -> str:
        """Format COST_MODIFIER modifier."""
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
    def _format_grant_keyword(cls, cond: str, target: str, str_val: str) -> str:
        """Format GRANT_KEYWORD modifier."""
        if str_val:
            # Try exact match first, then lowercase match
            keyword = cls.KEYWORD_TRANSLATION.get(str_val, 
                      cls.KEYWORD_TRANSLATION.get(str_val.lower(), str_val))
            result = f"{cond}{target}に「{keyword}」を与える。"
            print(f"[TextGen._format_grant_keyword] str_val='{str_val}', keyword='{keyword}', result='{result}'")
            return result
        # Fallback: if str_val is empty, show a more helpful message
        return f"{cond}{target}に能力を与える。"
    
    @classmethod
    def _format_set_keyword(cls, cond: str, target: str, str_val: str) -> str:
        """Format SET_KEYWORD modifier."""
        if str_val:
            # Try exact match first, then lowercase match
            keyword = cls.KEYWORD_TRANSLATION.get(str_val,
                      cls.KEYWORD_TRANSLATION.get(str_val.lower(), str_val))
            result = f"{cond}{target}は「{keyword}」を得る。"
            print(f"[TextGen._format_set_keyword] str_val='{str_val}', keyword='{keyword}', result='{result}'")
            return result
        # Fallback: if str_val is empty, show a more helpful message
        return f"{cond}{target}は能力を得る。"
    
    @classmethod
    def _format_modifier_target(cls, filter_def: Dict[str, Any]) -> str:
        """Format target description from filter with comprehensive support."""
        if not filter_def:
            return "対象"
        
        zones = filter_def.get("zones", [])
        types = filter_def.get("types", [])
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        owner = filter_def.get("owner", "")  # Will NOT apply prefix here (handled in _format_modifier)
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", 999)
        min_power = filter_def.get("min_power", 0)
        max_power = filter_def.get("max_power", 999999)
        is_tapped = filter_def.get("is_tapped")
        is_blocker = filter_def.get("is_blocker")
        is_evolution = filter_def.get("is_evolution")
        
        print(f"[TextGen._format_modifier_target] owner='{owner}', filter_keys={list(filter_def.keys())}")
        
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
        
        # Owner prefix
        # NOTE: Owner is handled by _format_modifier as a scope prefix, NOT here
        # This prevents duplication of "自分の" in the output
        # if owner:
        #     if owner == "SELF":
        #         parts.append("自分の")
        #     elif owner == "OPPONENT":
        #         parts.append("相手の")
        
        # Civilization adjective
        if civs:
            parts.append("/".join([tr(c) for c in civs]) + "の")
        
        # Race adjective
        if races:
            parts.append("/".join(races) + "の")
        
        # Cost range
        if min_cost > 0 and max_cost < 999:
            parts.append(f"コスト{min_cost}～{max_cost}の")
        elif min_cost > 0:
            parts.append(f"コスト{min_cost}以上の")
        elif max_cost < 999:
            parts.append(f"コスト{max_cost}以下の")
        
        # Power range
        if min_power > 0 and max_power < 999999:
            parts.append(f"パワー{min_power}～{max_power}の")
        elif min_power > 0:
            parts.append(f"パワー{min_power}以上の")
        elif max_power < 999999:
            parts.append(f"パワー{max_power}以下の")
        
        # Type noun
        type_noun = "カード"
        if types:
            if len(types) == 1:
                if types[0] == "CREATURE":
                    type_noun = "クリーチャー"
                elif types[0] == "SPELL":
                    type_noun = "呪文"
            else:
                type_words = []
                if "CREATURE" in types:
                    type_words.append("クリーチャー")
                if "SPELL" in types:
                    type_words.append("呪文")
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
    def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None) -> str:
        # Check if this is a Modifier (static ability)
        # Modifiers have a 'type' field with specific values (COST_MODIFIER, POWER_MODIFIER, GRANT_KEYWORD, SET_KEYWORD)
        # and do NOT have a 'trigger' field (or trigger is NONE)
        if isinstance(effect, dict):
            effect_type = effect.get("type", "")
            trigger = effect.get("trigger", "NONE")
            
            # Check if this is a known Modifier type
            if effect_type in ("COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"):
                # Verify it's not a triggered effect
                if trigger == "NONE" or trigger not in effect:
                    return cls._format_modifier(effect, sample=sample)
        
        trigger = effect.get("trigger", "NONE")
        condition = effect.get("condition", {})
        actions = effect.get("actions", [])

        trigger_text = cls.trigger_to_japanese(trigger, is_spell)
        cond_text = cls._format_condition(condition)
        cond_type = condition.get("type", "NONE")

        # Refined natural language logic
        if trigger != "NONE" and trigger != "PASSIVE_CONST":
            if cond_type == "DURING_YOUR_TURN" or cond_type == "DURING_OPPONENT_TURN":
                base_cond = cond_text.replace(": ", "")
                trigger_text = f"{base_cond}、{trigger_text}" # 自分のターン中、このクリーチャーが出た時
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"相手がカードを引いた時、{val}枚目以降なら"
                cond_text = ""

        action_texts = []
        # Keep parallel lists of raw and formatted for merging logic
        raw_items = []

        # Commands-First Policy (Migration Phase 4.3):
        # If 'commands' exist, they are the source of truth. Ignore 'actions'.
        # If only 'actions' exist, convert them on-the-fly to commands for rendering.
        commands = effect.get("commands", [])
        if commands:
            for command in commands:
                raw_items.append(command)
                action_texts.append(cls._format_command(command, is_spell, sample=sample))
        else:
            # Automatic migration: convert legacy actions to commands on-the-fly
            legacy_actions = effect.get("actions", [])
            if legacy_actions:
                from dm_toolkit.gui.editor.action_converter import ActionConverter
                for action in legacy_actions:
                    try:
                        # Convert Action to Command format
                        converted = ActionConverter.convert(action)
                        if converted and converted.get('type') != 'NONE':
                            raw_items.append(converted)
                            action_texts.append(cls._format_command(converted, is_spell, sample=sample))
                        else:
                            # Fallback if conversion fails
                            raw_items.append(action)
                            action_texts.append(cls._format_action(action, is_spell, sample=sample))
                    except Exception:
                        # Fallback to legacy rendering on conversion error
                        raw_items.append(action)
                        action_texts.append(cls._format_action(action, is_spell, sample=sample))

        # Try to merge common sequential patterns for more natural language
        full_action_text = cls._merge_action_texts(raw_items, action_texts)

        # If it's a Spell's main effect (ON_PLAY), we can often omit the trigger text "Played/Cast"
        if is_spell and trigger == "ON_PLAY":
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
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False) -> str:
        mapping = {
            "ON_PLAY": "このクリーチャーが出た時" if not is_spell else "この呪文を唱えた時", # Suppressed later for main spell effect
            "ON_OTHER_ENTER": "他のクリーチャーが出た時",
            "AT_ATTACK": "このクリーチャーが攻撃する時",
            "ON_DESTROY": "このクリーチャーが破壊された時",
            "AT_END_OF_TURN": "自分のターンの終わりに",
            "AT_END_OF_OPPONENT_TURN": "相手のターンの終わりに",
            "ON_BLOCK": "このクリーチャーがブロックした時",
            "ON_ATTACK_FROM_HAND": "手札から攻撃する時",
            "TURN_START": "自分のターンのはじめに",
            "S_TRIGGER": "S・トリガー",
            "PASSIVE_CONST": "（常在効果）",
            "ON_SHIELD_ADD": "カードがシールドゾーンに置かれた時",
            "AT_BREAK_SHIELD": "シールドをブレイクする時",
            "ON_CAST_SPELL": "呪文を唱えた時",
            "ON_OPPONENT_DRAW": "相手がカードを引いた時",
            "NONE": ""
        }
        # Fallback: try localization table before leaking raw enum-like tokens
        return mapping.get(trigger, tr(trigger))

    @classmethod
    def _normalize_zone_name(cls, zone: str) -> str:
        if not zone:
            return ""

        z = str(zone).split(".")[-1].upper()
        zone_map = {
            # CommandSystem short names -> JSON/UI long names
            "BATTLE": "BATTLE_ZONE",
            "MANA": "MANA_ZONE",
            "SHIELD": "SHIELD_ZONE",

            # Identity / already-long
            "BATTLE_ZONE": "BATTLE_ZONE",
            "MANA_ZONE": "MANA_ZONE",
            "SHIELD_ZONE": "SHIELD_ZONE",
            "HAND": "HAND",
            "GRAVEYARD": "GRAVEYARD",
            "DECK": "DECK",
            "DECK_TOP": "DECK_TOP",
            "DECK_BOTTOM": "DECK_BOTTOM",
            "BUFFER": "BUFFER",
            "UNDER_CARD": "UNDER_CARD",
        }
        return zone_map.get(z, z)

    @classmethod
    def _format_command(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None) -> str:
        if not command:
            return ""

        # Map CommandDef fields to Action-like dict to reuse _format_action logic where possible
        cmd_type = command.get("type", "NONE")

        # Mapping CommandType to ActionType logic where applicable
        original_cmd_type = cmd_type
        if cmd_type == "POWER_MOD": cmd_type = "MODIFY_POWER"
        elif cmd_type == "ADD_KEYWORD": 
            cmd_type = "ADD_KEYWORD"
            # Ensure mutation_kind is mapped to str_val for text generation
            if not command.get("str_val") and command.get("mutation_kind"):
                command["str_val"] = command["mutation_kind"]
        elif cmd_type == "MANA_CHARGE": cmd_type = "SEND_TO_MANA" # Or ADD_MANA depending on context
        elif cmd_type == "CHOICE": cmd_type = "SELECT_OPTION"

        # Construct proxy action object
        action_proxy = {
            "type": cmd_type,
            "scope": command.get("target_group", "NONE"),
            "filter": command.get("target_filter", {}),
            "value1": command.get("amount", 0),
            "optional": command.get("optional", False),
            "up_to": command.get("up_to", False),
            # Prefer the normalized key, but accept legacy key if present
            "str_val": command.get("str_param") or command.get("str_val", ""),
            "input_value_key": command.get("input_value_key", ""),
            "input_value_usage": command.get("input_value_usage", ""),
            "from_zone": command.get("from_zone", ""),
            "to_zone": command.get("to_zone", ""),
            "mutation_kind": command.get("mutation_kind", ""),
            "destination_zone": command.get("to_zone", ""), # For MOVE_CARD mapping
            "result": command.get("str_param", "") # For GAME_RESULT
        }

        # Extra passthrough fields for complex/structured commands
        if "options" in command:
            action_proxy["options"] = command.get("options")
        if "flags" in command:
            action_proxy["flags"] = command.get("flags")
        if "look_count" in command:
            action_proxy["look_count"] = command.get("look_count")
        if "add_count" in command:
            action_proxy["add_count"] = command.get("add_count")
        if "rest_zone" in command:
            action_proxy["rest_zone"] = command.get("rest_zone")
        if "max_cost" in command:
            action_proxy["max_cost"] = command.get("max_cost")
        if "token_id" in command:
            action_proxy["token_id"] = command.get("token_id")
        if "play_flags" in command:
            action_proxy["play_flags"] = command.get("play_flags")

        # Some templates expect source_zone rather than from_zone
        action_proxy["source_zone"] = command.get("from_zone", "")

        # Specific Adjustments
        if original_cmd_type == "MANA_CHARGE":
            if action_proxy["scope"] == "NONE":
                 action_proxy["type"] = "ADD_MANA" # Top of deck charge
            else:
                 action_proxy["type"] = "SEND_TO_MANA"

        if original_cmd_type == "SHIELD_TRIGGER":
             return "S・トリガー"

        # Normalize complex command representations into the Action-like proxy expected by _format_action
        if original_cmd_type == "LOOK_AND_ADD":
            if "look_count" in command and command.get("look_count") is not None:
                action_proxy["value1"] = command.get("look_count")
            if "add_count" in command and command.get("add_count") is not None:
                action_proxy["value2"] = command.get("add_count")
            rz = command.get("rest_zone") or command.get("destination_zone") or command.get("to_zone")
            if rz:
                action_proxy["rest_zone"] = rz
                action_proxy["destination_zone"] = rz
        elif original_cmd_type == "MEKRAID":
            if "max_cost" in command and command.get("max_cost") is not None:
                action_proxy["value1"] = command.get("max_cost")
            if "look_count" in command and command.get("look_count") is not None:
                action_proxy["look_count"] = command.get("look_count")
            if "rest_zone" in command and command.get("rest_zone") is not None:
                action_proxy["rest_zone"] = command.get("rest_zone")
        elif original_cmd_type == "SUMMON_TOKEN":
            # ACTION_MAP expects str_val for token name
            if "token_id" in command and command.get("token_id") is not None:
                action_proxy["str_val"] = command.get("token_id")
        elif original_cmd_type == "PLAY_FROM_ZONE":
            action_proxy["source_zone"] = command.get("from_zone", "")
            if "max_cost" in command and command.get("max_cost") is not None:
                action_proxy["value1"] = command.get("max_cost")
        elif original_cmd_type == "CHOICE":
            # Map CHOICE into SELECT_OPTION natural language generation
            flags = command.get("flags", []) or []
            if isinstance(flags, list) and "ALLOW_DUPLICATES" in flags:
                action_proxy["value2"] = 1
            action_proxy["value1"] = command.get("amount", 1)

        return cls._format_action(action_proxy, is_spell, sample=sample)

    @classmethod
    def _format_condition(cls, condition: Dict[str, Any]) -> str:
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ_raw = condition.get("str_val", "")
            civ = tr(civ_raw)
            return f"マナ武装 {val} ({civ}): "

        elif cond_type == "SHIELD_COUNT":
            val = condition.get("value", 0)
            op = condition.get("op", ">=")
            op_text = "以上" if op == ">=" else "以下" if op == "<=" else ""
            if op == "=": op_text = ""
            return f"自分のシールドが{val}つ{op_text}なら: "

        elif cond_type == "CIVILIZATION_MATCH":
             return "マナゾーンに同じ文明があれば: "

        elif cond_type == "COMPARE_STAT":
             key = condition.get("stat_key", "")
             op = condition.get("op", "=")
             val = condition.get("value", 0)
             stat_name, unit = cls.STAT_KEY_MAP.get(key, (key, ""))

             op_text = ""
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
             return f"自分の{stat_name}が{op_text}なら: "

        elif cond_type == "OPPONENT_PLAYED_WITHOUT_MANA":
            return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた時: "

        elif cond_type == "OPPONENT_DRAW_COUNT":
            val = condition.get("value", 0)
            return f"{val}枚目以降なら: "

        elif cond_type == "DURING_YOUR_TURN":
            return "自分のターン中: "
        elif cond_type == "DURING_OPPONENT_TURN":
            return "相手のターン中: "
        elif cond_type == "EVENT_FILTER_MATCH":
            return ""

        return ""

    @classmethod
    def _format_action(cls, action: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None) -> str:
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
        template = cls.ACTION_MAP.get(atype, "")

        # Up-to drawing: adjust template for explicit DRAW_CARD
        if atype == 'DRAW_CARD':
            if bool(action.get('up_to', False)):
                template = "最大{value1}枚まで引く。"

        # Special-case: treat TRANSITION from DECK->HAND as DRAW_CARD for natural language
        if atype == 'TRANSITION':
            from_zone = cls._normalize_zone_name(action.get('from_zone') or action.get('fromZone') or '')
            to_zone = cls._normalize_zone_name(action.get('to_zone') or action.get('toZone') or '')
            amt = action.get('amount') or action.get('value1') or 0
            up_to = bool(action.get('up_to', False))

            # Use short alias if available (e.g., "破壊" for BATTLE->GRAVEYARD)
            alias = cls.TRANSITION_ALIASES.get((from_zone, to_zone))
            if alias:
                 # Reconstruct natural sentences based on known aliases
                 if alias == "破壊":
                      return f"{{target}}を{amt}体破壊する。" if amt > 0 else f"{{target}}をすべて破壊する。"
                 elif alias == "捨てる":
                      return f"手札を{amt}枚捨てる。" if amt > 0 else "手札をすべて捨てる。"
                 elif alias == "手札に戻す":
                      return f"{{target}}を{amt}体手札に戻す。" if amt > 0 else f"{{target}}をすべて手札に戻す。"
                 elif alias == "マナチャージ":
                      return f"自分の山札の上から{amt}枚をマナゾーンに置く。"
                 elif alias == "シールド焼却":
                      return f"相手のシールドを{amt}つ選び、墓地に置く。"

            # If transition represents drawing from deck to hand
            if (from_zone == 'DECK' or from_zone == '') and to_zone == 'HAND':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                if up_to:
                    return f"山札からカードを最大{amt}枚まで手札に加える。"
                return f"カードを{amt}枚引く。"
            # If transition represents moving to mana zone, render as ADD_MANA
            if to_zone == 'MANA_ZONE':
                if not amt and isinstance(action.get('target_filter'), dict):
                    amt = action.get('target_filter', {}).get('count', 1)
                return f"自分の山札の上から{amt}枚をマナゾーンに置く。"

        # Determine verb form (standard or optional)
        optional = action.get("optional", False)

        # Resolve dynamic target strings
        target_str, unit = cls._resolve_target(action, is_spell)

        # Parameter Substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")
        input_key = action.get("input_value_key", "")
        input_usage = action.get("input_value_usage") or action.get("input_usage")

        is_generic_selection = atype in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA", "MOVE_CARD", "TRANSITION", "DISCARD"]

        # 1. Handle Input Variable Linking (Contextual substitution)
        if input_key:
            # 前のアクションの出力を参照する場合
            if atype == "DRAW_CARD": 
                up_to_flag = bool(action.get('up_to', False))
                template = "カードをその同じ枚数引く。"
                if up_to_flag:
                    template = "カードをその同じ枚数まで引く。"
            elif atype == "DESTROY": 
                template = "{target}をその同じ数だけ破壊する。"
            elif atype == "TAP": 
                template = "{target}をその同じ数だけ選び、タップする。"
            elif atype == "UNTAP": 
                template = "{target}をその同じ数だけ選び、アンタップする。"
            elif atype == "RETURN_TO_HAND": 
                template = "{target}をその同じ数だけ選び、手札に戻す。"
            elif atype == "SEND_TO_MANA": 
                template = "{target}をその同じ数だけ選び、マナゾーンに置く。"
            elif atype == "TRANSITION":
                # TRANSITION用の汎用的な参照表現
                val1 = "その同じ枚数"
                # 「まで」フラグがある場合は追加
                if bool(action.get('up_to', False)):
                    val1 = "その同じ枚数まで"
            elif atype == "MOVE_CARD":
                # MOVE_CARDの入力リンク対応（行き先に応じた自然文）
                dest_zone = action.get("destination_zone", "")
                up_to_suffix = "まで" if bool(action.get('up_to', False)) else ""
                if dest_zone == "DECK_BOTTOM":
                    template = f"{{target}}をその同じ数だけ{up_to_suffix}選び、山札の下に置く。"
                elif dest_zone == "GRAVEYARD":
                    template = f"{{target}}をその同じ数だけ{up_to_suffix}選び、墓地に置く。"
                elif dest_zone == "HAND":
                    template = f"{{target}}をその同じ数だけ{up_to_suffix}選び、手札に戻す。"
                elif dest_zone == "MANA_ZONE":
                    template = f"{{target}}をその同じ数だけ{up_to_suffix}選び、マナゾーンに置く。"
            elif atype == "DISCARD":
                # 前回の出力枚数と同じ枚数を捨てる
                template = "手札をその同じ枚数捨てる。"
                if bool(action.get('up_to', False)):
                    template = "手札をその同じ枚数まで捨てる。"
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
        if atype == "MODIFY_POWER":
            val = action.get("value1", 0)
            sign = "+" if val >= 0 else ""
            return f"{target_str}のパワーを{sign}{val}する。"

        elif atype == "SELECT_NUMBER":
            val1 = action.get("value1", 0)
            val2 = action.get("value2", 0)
            if val1 > 0 and val2 > 0:
                 template = f"{val1}～{val2}の数字を1つ選ぶ。"

        elif atype == "SELECT_OPTION":
            options = action.get("options", [])
            lines = []
            val1 = action.get("value1", 1)
            lines.append(f"次の中から{val1}回選ぶ。（同じものを選んでもよい）")
            for i, opt_chain in enumerate(options):
                parts = []
                for a in opt_chain:
                    # options may contain either legacy Action dicts or normalized Command dicts
                    if isinstance(a, dict) and (
                        'amount' in a or 'target_group' in a or 'mutation_kind' in a or 'from_zone' in a or 'to_zone' in a
                    ):
                        parts.append(cls._format_command(a, is_spell=is_spell, sample=sample))
                    else:
                        parts.append(cls._format_action(a, is_spell, sample=sample))
                chain_text = " ".join(parts)
                lines.append(f"> {chain_text}")
            return "\n".join(lines)

        elif atype == "MEKRAID":
            val1 = action.get("value1", 0)
            return f"メクレイド{val1}（自分の山札の上から3枚を見る。その中からコスト{val1}以下のクリーチャーを1体、コストを支払わずに召喚してもよい。残りを山札の下に好きな順序で置く）"

        elif atype == "FRIEND_BURST":
            str_val = action.get("str_val", "")
            return f"＜{str_val}＞のフレンド・バースト（このクリーチャーが出た時、自分の他の{str_val}・クリーチャーを1体タップしてもよい。そうしたら、このクリーチャーの呪文側をバトルゾーンに置いたまま、コストを支払わずに唱える。）"

        elif atype == "REVOLUTION_CHANGE":
             return ""

        elif atype == "APPLY_MODIFIER":
             str_val = action.get("str_val", "")
             val1 = action.get("value1", 0)
             if str_val == "SPEED_ATTACKER":
                 return f"{target_str}に「スピードアタッカー」を与える。"
             elif str_val == "BLOCKER":
                 return f"{target_str}に「ブロッカー」を与える。"
             elif str_val == "SLAYER":
                 return f"{target_str}に「スレイヤー」を与える。"
             elif str_val == "COST":
                 sign = "少なくする" if val1 > 0 else "増やす"
                 return f"{target_str}のコストを{abs(val1)}{sign}。"
             else:
                 jp_val = cls.KEYWORD_TRANSLATION.get(str_val.lower(), str_val)
                 if jp_val != str_val:
                     return f"{target_str}に「{jp_val}」を与える。"
                 return f"{target_str}に効果（{str_val}）を与える。"

        # --- Enhanced Command-like actions ---
        elif atype == "TRANSITION":
            from_z = cls._normalize_zone_name(action.get("from_zone", ""))
            to_z = cls._normalize_zone_name(action.get("to_zone", ""))
            amount = action.get("amount", 0)

            # Natural Language Mapping with explicit source/destination zones
            if from_z == "BATTLE_ZONE" and to_z == "GRAVEYARD":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
                if amount == 0 and not input_key:
                    template = "{from_z}の{target}をすべて{to_z}に置く。"
            elif from_z == "BATTLE_ZONE" and to_z == "MANA_ZONE":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
                if amount == 0 and not input_key:
                    template = "{from_z}の{target}をすべて{to_z}に置く。"
            elif from_z == "BATTLE_ZONE" and to_z == "HAND":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に戻す。"
                if amount == 0 and not input_key:
                    template = "{from_z}の{target}をすべて{to_z}に戻す。"
            elif from_z == "HAND" and to_z == "MANA_ZONE":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
            elif from_z == "DECK" and to_z == "HAND":
                # Draw: include both zones explicitly
                template = "山札からカードを{amount}枚手札に加える。"
                if target_str != "カード":  # Search logic
                    template = "{from_z}から{target}を{amount}{unit}{to_z}に加える。"
            elif from_z == "GRAVEYARD" and to_z == "HAND":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に戻す。"
            elif from_z == "GRAVEYARD" and to_z == "BATTLE_ZONE":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に出す。"
            elif to_z == "GRAVEYARD":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"  # Generic discard/mill
            elif to_z == "DECK_BOTTOM":
                template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
                if input_key:
                    # 入力リンクがある場合は単位重複を避けた表現へ
                    template = "{from_z}の{target}をその同じ数だけ選び、{to_z}に置く。"
            else:
                template = "{target}を{from_z}から{to_z}へ移動する。"

            # input_value_keyがある場合は「その同じ枚数」と表示
            if input_key:
                val1 = "その同じ枚数"
            elif amount == 0 and not is_generic_selection:
                val1 = "すべて"
            else:
                val1 = amount

            # Zone name localization when placeholders are present
            if "{from_z}" in template:
                template = template.replace("{from_z}", tr(from_z))
            if "{to_z}" in template:
                template = template.replace("{to_z}", tr(to_z))

        elif atype == "MUTATE":
             mkind = action.get("mutation_kind", "")
             val1 = action.get("amount", 0)
             str_param = action.get("str_param", "")

             if mkind == "TAP":
                 template = "{target}を{amount}{unit}選び、タップする。"
             elif mkind == "UNTAP":
                 template = "{target}を{amount}{unit}選び、アンタップする。"
             elif mkind == "POWER_MOD":
                 sign = "+" if val1 >= 0 else ""
                 return f"{target_str}のパワーを{sign}{val1}する。"
             elif mkind == "ADD_KEYWORD":
                 keyword = cls.KEYWORD_TRANSLATION.get(str_param.lower(), str_param)
                 return f"{target_str}に「{keyword}」を与える。"
             elif mkind == "REMOVE_KEYWORD":
                 keyword = cls.KEYWORD_TRANSLATION.get(str_param.lower(), str_param)
                 return f"{target_str}の「{keyword}」を無視する。"
             elif mkind == "ADD_PASSIVE_EFFECT" or mkind == "ADD_MODIFIER":
                 # Use str_param if available to describe what is added
                 if str_param:
                     kw = cls.KEYWORD_TRANSLATION.get(str_param.lower(), str_param)
                     # Check if it looks like a keyword (standard mapping) or generic
                     return f"{target_str}に「{kw}」を与える。"
                 else:
                     return f"{target_str}にパッシブ効果を与える。"
             elif mkind == "ADD_COST_MODIFIER":
                 return f"{target_str}にコスト修正を追加する。"
             else:
                 template = f"状態変更({tr(mkind)}): {{target}} (値:{val1})"

             if val1 == 0:
                 template = template.replace("{amount}{unit}選び、", "すべて") # Simplified "choose all"
                 val1 = ""

        elif atype == "QUERY":
             # If this is a stat query, prefer human-readable stat labels
             # Accept several possible keys produced by converters: 'stat', 'str_param', 'str_val'
             key = action.get('stat') or action.get('str_param') or action.get('str_val') or action.get('query_mode')
             if key:
                 stat_name, unit = cls.STAT_KEY_MAP.get(key, (None, None))
                 if stat_name:
                     # Return concise stat reference
                     base = f"{stat_name}{unit}を数える。"
                     if input_key:
                         usage_label = cls._format_input_usage_label(input_usage)
                         if usage_label:
                             base += f"（{usage_label}）"
                     return base
             mode = action.get("query_mode", "")
             base = f"質問: {tr(mode)}"
             if input_key:
                 usage_label = cls._format_input_usage_label(input_usage)
                 if usage_label:
                     base += f"（{usage_label}）"
             return base

        elif atype == "FLOW":
             ftype = action.get("flow_type", "")
             val1 = action.get("value1", 0) # Often raw int

             if ftype == "PHASE_CHANGE":
                 phase_name = cls.PHASE_MAP.get(val1, str(val1))
                 return f"{phase_name}フェーズへ移行する。"
             elif ftype == "TURN_CHANGE":
                 return f"ターンを終了する。"
             elif ftype == "SET_ACTIVE_PLAYER":
                 return f"手番を変更する。"

             return f"進行制御({tr(ftype)}): {val1}"

        elif atype == "GAME_RESULT":
             res = action.get("result", "")
             return f"ゲームを終了する（{tr(res)}）。"

        elif atype == "ATTACH":
            # Resolving base target might be tricky without a full "base_target" definition in Action,
            # usually ATTACH targets an existing card.
            return f"{target_str}をカードの下に重ねる。"

        # --- Additional command types previously unformatted ---
        elif atype == "DECIDE":
            # Finalize selection/decision. Card JSON rarely uses this, but provide readable output.
            sel = action.get("selected_option_index")
            if isinstance(sel, int) and sel >= 0:
                return f"選択肢{sel}を確定する。"
            indices = action.get("selected_indices") or []
            if isinstance(indices, list) and indices:
                return f"選択（{indices}）を確定する。"
            return "選択を確定する。"

        elif atype == "DECLARE_REACTION":
            # Declare/pass reaction. Keep concise natural phrasing.
            if action.get("pass"):
                return "リアクション: パスする。"
            idx = action.get("reaction_index")
            if isinstance(idx, int):
                return f"リアクションを宣言する（インデックス {idx}）。"
            return "リアクションを宣言する。"

        elif atype == "STAT":
            # Update game stats; prefer human-readable stat labels.
            key = action.get('stat') or action.get('str_param') or action.get('str_val')
            amount = action.get('amount', action.get('value1', 0))
            if key:
                stat_name, unit = cls.STAT_KEY_MAP.get(str(key), (None, None))
                if stat_name:
                    return f"統計更新: {stat_name} += {amount}"
            return f"統計更新: {tr(str(key))} += {amount}"

        if not template:
            return f"({tr(atype)})"

        if atype == "GRANT_KEYWORD" or atype == "ADD_KEYWORD":
            # キーワードの翻訳を適用
            keyword = cls.KEYWORD_TRANSLATION.get(str_val.lower(), str_val)
            str_val = keyword

        elif atype == "GET_GAME_STAT":
            # str_val holds the stat key, e.g. MANA_CIVILIZATION_COUNT
            key = action.get('str_val') or action.get('result') or ''
            stat_name, unit = cls.STAT_KEY_MAP.get(key, (None, None))
            if stat_name:
                # If a sample is provided, attempt to compute an example value
                if sample is not None:
                    try:
                        val = cls._compute_stat_from_sample(key, sample)
                        if val is not None:
                            return f"{stat_name}（例: {val}{unit}）"
                    except Exception:
                        # Fall back to concise name on error
                        pass
                return f"{stat_name}"
            return f"（{tr(key)}を参照）"

        elif atype == "COUNT_CARDS":
            if not target_str or target_str == "カード":
                 return f"({tr('COUNT_CARDS')})"
            return f"{target_str}の数を数える。"

        elif atype == "MOVE_CARD":
            dest_zone = action.get("destination_zone", "")
            is_all = (val1 == 0 and not input_key)

            # Include source zone when available for clearer movement description
            src_zone = action.get("source_zone", "")
            src_str = tr(src_zone) if src_zone else ""
            zone_str = tr(dest_zone) if dest_zone else "どこか"

            if dest_zone == "HAND":
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に戻す。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に戻す。")
            elif dest_zone == "MANA_ZONE":
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
            elif dest_zone == "GRAVEYARD":
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
            elif dest_zone == "DECK_BOTTOM":
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
                if is_all:
                    template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                                else f"{src_str}の{{target}}をすべて{zone_str}に置く。")

        elif atype == "CAST_SPELL":
            # CAST_SPELL: フィルタの詳細を反映した呪文テキスト生成
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter
            
            # フィルタでSPELLタイプが指定されている場合、詳細なターゲット文字列を生成
            types = temp_filter.get("types", [])
            if "SPELL" in types:
                # フィルタに基づいた詳細なターゲット文字列を取得
                target_str, unit = cls._resolve_target(action)
                # 「呪文」部分を削除（すでに末尾に「呪文」が含まれている場合）
                if target_str.endswith("呪文"):
                    # 「火の呪文」→「火の」のように、「呪文」を除去
                    prefix = target_str[:-2] if len(target_str) > 2 else ""
                    template = f"{prefix}呪文をコストを支払わずに唱える。"
                elif target_str == "カード" or target_str == "":
                    template = "呪文をコストを支払わずに唱える。"
                else:
                    template = f"{target_str}の呪文をコストを支払わずに唱える。"
            else:
                # SPELLタイプ以外の場合は通常のターゲット文字列
                target_str, unit = cls._resolve_target(action)
                if target_str == "" or target_str == "カード":
                    template = "カードをコストを支払わずに唱える。"
                else:
                    template = f"{target_str}をコストを支払わずに唱える。"

        elif atype == "PLAY_FROM_ZONE":
            action = action.copy()
            temp_filter = action.get("filter", {}).copy()
            action["filter"] = temp_filter

            if not action.get("source_zone") and "zones" in temp_filter:
                zones = temp_filter["zones"]
                if len(zones) == 1:
                    action["source_zone"] = zones[0]

            if action.get("value1", 0) == 0:
                max_cost = temp_filter.get("max_cost", 999)
                if max_cost < 999:
                    action["value1"] = max_cost
                    if not input_key: val1 = max_cost
                    if "max_cost" in temp_filter: del temp_filter["max_cost"]

            if "zones" in temp_filter: temp_filter["zones"] = []
            scope = action.get("scope", "NONE")
            if scope in ["PLAYER_SELF", "SELF"]: action["scope"] = "NONE"

            target_str, unit = cls._resolve_target(action)
            verb = "プレイする"
            types = temp_filter.get("types", [])
            if "SPELL" in types and "CREATURE" not in types:
                verb = "唱える"
            elif "CREATURE" in types:
                verb = "召喚する"

            if action.get("source_zone"):
                template = "{source_zone}からコスト{value1}以下の{target}を" + verb + "。"
            else:
                template = "コスト{value1}以下の{target}を" + verb + "。"

        # Destination/Source Resolution
        dest_zone = action.get("destination_zone", "")
        zone_str = tr(dest_zone) if dest_zone else "どこか"
        src_zone = action.get("source_zone", "")
        src_str = tr(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

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
                replacement = "この呪文" if is_spell else "このクリーチャー"
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
    def _format_input_usage_label(cls, usage: Any) -> str:
        """Return a short label indicating how an input value is used."""
        if usage is None:
            return ""
        norm = str(usage).upper()
        if norm in cls.INPUT_USAGE_LABELS:
            return cls.INPUT_USAGE_LABELS[norm]
        # Fallback to raw string for custom labels
        return tr(str(usage)) if str(usage) else ""

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False) -> Tuple[str, str]:
        """
        Attempt to describe the target based on scope, filter, etc.
        Returns (target_description, unit_counter)
        """
        scope = action.get("scope", "NONE")
        filter_def = action.get("filter", {})
        atype = action.get("type", "")

        target_desc = ""
        prefix = ""
        adjectives = ""
        zone_noun = ""
        type_noun = "カード"
        unit = "枚"

        if atype == "DISCARD" and scope == "NONE":
             scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
             target = "この呪文" if is_spell else "このクリーチャー"
             return (target, "枚")

        if scope == "PLAYER_OPPONENT": prefix = "相手の"
        elif scope == "PLAYER_SELF" or scope == "SELF": prefix = "自分の"
        elif scope == "ALL_PLAYERS": prefix = "すべてのプレイヤーの"
        elif scope == "RANDOM": prefix = "ランダムな"

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])
            owner = filter_def.get("owner", "NONE")

            # Handle explicit owner filter if scope is generic
            if owner == "PLAYER_OPPONENT" and not prefix: prefix = "相手の"
            elif owner == "PLAYER_SELF" and not prefix: prefix = "自分の"

            temp_adjs = []
            if civs: temp_adjs.append("/".join([tr(c) for c in civs]))
            if races: temp_adjs.append("/".join(races))

            if temp_adjs: adjectives += "/".join(temp_adjs) + "の"

            if filter_def.get("min_cost", 0) > 0: adjectives += f"コスト{filter_def['min_cost']}以上の"
            if filter_def.get("max_cost", 999) < 999: adjectives += f"コスト{filter_def['max_cost']}以下の"

            if filter_def.get("is_tapped", None) is True: adjectives = "タップされている" + adjectives
            elif filter_def.get("is_tapped", None) is False: adjectives = "アンタップされている" + adjectives
            if filter_def.get("is_blocker", None) is True: adjectives = "ブロッカーを持つ" + adjectives
            if filter_def.get("is_evolution", None) is True: adjectives = "進化" + adjectives

            if "BATTLE_ZONE" in zones:
                zone_noun = "バトルゾーン"
                if "CARD" in types:
                    type_noun = "カード"
                    unit = "枚"
                elif "ELEMENT" in types:
                    type_noun = "エレメント"
                    unit = "つ"
                elif "CREATURE" in types or (not types):
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "CROSS_GEAR" in types:
                    type_noun = "クロスギア"
            elif "MANA_ZONE" in zones:
                zone_noun = "マナゾーン"
            elif "HAND" in zones:
                zone_noun = "手札"
            elif "SHIELD_ZONE" in zones:
                zone_noun = "シールドゾーン"
                type_noun = "カード"
                unit = "つ"
            elif "GRAVEYARD" in zones:
                zone_noun = "墓地"
                if "CREATURE" in types:
                     type_noun = "クリーチャー"
                     unit = "体"
                elif "SPELL" in types:
                     type_noun = "呪文"
            elif "DECK" in zones:
                 zone_noun = "山札"

            if not zone_noun:
                if "CARD" in types:
                    type_noun = "カード"
                    unit = "枚"
                elif "ELEMENT" in types:
                    type_noun = "エレメント"
                    unit = "つ"
                elif "CREATURE" in types:
                    type_noun = "クリーチャー"
                    unit = "体"
                elif "SPELL" in types:
                    type_noun = "呪文"
                elif len(types) > 1:
                     # Join multiple types (e.g., Creature/Spell)
                     type_noun = "または".join([tr(t) for t in types])

            # Special case for SEARCH_DECK
            if atype == "SEARCH_DECK":
                 # Usually searching for a specific card in deck
                 pass

            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "の")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)
            target_desc = "".join(parts)

            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"
            if "BATTLE_ZONE" in zones:
                 target_desc = target_desc.replace("バトルゾーンの", "")

        else:
            if atype == "DESTROY":
                 if scope == "PLAYER_OPPONENT" or scope == "OPPONENT":
                     target_desc = "相手のクリーチャー"
                     unit = "体"
            elif atype == "TAP" or atype == "UNTAP":
                 if "クリーチャー" not in target_desc:
                      target_desc = prefix + "クリーチャー"
                      unit = "体"
            elif atype == "DISCARD":
                 target_desc = "手札"
            else:
                 target_desc = "カード"

        if not target_desc: target_desc = "カード"
        return target_desc, unit

    @classmethod
    def _merge_action_texts(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        """Post-process sequence of formatted action/command texts to produce
        more natural combined sentences for common patterns.

        Currently implements:
        - Draw (DECK->HAND) followed by move to deck-bottom ->
          "...カードをN枚引く。その後、引いた枚数と同じ枚数を山札の下に置く。"
        """
        if not formatted_texts:
            return ""

        # helper predicates
        def is_draw_item(it):
            if not isinstance(it, dict):
                return False
            t = it.get('type', '')
            if t == 'DRAW_CARD':
                return True
            if t == 'TRANSITION':
                from_z = (it.get('from_zone') or it.get('fromZone') or '').upper()
                to_z = (it.get('to_zone') or it.get('toZone') or '').upper()
                if (from_z == '' or 'DECK' in from_z) and 'HAND' in to_z:
                    return True
            return False

        def is_deck_bottom_move(it):
            if not isinstance(it, dict):
                return False
            # check common keys that indicate deck-bottom destination
            dest = (it.get('destination_zone') or it.get('to_zone') or it.get('toZone') or '').upper()
            if 'DECK_BOTTOM' in dest or 'DECKBOTTOM' in dest:
                return True
            # type names sometimes include DECK_BOTTOM
            t = (it.get('type') or '').upper()
            if 'DECK_BOTTOM' in t:
                return True
            return False

        # Pattern: first item is draw, second is deck-bottom move
        if len(raw_items) >= 2 and is_draw_item(raw_items[0]) and is_deck_bottom_move(raw_items[1]):
            first = formatted_texts[0].rstrip('。')
            # Compose merged sentence: keep first action, then reference drawn count
            tail = 'その後、引いた枚数と同じ枚数を山札の下に置く。'
            merged = f"{first}。{tail}"
            # Append remaining formatted_texts after the first two, if any
            if len(formatted_texts) > 2:
                rest = ' '.join(formatted_texts[2:]).strip()
                if rest:
                    merged = merged.rstrip('。') + '、' + rest
            return merged

        # Default: join by space
        return ' '.join([t for t in formatted_texts if t]).strip()
