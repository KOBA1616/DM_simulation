# -*- coding: cp932 -*-
from typing import Dict, Any, List, Tuple
from dm_toolkit.gui.localization import tr

class CardTextGenerator:
    """
    Generates Japanese rule text for Duel Masters cards based on JSON data.
    """

    # Maps kept for reverse lookups or specific logic, but tr() should be preferred for output
    CIVILIZATION_MAP = {
        "LIGHT": "\x8c\xf5",
        "WATER": "\x90\x85",
        "DARKNESS": "\x88\xc5",
        "FIRE": "\x89\xce",
        "NATURE": "\x8e\xa9\x91R",
        "ZERO": "\x83[\x83\x8d"
    }

    TYPE_MAP = {
        "CREATURE": "\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "SPELL": "\x8e\xf4\x95\xb6",
        "CROSS_GEAR": "\x83N\x83\x8d\x83X\x83M\x83A",
        "CASTLE": "\x8f\xe9",
        "EVOLUTION_CREATURE": "\x90i\x89\xbb\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "NEO_CREATURE": "NEO\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "PSYCHIC_CREATURE": "\x83T\x83C\x83L\x83b\x83N\x81E\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "PSYCHIC_SUPER_CREATURE": "\x83T\x83C\x83L\x83b\x83N\x81E\x83X\x81[\x83p\x81[\x81E\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "DRAGHEART_CREATURE": "\x83h\x83\x89\x83O\x83n\x81[\x83g\x81E\x83N\x83\x8a\x81[\x83`\x83\x83\x81[",
        "DRAGHEART_WEAPON": "\x83h\x83\x89\x83O\x83n\x81[\x83g\x81E\x83E\x83G\x83|\x83\x93",
        "DRAGHEART_FORTRESS": "\x83h\x83\x89\x83O\x83n\x81[\x83g\x81E\x83t\x83H\x81[\x83g\x83\x8c\x83X",
        "AURA": "\x83I\x81[\x83\x89",
        "FIELD": "\x83t\x83B\x81[\x83\x8b\x83h",
        "D2_FIELD": "D2\x83t\x83B\x81[\x83\x8b\x83h",
    }

    # Backup map if tr() returns the key itself
    KEYWORD_TRANSLATION = {
        "speed_attacker": "\x83X\x83s\x81[\x83h\x83A\x83^\x83b\x83J\x81[",
        "blocker": "\x83u\x83\x8d\x83b\x83J\x81[",
        "slayer": "\x83X\x83\x8c\x83C\x83\x84\x81[",
        "double_breaker": "W\x81E\x83u\x83\x8c\x83C\x83J\x81[",
        "triple_breaker": "T\x81E\x83u\x83\x8c\x83C\x83J\x81[",
        "world_breaker": "\x83\x8f\x81[\x83\x8b\x83h\x81E\x83u\x83\x8c\x83C\x83J\x81[",
        "shield_trigger": "S\x81E\x83g\x83\x8a\x83K\x81[",
        "evolution": "\x90i\x89\xbb",
        "just_diver": "\x83W\x83\x83\x83X\x83g\x83_\x83C\x83o\x81[",
        "mach_fighter": "\x83}\x83b\x83n\x83t\x83@\x83C\x83^\x81[",
        "g_strike": "G\x81E\x83X\x83g\x83\x89\x83C\x83N",
        "hyper_energy": "\x83n\x83C\x83p\x81[\x83G\x83i\x83W\x81[",
        "shield_burn": "\x83V\x81[\x83\x8b\x83h\x8fċp",
        "revolution_change": "\x8av\x96\xbd\x83`\x83F\x83\x93\x83W",
        "untap_in": "\x83^\x83b\x83v\x82\xb5\x82ďo\x82\xe9",
        "meta_counter_play": "\x83\x81\x83^\x83J\x83E\x83\x93\x83^\x81[",
        "power_attacker": "\x83p\x83\x8f\x81[\x83A\x83^\x83b\x83J\x81[",
        "g_zero": "G\x81E\x83[\x83\x8d",
        "ex_life": "EX\x83\x89\x83C\x83t",
        "unblockable": "\x83u\x83\x8d\x83b\x83N\x82\xb3\x82\xea\x82Ȃ\xa2"
    }

    ACTION_MAP = {
        "DRAW_CARD": "\x83J\x81[\x83h\x82\xf0{value1}\x96\x87\x88\xf8\x82\xad\x81B",
        "ADD_MANA": "\x8e\xa9\x95\xaa\x82̎R\x8eD\x82̏ォ\x82\xe7{value1}\x96\x87\x82\xf0\x83}\x83i\x83]\x81[\x83\x93\x82ɒu\x82\xad\x81B",
        "DESTROY": "{target}\x82\xf0{value1}{unit}\x94j\x89󂷂\xe9\x81B",
        "TAP": "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x83^\x83b\x83v\x82\xb7\x82\xe9\x81B",
        "UNTAP": "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x83A\x83\x93\x83^\x83b\x83v\x82\xb7\x82\xe9\x81B",
        "RETURN_TO_HAND": "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x8e\xe8\x8eD\x82ɖ߂\xb7\x81B",
        "SEND_TO_MANA": "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x83}\x83i\x83]\x81[\x83\x93\x82ɒu\x82\xad\x81B",
        "MODIFY_POWER": "{target}\x82̃p\x83\x8f\x81[\x82\xf0{value1}\x82\xb7\x82\xe9\x81B",
        "BREAK_SHIELD": "\x91\x8a\x8e\xe8\x82̃V\x81[\x83\x8b\x83h\x82\xf0{value1}\x82u\x83\x8c\x83C\x83N\x82\xb7\x82\xe9\x81B",
        "LOOK_AND_ADD": "\x8e\xa9\x95\xaa\x82̎R\x8eD\x82̏ォ\x82\xe7{value1}\x96\x87\x82\xf0\x8c\xa9\x82\xe9\x81B\x82\xbb\x82̒\x86\x82\xa9\x82\xe7{value2}\x96\x87\x82\xf0\x8e\xe8\x8eD\x82ɉ\xc1\x82\xa6\x81A\x8ec\x82\xe8\x82\xf0{zone}\x82ɒu\x82\xad\x81B",
        "SEARCH_DECK_BOTTOM": "\x8e\xa9\x95\xaa\x82̎R\x8eD\x82̉\xba\x82\xa9\x82\xe7{value1}\x96\x87\x82\xf0\x8c\xa9\x82\xe9\x81B",
        "SEARCH_DECK": "\x8e\xa9\x95\xaa\x82̎R\x8eD\x82\xf0\x8c\xa9\x82\xe9\x81B\x82\xbb\x82̒\x86\x82\xa9\x82\xe7{filter}\x82\xf01\x96\x87\x91I\x82сA{zone}\x82ɒu\x82\xad\x81B\x82\xbb\x82̌\xe3\x81A\x8eR\x8eD\x82\xf0\x83V\x83\x83\x83b\x83t\x83\x8b\x82\xb7\x82\xe9\x81B",
        "MEKRAID": "\x83\x81\x83N\x83\x8c\x83C\x83h{value1}",
        "DISCARD": "\x8e\xe8\x8eD\x82\xf0{value1}\x96\x87\x8êĂ\xe9\x81B",
        "PLAY_FROM_ZONE": "{source_zone}\x82\xa9\x82\xe7\x83R\x83X\x83g{value1}\x88ȉ\xba\x82\xcc{target}\x82\xf0\x83v\x83\x8c\x83C\x82\xb5\x82Ă\xe0\x82悢\x81B",
        "COUNT_CARDS": "\x81i{filter}\x82̐\x94\x82𐔂\xa6\x82\xe9\x81j",
        "GET_GAME_STAT": "\x81i{str_val}\x82\xf0\x8eQ\x8fƁj",
        "REVEAL_CARDS": "\x8eR\x8eD\x82̏ォ\x82\xe7{value1}\x96\x87\x82\xf0\x95\\x8c\xfc\x82\xab\x82ɂ\xb7\x82\xe9\x81B",
        "SHUFFLE_DECK": "\x8eR\x8eD\x82\xf0\x83V\x83\x83\x83b\x83t\x83\x8b\x82\xb7\x82\xe9\x81B",
        "ADD_SHIELD": "\x8eR\x8eD\x82̏ォ\x82\xe7{value1}\x96\x87\x82\xf0\x83V\x81[\x83\x8b\x83h\x89\xbb\x82\xb7\x82\xe9\x81B",
        "SEND_SHIELD_TO_GRAVE": "\x91\x8a\x8e\xe8\x82̃V\x81[\x83\x8b\x83h\x82\xf0{value1}\x82I\x82сA\x95\xe6\x92n\x82ɒu\x82\xad\x81B",
        "SEND_TO_DECK_BOTTOM": "{target}\x82\xf0{value1}\x96\x87\x81A\x8eR\x8eD\x82̉\xba\x82ɒu\x82\xad\x81B",
        "CAST_SPELL": "{target}\x82\xf0\x83R\x83X\x83g\x82\xf0\x8ex\x95\xa5\x82킸\x82ɏ\xa5\x82\xa6\x82\xe9\x81B",
        "PUT_CREATURE": "{target}\x82\xf0\x83o\x83g\x83\x8b\x83]\x81[\x83\x93\x82ɏo\x82\xb7\x81B",
        "GRANT_KEYWORD": "{target}\x82Ɂu{str_val}\x81v\x82\xf0\x97^\x82\xa6\x82\xe9\x81B",
        "MOVE_CARD": "{target}\x82\xf0{zone}\x82ɒu\x82\xad\x81B",
        "COST_REFERENCE": "",
        "SUMMON_TOKEN": "\x81u{str_val}\x81v\x82\xf0{value1}\x91̏o\x82\xb7\x81B",
        "RESET_INSTANCE": "{target}\x82̏\xf3\x91Ԃ\xf0\x83\x8a\x83Z\x83b\x83g\x82\xb7\x82\xe9\x81i\x83A\x83\x93\x83^\x83b\x83v\x93\x99\x81j\x81B",
        "REGISTER_DELAYED_EFFECT": "\x81u{str_val}\x81v\x82̌\xf8\x89ʂ\xf0{value1}\x83^\x81[\x83\x93\x93o\x98^\x82\xb7\x82\xe9\x81B",
        "FRIEND_BURST": "{str_val}\x82̃t\x83\x8c\x83\x93\x83h\x81E\x83o\x81[\x83X\x83g",
        "MOVE_TO_UNDER_CARD": "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x83J\x81[\x83h\x82̉\xba\x82ɒu\x82\xad\x81B",
        "SELECT_NUMBER": "\x90\x94\x8e\x9a\x82\xf01\x82I\x82ԁB",
        "DECLARE_NUMBER": "{value1}\x81`{value2}\x82̐\x94\x8e\x9a\x82\xf01\x82錾\x82\xb7\x82\xe9\x81B",
        "COST_REDUCTION": "{target}\x82̃R\x83X\x83g\x82\xf0{value1}\x8f\xad\x82Ȃ\xad\x82\xb7\x82\xe9\x81B\x82\xbd\x82\xbe\x82\xb5\x81A\x83R\x83X\x83g\x82\xcd0\x88ȉ\xba\x82ɂ͂Ȃ\xe7\x82Ȃ\xa2\x81B",
        "LOOK_TO_BUFFER": "{source_zone}\x82\xa9\x82\xe7{value1}\x96\x87\x82\xf0\x8c\xa9\x82\xe9\x81i\x83o\x83b\x83t\x83@\x82ցj\x81B",
        "SELECT_FROM_BUFFER": "\x83o\x83b\x83t\x83@\x82\xa9\x82\xe7{value1}\x96\x87\x91I\x82ԁi{filter}\x81j\x81B",
        "PLAY_FROM_BUFFER": "\x83o\x83b\x83t\x83@\x82\xa9\x82\xe7\x83v\x83\x8c\x83C\x82\xb7\x82\xe9\x81B",
        "MOVE_BUFFER_TO_ZONE": "\x83o\x83b\x83t\x83@\x82\xa9\x82\xe7{zone}\x82ɒu\x82\xad\x81B",
        "SELECT_OPTION": "\x8e\x9f\x82̒\x86\x82\xa9\x82\xe7\x91I\x82ԁB",
        "LOCK_SPELL": "\x91\x8a\x8e\xe8\x82͎\xf4\x95\xb6\x82\xf0\x8f\xa5\x82\xa6\x82\xe7\x82\xea\x82Ȃ\xa2\x81B",
        "APPLY_MODIFIER": "\x8c\xf8\x89ʂ\xf0\x95t\x97^\x82\xb7\x82\xe9\x81B",
    }

    STAT_KEY_MAP = {
        "MANA_COUNT": ("\x83}\x83i\x83]\x81[\x83\x93\x82̃J\x81[\x83h", "\x96\x87"),
        "CREATURE_COUNT": ("\x83N\x83\x8a\x81[\x83`\x83\x83\x81[", "\x91\xcc"),
        "SHIELD_COUNT": ("\x83V\x81[\x83\x8b\x83h", "\x82\xc2"),
        "HAND_COUNT": ("\x8e\xe8\x8eD", "\x96\x87"),
        "GRAVEYARD_COUNT": ("\x95\xe6\x92n\x82̃J\x81[\x83h", "\x96\x87"),
        "BATTLE_ZONE_COUNT": ("\x83o\x83g\x83\x8b\x83]\x81[\x83\x93\x82̃J\x81[\x83h", "\x96\x87"),
        "OPPONENT_MANA_COUNT": ("\x91\x8a\x8e\xe8\x82̃}\x83i\x83]\x81[\x83\x93\x82̃J\x81[\x83h", "\x96\x87"),
        "OPPONENT_CREATURE_COUNT": ("\x91\x8a\x8e\xe8\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[", "\x91\xcc"),
        "OPPONENT_SHIELD_COUNT": ("\x91\x8a\x8e\xe8\x82̃V\x81[\x83\x8b\x83h", "\x82\xc2"),
        "OPPONENT_HAND_COUNT": ("\x91\x8a\x8e\xe8\x82̎\xe8\x8eD", "\x96\x87"),
        "OPPONENT_GRAVEYARD_COUNT": ("\x91\x8a\x8e\xe8\x82̕\xe6\x92n\x82̃J\x81[\x83h", "\x96\x87"),
        "OPPONENT_BATTLE_ZONE_COUNT": ("\x91\x8a\x8e\xe8\x82̃o\x83g\x83\x8b\x83]\x81[\x83\x93\x82̃J\x81[\x83h", "\x96\x87"),
        "CARDS_DRAWN_THIS_TURN": ("\x82\xb1\x82̃^\x81[\x83\x93\x82Ɉ\xf8\x82\xa2\x82\xbd\x83J\x81[\x83h", "\x96\x87"),
        "MANA_CIVILIZATION_COUNT": ("\x83}\x83i\x83]\x81[\x83\x93\x82̕\xb6\x96\xbe\x90\x94", ""),
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
        name = data.get("name", "Unknown")
        cost = data.get("cost", 0)

        # Handle both list and string formats for civilization
        civs_data = data.get("civilizations", [])
        if not civs_data and "civilization" in data:
            civ_single = data.get("civilization")
            if civ_single:
                civs_data = [civ_single]
        civs = cls._format_civs(civs_data)

        type_str = tr(data.get("type", "CREATURE"))
        races = " / ".join(data.get("races", []))

        header = f"\x81y{name}\x81z {civs} \x83R\x83X\x83g{cost}"
        if races:
            header += f" {races}"
        lines.append(header)
        lines.append(f"[{type_str}]")

        power = data.get("power", 0)
        if power > 0:
             lines.append(f"\x83p\x83\x8f\x81[ {power}")

        lines.append("-" * 20)

        # 2. Keywords
        keywords = data.get("keywords", {})
        kw_lines = []
        if keywords:
            for k, v in keywords.items():
                if v:
                    # Try explicit map first, then tr()
                    kw_str = cls.KEYWORD_TRANSLATION.get(k, tr(k))

                    if k == "power_attacker":
                        bonus = data.get("power_attacker_bonus", 0)
                        if bonus > 0:
                            kw_str += f" +{bonus}"
                    elif k == "revolution_change":
                        cond = data.get("revolution_change_condition", {})
                        if cond:
                            cond_text = cls._describe_simple_filter(cond)
                            kw_str += f"\x81F{cond_text}"
                            kw_str += f"\x81i\x8e\xa9\x95\xaa\x82\xcc{cond_text}\x82\xaa\x8dU\x8c\x82\x82\xb7\x82鎞\x81A\x82\xbb\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82Ǝ\xe8\x8eD\x82̂\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82Ɠ\xfc\x82\xea\x91ւ\xa6\x82Ă\xe0\x82悢\x81j"
                    elif k == "hyper_energy":
                        kw_str += "\x81i\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf0\x8f\xa2\x8a\xab\x82\xb7\x82鎞\x81A\x83R\x83X\x83g\x82̑\xe3\x82\xed\x82\xe8\x82Ɏ\xa9\x95\xaa\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf02\x91̃^\x83b\x83v\x82\xb5\x82Ă\xe0\x82悢\x81j"
                    elif k == "just_diver":
                        kw_str += "\x81i\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e\x81A\x8e\x9f\x82̎\xa9\x95\xaa\x82̃^\x81[\x83\x93\x82̂͂\xb6\x82߂܂ŁA\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82͑\x8a\x8e\xe8\x82ɑI\x82΂ꂸ\x81A\x8dU\x8c\x82\x82\xb3\x82\xea\x82Ȃ\xa2\x81j"

                    kw_lines.append(f"\x81\xa1 {kw_str}")
        if kw_lines:
            lines.extend(kw_lines)

        # 2.5 Cost Reductions
        cost_reductions = data.get("cost_reductions", [])
        for cr in cost_reductions:
            text = cls._format_cost_reduction(cr)
            if text:
                lines.append(f"\x81\xa1 {text}")

        # 2.6 Reaction Abilities
        reactions = data.get("reaction_abilities", [])
        for r in reactions:
            text = cls._format_reaction(r)
            if text:
                lines.append(f"\x81\xa1 {text}")

        # 3. Effects
        effects = data.get("effects", [])
        is_spell = data.get("type", "CREATURE") == "SPELL"

        for effect in effects:
            text = cls._format_effect(effect, is_spell)
            if text:
                lines.append(f"\x81\xa1 {text}")

        # 3.5 Metamorph Abilities (Ultra Soul Cross, etc.)
        metamorphs = data.get("metamorph_abilities", [])
        if metamorphs:
            lines.append("\x81y\x92ǉ\xc1\x94\\x97́z")
            for effect in metamorphs:
                 text = cls._format_effect(effect, is_spell)
                 if text:
                     lines.append(f"\x81\xa1 {text}")

        # 4. Twinpact (Spell Side)
        spell_side = data.get("spell_side")
        if spell_side and include_twinpact:
            lines.append("\n" + "=" * 20 + " \x8e\xf4\x95\xb6\x91\xa4 " + "=" * 20 + "\n")
            lines.append(cls.generate_text(spell_side))

        return "\n".join(lines)

    @classmethod
    def _format_civs(cls, civs: List[str]) -> str:
        if not civs:
            return "\x96\xb3\x90F"
        return "/".join([tr(c) for c in civs])

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
            adjectives.append(f"\x83R\x83X\x83g{min_cost}\x88ȏ\xe3")
        if max_cost < 999:
            adjectives.append(f"\x83R\x83X\x83g{max_cost}\x88ȉ\xba")

        adj_str = "\x82\xcc".join(adjectives)
        if adj_str:
            adj_str += "\x82\xcc"

        noun_str = "\x83N\x83\x8a\x81[\x83`\x83\x83\x81["
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
             return f"\x83j\x83\x93\x83W\x83\x83\x81E\x83X\x83g\x83\x89\x83C\x83N {cost}"
        elif rtype == "STRIKE_BACK":
             return "\x83X\x83g\x83\x89\x83C\x83N\x81E\x83o\x83b\x83N"
        elif rtype == "REVOLUTION_0_TRIGGER":
             return "\x8av\x96\xbd0\x83g\x83\x8a\x83K\x81["
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
        return f"\x83R\x83X\x83g\x8cy\x8c\xb8: {desc}"

    @classmethod
    def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False) -> str:
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
                trigger_text = f"{base_cond}\x81A{trigger_text}" # \x8e\xa9\x95\xaa\x82̃^\x81[\x83\x93\x92\x86\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"\x91\x8a\x8e肪\x83J\x81[\x83h\x82\xf0\x88\xf8\x82\xa2\x82\xbd\x8e\x9e\x81A{val}\x96\x87\x96ڈȍ~\x82Ȃ\xe7"
                cond_text = ""

        action_texts = []
        for action in actions:
            action_texts.append(cls._format_action(action))

        full_action_text = " ".join(action_texts).strip()

        # If it's a Spell's main effect (ON_PLAY), we can often omit the trigger text "Played/Cast"
        # unless there's a condition or it's a specific sub-trigger.
        # But usually spells just list the effect. S-Trigger is handled as a Keyword (mostly),
        # but in legacy JSON it might be in effects? No, keywords.
        # However, "S-Trigger" is displayed via keywords.
        # If trigger is ON_PLAY and is_spell is True, we might suppress "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e"
        # but if it was mapped to "\x8e\xf4\x95\xb6\x82\xf0\x8f\xa5\x82\xa6\x82\xbd\x8e\x9e", we might keep it?
        # Standard duel masters text: Spells don't say "When you cast this spell" for the main effect.
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
            "ON_PLAY": "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e" if not is_spell else "\x82\xb1\x82̎\xf4\x95\xb6\x82\xf0\x8f\xa5\x82\xa6\x82\xbd\x8e\x9e", # Suppressed later for main spell effect
            "ON_OTHER_ENTER": "\x91\xbc\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e",
            "AT_ATTACK": "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8dU\x8c\x82\x82\xb7\x82鎞",
            "ON_DESTROY": "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x94j\x89󂳂ꂽ\x8e\x9e",
            "AT_END_OF_TURN": "\x8e\xa9\x95\xaa\x82̃^\x81[\x83\x93\x82̏I\x82\xed\x82\xe8\x82\xc9",
            "AT_END_OF_OPPONENT_TURN": "\x91\x8a\x8e\xe8\x82̃^\x81[\x83\x93\x82̏I\x82\xed\x82\xe8\x82\xc9",
            "ON_BLOCK": "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x83u\x83\x8d\x83b\x83N\x82\xb5\x82\xbd\x8e\x9e",
            "ON_ATTACK_FROM_HAND": "\x8e\xe8\x8eD\x82\xa9\x82\xe7\x8dU\x8c\x82\x82\xb7\x82鎞",
            "TURN_START": "\x8e\xa9\x95\xaa\x82̃^\x81[\x83\x93\x82̂͂\xb6\x82߂\xc9",
            "S_TRIGGER": "S\x81E\x83g\x83\x8a\x83K\x81[",
            "PASSIVE_CONST": "\x81i\x8f\xed\x8d݌\xf8\x89ʁj",
            "ON_SHIELD_ADD": "\x83J\x81[\x83h\x82\xaa\x83V\x81[\x83\x8b\x83h\x83]\x81[\x83\x93\x82ɒu\x82\xa9\x82ꂽ\x8e\x9e",
            "AT_BREAK_SHIELD": "\x83V\x81[\x83\x8b\x83h\x82\xf0\x83u\x83\x8c\x83C\x83N\x82\xb7\x82鎞",
            "ON_CAST_SPELL": "\x8e\xf4\x95\xb6\x82\xf0\x8f\xa5\x82\xa6\x82\xbd\x8e\x9e",
            "ON_OPPONENT_DRAW": "\x91\x8a\x8e肪\x83J\x81[\x83h\x82\xf0\x88\xf8\x82\xa2\x82\xbd\x8e\x9e",
            "NONE": ""
        }
        return mapping.get(trigger, trigger)

    @classmethod
    def _format_condition(cls, condition: Dict[str, Any]) -> str:
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        if cond_type == "MANA_ARMED":
            val = condition.get("value", 0)
            civ_raw = condition.get("str_val", "")
            civ = tr(civ_raw)
            return f"\x83}\x83i\x95\x90\x91\x95 {val} ({civ}): "

        elif cond_type == "SHIELD_COUNT":
            val = condition.get("value", 0)
            op = condition.get("op", ">=")
            op_text = "\x88ȏ\xe3" if op == ">=" else "\x88ȉ\xba" if op == "<=" else ""
            if op == "=": op_text = ""
            return f"\x8e\xa9\x95\xaa\x82̃V\x81[\x83\x8b\x83h\x82\xaa{val}\x82\xc2{op_text}\x82Ȃ\xe7: "

        elif cond_type == "CIVILIZATION_MATCH":
             return "\x83}\x83i\x83]\x81[\x83\x93\x82ɓ\xaf\x82\xb6\x95\xb6\x96\xbe\x82\xaa\x82\xa0\x82\xea\x82\xce: "

        elif cond_type == "COMPARE_STAT":
             key = condition.get("stat_key", "")
             op = condition.get("op", "=")
             val = condition.get("value", 0)
             stat_name, unit = cls.STAT_KEY_MAP.get(key, (key, ""))

             op_text = ""
             if op == ">=":
                 op_text = f"{val}{unit}\x88ȏ\xe3"
             elif op == "<=":
                 op_text = f"{val}{unit}\x88ȉ\xba"
             elif op == "=" or op == "==":
                 op_text = f"{val}{unit}"
             elif op == ">":
                 op_text = f"{val}{unit}\x82\xe6\x82葽\x82\xa2"
             elif op == "<":
                 op_text = f"{val}{unit}\x82\xe6\x82菭\x82Ȃ\xa2"
             return f"\x8e\xa9\x95\xaa\x82\xcc{stat_name}\x82\xaa{op_text}\x82Ȃ\xe7: "

        elif cond_type == "OPPONENT_PLAYED_WITHOUT_MANA":
            return "\x91\x8a\x8e肪\x83}\x83i\x83]\x81[\x83\x93\x82̃J\x81[\x83h\x82\xf0\x83^\x83b\x83v\x82\xb9\x82\xb8\x82ɁA\x83N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf0\x8fo\x82\xb7\x82\xa9\x8e\xf4\x95\xb6\x82\xf0\x8f\xa5\x82\xa6\x82\xbd\x8e\x9e: "

        elif cond_type == "OPPONENT_DRAW_COUNT":
            val = condition.get("value", 0)
            return f"{val}\x96\x87\x96ڈȍ~\x82Ȃ\xe7: "

        elif cond_type == "DURING_YOUR_TURN":
            return "\x8e\xa9\x95\xaa\x82̃^\x81[\x83\x93\x92\x86: "
        elif cond_type == "DURING_OPPONENT_TURN":
            return "\x91\x8a\x8e\xe8\x82̃^\x81[\x83\x93\x92\x86: "
        elif cond_type == "EVENT_FILTER_MATCH":
            return ""

        return ""

    @classmethod
    def _format_action(cls, action: Dict[str, Any]) -> str:
        if not action:
            return ""

        atype = action.get("type", "NONE")
        template = cls.ACTION_MAP.get(atype, "")

        # Determine verb form (standard or optional)
        optional = action.get("optional", False)

        # Resolve dynamic target strings
        target_str, unit = cls._resolve_target(action)

        # Complex Action Logic
        if atype == "MODIFY_POWER":
            val = action.get("value1", 0)
            sign = "+" if val >= 0 else ""
            return f"{target_str}\x82̃p\x83\x8f\x81[\x82\xf0{sign}{val}\x82\xb7\x82\xe9\x81B"

        elif atype == "SELECT_NUMBER":
            val1 = action.get("value1", 0)
            val2 = action.get("value2", 0)
            if val1 > 0 and val2 > 0:
                 template = f"{val1}\x81`{val2}\x82̐\x94\x8e\x9a\x82\xf01\x82I\x82ԁB"

        elif atype == "COST_REFERENCE":
            str_val = action.get("str_val", "")
            if str_val == "G_ZERO":
                cond = action.get("condition", {})
                cond_text = cls._format_condition(cond).strip().rstrip(':')
                return f"G\x81E\x83[\x83\x8d\x81F{cond_text}\x81i\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf0\x83R\x83X\x83g\x82\xf0\x8ex\x95\xa5\x82킸\x82ɏ\xa2\x8a\xab\x82\xb5\x82Ă\xe0\x82悢\x81j"
            elif str_val == "HYPER_ENERGY":
                return "\x83n\x83C\x83p\x81[\x83G\x83i\x83W\x81["
            elif str_val in ["SYM_CREATURE", "SYM_SPELL", "SYM_SHIELD"]:
                val1 = action.get("value1", 0)
                cost_term = "\x8f\xa2\x8a\xab\x83R\x83X\x83g" if "CREATURE" in str_val else "\x83R\x83X\x83g"
                return f"{target_str}1{unit}\x82ɂ\xab\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xcc{cost_term}\x82\xf0{val1}\x8f\xad\x82Ȃ\xad\x82\xb7\x82\xe9\x81B\x82\xbd\x82\xbe\x82\xb5\x81A\x83R\x83X\x83g\x82\xcd0\x88ȉ\xba\x82ɂ͂Ȃ\xe7\x82Ȃ\xa2\x81B"
            elif str_val == "CARDS_DRAWN_THIS_TURN":
                val1 = action.get("value1", 0)
                return f"\x82\xb1\x82̃^\x81[\x83\x93\x82Ɉ\xf8\x82\xa2\x82\xbd\x83J\x81[\x83h1\x96\x87\x82ɂ\xab\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82̏\xa2\x8a\xab\x83R\x83X\x83g\x82\xf0{val1}\x8f\xad\x82Ȃ\xad\x82\xb7\x82\xe9\x81B\x82\xbd\x82\xbe\x82\xb5\x81A\x83R\x83X\x83g\x82\xcd0\x88ȉ\xba\x82ɂ͂Ȃ\xe7\x82Ȃ\xa2\x81B"
            else:
                filter_def = action.get("filter")
                if filter_def:
                     val1 = action.get("value1", 0)
                     return f"{target_str}1{unit}\x82ɂ\xab\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82̃R\x83X\x83g\x82\xf0{val1}\x8f\xad\x82Ȃ\xad\x82\xb7\x82\xe9\x81B\x82\xbd\x82\xbe\x82\xb5\x81A\x83R\x83X\x83g\x82\xcd0\x88ȉ\xba\x82ɂ͂Ȃ\xe7\x82Ȃ\xa2\x81B"
                return "\x83R\x83X\x83g\x82\xf0\x8cy\x8c\xb8\x82\xb7\x82\xe9\x81B"

        elif atype == "SELECT_OPTION":
            options = action.get("options", [])
            lines = []
            val1 = action.get("value1", 1)
            lines.append(f"\x8e\x9f\x82̒\x86\x82\xa9\x82\xe7{val1}\x89\xf1\x91I\x82ԁB\x81i\x93\xaf\x82\xb6\x82\xe0\x82̂\xf0\x91I\x82\xf1\x82ł\xe0\x82悢\x81j")
            for i, opt_chain in enumerate(options):
                chain_text = " ".join([cls._format_action(a) for a in opt_chain])
                lines.append(f"> {chain_text}")
            return "\n".join(lines)

        elif atype == "MEKRAID":
            val1 = action.get("value1", 0)
            # Add simple civ detection if possible, otherwise generic
            return f"\x83\x81\x83N\x83\x8c\x83C\x83h{val1}\x81i\x8e\xa9\x95\xaa\x82̎R\x8eD\x82̏ォ\x82\xe73\x96\x87\x82\xf0\x8c\xa9\x82\xe9\x81B\x82\xbb\x82̒\x86\x82\xa9\x82\xe7\x83R\x83X\x83g{val1}\x88ȉ\xba\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf01\x91́A\x83R\x83X\x83g\x82\xf0\x8ex\x95\xa5\x82킸\x82ɏ\xa2\x8a\xab\x82\xb5\x82Ă\xe0\x82悢\x81B\x8ec\x82\xe8\x82\xf0\x8eR\x8eD\x82̉\xba\x82ɍD\x82\xab\x82ȏ\x87\x8f\x98\x82Œu\x82\xad\x81j"

        elif atype == "FRIEND_BURST":
            str_val = action.get("str_val", "")
            return f"\x81\x83{str_val}\x81\x84\x82̃t\x83\x8c\x83\x93\x83h\x81E\x83o\x81[\x83X\x83g\x81i\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xaa\x8fo\x82\xbd\x8e\x9e\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82\xf0\x83^\x83b\x83v\x82\xb5\x82Ă\xe0\x82悢\x81B\x82\xbb\x82\xa4\x82\xb5\x82\xbd\x82\xe7\x81A\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[\x82̎\xf4\x95\xb6\x91\xa4\x82\xf0\x81A\x83o\x83g\x83\x8b\x83]\x81[\x83\x93\x82ɒu\x82\xa2\x82\xbd\x82܂܃R\x83X\x83g\x82\xf0\x8ex\x95\xa5\x82킸\x82ɏ\xa5\x82\xa6\x82\xe9\x81B\x81j"

        elif atype == "REVOLUTION_CHANGE":
             return "" # Covered by keyword

        elif atype == "APPLY_MODIFIER":
             # Need to know WHAT modifier
             str_val = action.get("str_val", "")
             val1 = action.get("value1", 0)
             if str_val == "SPEED_ATTACKER":
                 return f"{target_str}\x82Ɂu\x83X\x83s\x81[\x83h\x83A\x83^\x83b\x83J\x81[\x81v\x82\xf0\x97^\x82\xa6\x82\xe9\x81B"
             elif str_val == "BLOCKER":
                 return f"{target_str}\x82Ɂu\x83u\x83\x8d\x83b\x83J\x81[\x81v\x82\xf0\x97^\x82\xa6\x82\xe9\x81B"
             elif str_val == "SLAYER":
                 return f"{target_str}\x82Ɂu\x83X\x83\x8c\x83C\x83\x84\x81[\x81v\x82\xf0\x97^\x82\xa6\x82\xe9\x81B"
             elif str_val == "COST":
                 sign = "\x8f\xad\x82Ȃ\xad\x82\xb7\x82\xe9" if val1 > 0 else "\x91\x9d\x82₷"
                 return f"{target_str}\x82̃R\x83X\x83g\x82\xf0{abs(val1)}{sign}\x81B"
             else:
                 return f"{target_str}\x82Ɍ\xf8\x89ʁi{str_val}\x81j\x82\xf0\x97^\x82\xa6\x82\xe9\x81B"

        if not template:
            return f"({tr(atype)})"

        # Parameter Substitution
        val1 = action.get("value1", 0)
        val2 = action.get("value2", 0)
        str_val = action.get("str_val", "")
        input_key = action.get("input_value_key", "")

        is_generic_selection = atype in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA", "MOVE_CARD"]

        if input_key:
             val1 = "\x81i\x82\xbb\x82̐\x94\x81j"
        elif val1 == 0 and is_generic_selection:
             # Logic for "All" if 0 and generic
             if atype == "DESTROY": template = "{target}\x82\xf0\x82\xb7\x82ׂĔj\x89󂷂\xe9\x81B"
             elif atype == "TAP": template = "{target}\x82\xf0\x82\xb7\x82ׂă^\x83b\x83v\x82\xb7\x82\xe9\x81B"
             elif atype == "UNTAP": template = "{target}\x82\xf0\x82\xb7\x82ׂăA\x83\x93\x83^\x83b\x83v\x82\xb7\x82\xe9\x81B"
             elif atype == "RETURN_TO_HAND": template = "{target}\x82\xf0\x82\xb7\x82ׂĎ\xe8\x8eD\x82ɖ߂\xb7\x81B"
             elif atype == "SEND_TO_MANA": template = "{target}\x82\xf0\x82\xb7\x82ׂă}\x83i\x83]\x81[\x83\x93\x82ɒu\x82\xad\x81B"
             elif atype == "MOVE_CARD":
                 # Fallback handled in specific logic below, this is just template swap
                 pass

        if atype == "GRANT_KEYWORD":
            str_val = tr(str_val)

        elif atype == "GET_GAME_STAT":
            stat_name = cls.STAT_KEY_MAP.get(str_val, (str_val, ""))[0]
            return f"\x81i{stat_name}\x82\xf0\x8eQ\x8fƁj"

        elif atype == "COUNT_CARDS":
            mode = str_val
            if not mode or mode == "CARDS_MATCHING_FILTER":
                 return f"\x81i{target_str}\x82̐\x94\x82𐔂\xa6\x82\xe9\x81j"
            else:
                 stat_name = cls.STAT_KEY_MAP.get(mode, (mode, ""))[0]
                 return f"\x81i{stat_name}\x82𐔂\xa6\x82\xe9\x81j"

        elif atype == "MOVE_CARD":
            dest_zone = action.get("destination_zone", "")
            # Check for "All" condition
            is_all = (val1 == 0 and not input_key)

            if dest_zone == "HAND":
                template = "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x8e\xe8\x8eD\x82ɖ߂\xb7\x81B"
                if is_all: template = "{target}\x82\xf0\x82\xb7\x82ׂĎ\xe8\x8eD\x82ɖ߂\xb7\x81B"
            elif dest_zone == "MANA_ZONE":
                template = "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x83}\x83i\x83]\x81[\x83\x93\x82ɒu\x82\xad\x81B"
                if is_all: template = "{target}\x82\xf0\x82\xb7\x82ׂă}\x83i\x83]\x81[\x83\x93\x82ɒu\x82\xad\x81B"
            elif dest_zone == "GRAVEYARD":
                template = "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x95\xe6\x92n\x82ɒu\x82\xad\x81B"
                if is_all: template = "{target}\x82\xf0\x82\xb7\x82ׂĕ\xe6\x92n\x82ɒu\x82\xad\x81B"
            elif dest_zone == "DECK_BOTTOM":
                template = "{target}\x82\xf0{value1}{unit}\x91I\x82сA\x8eR\x8eD\x82̉\xba\x82ɒu\x82\xad\x81B"
                if is_all: template = "{target}\x82\xf0\x82\xb7\x82ׂĎR\x8eD\x82̉\xba\x82ɒu\x82\xad\x81B"

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

            # Re-resolve target with cleaned filter
            target_str, unit = cls._resolve_target(action)
            verb = "\x83v\x83\x8c\x83C\x82\xb7\x82\xe9"
            types = temp_filter.get("types", [])
            if "SPELL" in types and "CREATURE" not in types:
                verb = "\x8f\xa5\x82\xa6\x82\xe9"
            elif "CREATURE" in types:
                verb = "\x8f\xa2\x8a\xab\x82\xb7\x82\xe9"

            if action.get("source_zone"):
                template = "{source_zone}\x82\xa9\x82\xe7\x83R\x83X\x83g{value1}\x88ȉ\xba\x82\xcc{target}\x82\xf0" + verb + "\x81B"
            else:
                template = "\x83R\x83X\x83g{value1}\x88ȉ\xba\x82\xcc{target}\x82\xf0" + verb + "\x81B"

        # Destination/Source Resolution
        dest_zone = action.get("destination_zone", "")
        zone_str = tr(dest_zone) if dest_zone else "\x82ǂ\xb1\x82\xa9"
        src_zone = action.get("source_zone", "")
        src_str = tr(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        if "{filter}" in text:
             text = text.replace("{filter}", target_str)

        if atype == "COST_REDUCTION":
            if target_str == "\x83J\x81[\x83h" or target_str == "\x8e\xa9\x95\xaa\x82̃J\x81[\x83h":
                text = text.replace("\x83J\x81[\x83h", "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[")
                text = text.replace("\x8e\xa9\x95\xaa\x82̃J\x81[\x83h", "\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[")
            cond = action.get("condition", {})
            if cond:
                cond_text = cls._format_condition(cond)
                text = f"{cond_text}{text}"

        # Verb Conjugation for Optional Actions
        if optional:
            if text.endswith("\x82\xb7\x82\xe9\x81B"):
                text = text[:-3] + "\x82\xb5\x82Ă\xe0\x82悢\x81B"
            elif text.endswith("\x82\xad\x81B"): # \x88\xf8\x82\xad\x81A\x92u\x82\xad
                text = text[:-2] + "\x82\xa2\x82Ă\xe0\x82悢\x81B" # \x88\xf8\x82\xa2\x82Ă\xe0\x82悢
            elif text.endswith("\x82\xb7\x81B"): # \x96߂\xb7\x81A\x8fo\x82\xb7
                text = text[:-2] + "\x82\xb5\x82Ă\xe0\x82悢\x81B" # \x96߂\xb5\x82Ă\xe0\x82悢
            elif text.endswith("\x82\xe9\x81B"): # \x8c\xa9\x82\xe9\x81A\x8êĂ\xe9\x81A\x8f\xa5\x82\xa6\x82\xe9
                text = text[:-2] + "\x82Ă\xe0\x82悢\x81B"
            elif text.endswith("\x82\xa4\x81B"): # \x8ex\x95\xa5\x82\xa4
                text = text[:-2] + "\x82\xc1\x82Ă\xe0\x82悢\x81B"
            else:
                if not text.endswith("\x82Ă\xe0\x82悢\x81B"):
                    text = text[:-1] + "\x82Ă\xe0\x82悢\x81B"

        return text

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any]) -> Tuple[str, str]:
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
        type_noun = "\x83J\x81[\x83h"
        unit = "\x96\x87"

        if atype == "DISCARD" and scope == "NONE":
             scope = "PLAYER_SELF"
        if atype == "COST_REDUCTION" and not filter_def and scope == "NONE":
             return ("\x82\xb1\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81[", "\x96\x87")

        if scope == "PLAYER_OPPONENT": prefix = "\x91\x8a\x8e\xe8\x82\xcc"
        elif scope == "PLAYER_SELF" or scope == "SELF": prefix = "\x8e\xa9\x95\xaa\x82\xcc"
        elif scope == "ALL_PLAYERS": prefix = "\x82\xb7\x82ׂẴv\x83\x8c\x83C\x83\x84\x81[\x82\xcc"
        elif scope == "RANDOM": prefix = "\x83\x89\x83\x93\x83_\x83\x80\x82\xc8"

        if filter_def:
            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            races = filter_def.get("races", [])
            civs = filter_def.get("civilizations", [])
            owner = filter_def.get("owner", "NONE")

            # Handle explicit owner filter if scope is generic
            if owner == "PLAYER_OPPONENT" and not prefix: prefix = "\x91\x8a\x8e\xe8\x82\xcc"
            elif owner == "PLAYER_SELF" and not prefix: prefix = "\x8e\xa9\x95\xaa\x82\xcc"

            temp_adjs = []
            if civs: temp_adjs.append("/".join([tr(c) for c in civs]))
            if races: temp_adjs.append("/".join(races))

            if temp_adjs: adjectives += "/".join(temp_adjs) + "\x82\xcc"

            if filter_def.get("min_cost", 0) > 0: adjectives += f"\x83R\x83X\x83g{filter_def['min_cost']}\x88ȏ\xe3\x82\xcc"
            if filter_def.get("max_cost", 999) < 999: adjectives += f"\x83R\x83X\x83g{filter_def['max_cost']}\x88ȉ\xba\x82\xcc"

            if filter_def.get("is_tapped", None) is True: adjectives = "\x83^\x83b\x83v\x82\xb3\x82\xea\x82Ă\xa2\x82\xe9" + adjectives
            elif filter_def.get("is_tapped", None) is False: adjectives = "\x83A\x83\x93\x83^\x83b\x83v\x82\xb3\x82\xea\x82Ă\xa2\x82\xe9" + adjectives
            if filter_def.get("is_blocker", None) is True: adjectives = "\x83u\x83\x8d\x83b\x83J\x81[\x82\xf0\x8e\x9d\x82\xc2" + adjectives
            if filter_def.get("is_evolution", None) is True: adjectives = "\x90i\x89\xbb" + adjectives

            if "BATTLE_ZONE" in zones:
                zone_noun = "\x83o\x83g\x83\x8b\x83]\x81[\x83\x93"
                if "CARD" in types:
                    type_noun = "\x83J\x81[\x83h"
                    unit = "\x96\x87"
                elif "ELEMENT" in types:
                    type_noun = "\x83G\x83\x8c\x83\x81\x83\x93\x83g"
                    unit = "\x82\xc2"
                elif "CREATURE" in types or (not types):
                    type_noun = "\x83N\x83\x8a\x81[\x83`\x83\x83\x81["
                    unit = "\x91\xcc"
                elif "CROSS_GEAR" in types:
                    type_noun = "\x83N\x83\x8d\x83X\x83M\x83A"
            elif "MANA_ZONE" in zones:
                zone_noun = "\x83}\x83i\x83]\x81[\x83\x93"
            elif "HAND" in zones:
                zone_noun = "\x8e\xe8\x8eD"
            elif "SHIELD_ZONE" in zones:
                zone_noun = "\x83V\x81[\x83\x8b\x83h\x83]\x81[\x83\x93"
                type_noun = "\x83J\x81[\x83h"
                unit = "\x82\xc2"
            elif "GRAVEYARD" in zones:
                zone_noun = "\x95\xe6\x92n"
                if "CREATURE" in types:
                     type_noun = "\x83N\x83\x8a\x81[\x83`\x83\x83\x81["
                     unit = "\x91\xcc"
                elif "SPELL" in types:
                     type_noun = "\x8e\xf4\x95\xb6"
            elif "DECK" in zones:
                 zone_noun = "\x8eR\x8eD"

            if not zone_noun:
                if "CARD" in types:
                    type_noun = "\x83J\x81[\x83h"
                    unit = "\x96\x87"
                elif "ELEMENT" in types:
                    type_noun = "\x83G\x83\x8c\x83\x81\x83\x93\x83g"
                    unit = "\x82\xc2"
                elif "CREATURE" in types:
                    type_noun = "\x83N\x83\x8a\x81[\x83`\x83\x83\x81["
                    unit = "\x91\xcc"
                elif "SPELL" in types:
                    type_noun = "\x8e\xf4\x95\xb6"
                elif len(types) > 1:
                     # Join multiple types (e.g., Creature/Spell)
                     type_noun = "\x82܂\xbd\x82\xcd".join([tr(t) for t in types])

            # Special case for SEARCH_DECK
            if atype == "SEARCH_DECK":
                 # Usually searching for a specific card in deck
                 pass

            parts = []
            if prefix: parts.append(prefix)
            if zone_noun: parts.append(zone_noun + "\x82\xcc")
            if adjectives: parts.append(adjectives)
            parts.append(type_noun)
            target_desc = "".join(parts)

            if "SHIELD_ZONE" in zones and (not types or "CARD" in types):
                target_desc = target_desc.replace("\x83V\x81[\x83\x8b\x83h\x83]\x81[\x83\x93\x82̃J\x81[\x83h", "\x83V\x81[\x83\x8b\x83h")
                unit = "\x82\xc2"
            if "BATTLE_ZONE" in zones:
                 target_desc = target_desc.replace("\x83o\x83g\x83\x8b\x83]\x81[\x83\x93\x82\xcc", "")

        else:
            if atype == "DESTROY":
                 if scope == "PLAYER_OPPONENT" or scope == "OPPONENT":
                     target_desc = "\x91\x8a\x8e\xe8\x82̃N\x83\x8a\x81[\x83`\x83\x83\x81["
                     unit = "\x91\xcc"
            elif atype == "TAP" or atype == "UNTAP":
                 if "\x83N\x83\x8a\x81[\x83`\x83\x83\x81[" not in target_desc:
                      target_desc = prefix + "\x83N\x83\x8a\x81[\x83`\x83\x83\x81["
                      unit = "\x91\xcc"
            elif atype == "DISCARD":
                 target_desc = "\x8e\xe8\x8eD"
            else:
                 target_desc = "\x83J\x81[\x83h"

        if not target_desc: target_desc = "\x83J\x81[\x83h"
        return target_desc, unit
