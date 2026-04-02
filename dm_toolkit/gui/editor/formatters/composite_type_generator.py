from typing import List, Dict, Any

class CompositeTypeGenerator:
    """
    Handles generation of proper DM composite types (e.g., ドラゴン・エレメント, タマシード/クリーチャー)
    from lists of types and races, rather than simplistically joining them with "の".
    """

    # Types that should be joined with "・" to races (e.g. ドラゴン・エレメント, ドラゴン・クリーチャー)
    COMPOSITE_RACE_TYPES = {"エレメント", "クリーチャー"}

    @classmethod
    def format_types_and_races(cls, types: List[str], races: List[str]) -> str:
        if not types and not races:
            return "カード"

        type_str = ""
        if types:
            # Join multiple types with "/"
            from dm_toolkit.gui.i18n import tr
            from dm_toolkit.consts import CardType
            words = []
            for t in types:
                if t == CardType.CREATURE.value: words.append("クリーチャー")
                elif t == CardType.SPELL.value: words.append("呪文")
                elif t == CardType.ELEMENT.value: words.append("エレメント")
                else: words.append(tr(t) if tr(t) else "カード")
            type_str = "/".join(words)
        else:
            type_str = "カード" # Default fallback if only races exist but no type. Actually, often just race.

        race_str = "/".join(races) if races else ""

        if not type_str or type_str == "カード":
            return race_str if race_str else type_str

        if not race_str:
            return type_str

        # If both exist, determine joiner
        # E.g. "ドラゴン" and "エレメント" -> "ドラゴン・エレメント"
        # If type is not in COMPOSITE_RACE_TYPES, fallback to "の" (e.g. ドラゴンの呪文)

        # Check if type_str consists entirely of COMPOSITE_RACE_TYPES
        # For simplicity, if the exact type_str is in COMPOSITE_RACE_TYPES or
        # if it's a composite type where the last part is, we use "・"

        if type_str in cls.COMPOSITE_RACE_TYPES:
            return f"{race_str}・{type_str}"

        # If type_str is like "タマシード/クリーチャー" and race is "ドラゴン",
        # the convention is usually "ドラゴン・タマシード/クリーチャー"
        if any(c in type_str for c in cls.COMPOSITE_RACE_TYPES):
             return f"{race_str}・{type_str}"

        # Otherwise, fallback to "の"
        return f"{race_str}の{type_str}"
