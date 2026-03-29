import json
import os
from typing import Dict, Optional, Tuple, Any, List, Callable
from dm_toolkit.consts import TargetScope
from dm_toolkit.stat_keys import (
    COMPARE_STAT_EDITOR_KEYS as SHARED_COMPARE_STAT_EDITOR_KEYS,
    EDITOR_QUICK_STATS_KEYS as SHARED_EDITOR_QUICK_STATS_KEYS,
)

class CardTextResourcesMeta(type):
    _data = {}
    _ZONE_MOVE_TEMPLATES = {}
    _TRANSITION_ALIASES = {}

    def load_resources(cls) -> None:
        if cls._data: return
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "resources", "text_resources.json"
        )
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cls._data = json.load(f)

            for k, v in cls._data.get("ZONE_MOVE_TEMPLATES", {}).items():
                if "|" in k:
                    parts = k.split("|", 1)
                    cls._ZONE_MOVE_TEMPLATES[(parts[0], parts[1])] = v
                else:
                    cls._ZONE_MOVE_TEMPLATES[k] = v
            for k, v in cls._data.get("TRANSITION_ALIASES", {}).items():
                if "|" in k:
                    parts = k.split("|", 1)
                    cls._TRANSITION_ALIASES[(parts[0], parts[1])] = v
                else:
                    cls._TRANSITION_ALIASES[k] = v
        except Exception as e:
            print(f"Failed to load text_resources.json: {e}")
            cls._data = {}

    @property
    def CONDITION_JAPANESE(cls): cls.load_resources(); return cls._data.get("CONDITION_JAPANESE", {})
    @property
    def CONDITION_TYPE_LABELS(cls): cls.load_resources(); return cls._data.get("CONDITION_TYPE_LABELS", {})
    @property
    def SCOPE_JAPANESE(cls): cls.load_resources(); return cls._data.get("SCOPE_JAPANESE", {})
    @property
    def TRIGGER_JAPANESE(cls): cls.load_resources(); return cls._data.get("TRIGGER_JAPANESE", {})
    @property
    def SPELL_TRIGGER_JAPANESE(cls): cls.load_resources(); return cls._data.get("SPELL_TRIGGER_JAPANESE", {})
    @property
    def MODIFIER_TYPE_JAPANESE(cls): cls.load_resources(); return cls._data.get("MODIFIER_TYPE_JAPANESE", {})
    @property
    def KEYWORD_TRANSLATION(cls): cls.load_resources(); return cls._data.get("KEYWORD_TRANSLATION", {})
    @property
    def DELAYED_EFFECT_TRANSLATION(cls): cls.load_resources(); return cls._data.get("DELAYED_EFFECT_TRANSLATION", {})
    @property
    def DURATION_TRANSLATION(cls): cls.load_resources(); return cls._data.get("DURATION_TRANSLATION", {})
    @property
    def ZONE_JAPANESE(cls): cls.load_resources(); return cls._data.get("ZONE_JAPANESE", {})
    @property
    def CIVILIZATION_JAPANESE(cls): cls.load_resources(); return cls._data.get("CIVILIZATION_JAPANESE", {})
    @property
    def CARD_TYPE_JAPANESE(cls): cls.load_resources(); return cls._data.get("CARD_TYPE_JAPANESE", {})
    @property
    def PHASE_MAP(cls): cls.load_resources(); return {int(k): v for k, v in cls._data.get("PHASE_MAP", {}).items()}
    @property
    def ACTION_MAP(cls): cls.load_resources(); return cls._data.get("ACTION_MAP", {})
    @property
    def SPECIAL_EFFECT_TEMPLATES(cls): cls.load_resources(); return cls._data.get("SPECIAL_EFFECT_TEMPLATES", {})
    @property
    def COMMAND_ALIASES(cls): cls.load_resources(); return cls._data.get("COMMAND_ALIASES", {})
    @property
    def TRIGGER_COMPOSITION_TEMPLATES(cls): cls.load_resources(); return cls._data.get("TRIGGER_COMPOSITION_TEMPLATES", {})
    @property
    def STAT_KEY_MAP(cls): cls.load_resources(); return cls._data.get("STAT_KEY_MAP", {})
    @property
    def STAT_KEY_ALIASES(cls): cls.load_resources(); return cls._data.get("STAT_KEY_ALIASES", {})
    @property
    def INPUT_USAGE_LABELS(cls): cls.load_resources(); return cls._data.get("INPUT_USAGE_LABELS", {})
    @property
    def ZONE_MOVE_TEMPLATES(cls): cls.load_resources(); return cls._ZONE_MOVE_TEMPLATES
    @property
    def TRANSITION_ALIASES(cls): cls.load_resources(); return cls._TRANSITION_ALIASES
    @property
    def CONJUGATION_RULES(cls): cls.load_resources(); return cls._data.get("CONJUGATION_RULES", {})

class CardTextResources(metaclass=CardTextResourcesMeta):
    """
    Centralized Japanese text resource library for card abilities.
    Used by TextGenerator, FormEditors, and UI components.
    Loads data dynamically from JSON file for pure data-driven mappings.
    """
    
    # These contain logic/lambdas or specific tuples, kept in Python
    REACTION_TEXT_MAP: Dict[str, Callable[[Dict[str, Any]], str]] = {
        "NINJA_STRIKE": lambda r: f"ニンジャ・ストライク {r.get('cost', 0)}",
        "STRIKE_BACK": lambda r: "ストライク・バック",
        "COUNTER_ATTACK": lambda r: f"カウンター・アタック {r.get('cost', 0)}",
        "REVOLUTION_0_TRIGGER": lambda r: "革命0トリガー",
        "SHIELD_TRIGGER": lambda r: "シールド・トリガー",
        "RETURN_ATTACK": lambda r: f"リターン・アタック {r.get('cost', 0)}",
        "ON_DEFEND": lambda r: "守りのトリガー",
    }
    
    COMPARE_STAT_EDITOR_KEYS: Tuple[str, ...] = SHARED_COMPARE_STAT_EDITOR_KEYS
    EDITOR_QUICK_STATS_KEYS: Tuple[str, ...] = SHARED_EDITOR_QUICK_STATS_KEYS

    TRIGGER_REPLACEMENT_MAP: List[Tuple[str, str]] = [
        ("した時", "する時"),
        ("された時", "される時"),
        ("出た時", "出る時"),
        ("置かれた時", "置かれる時"),
        ("ブレイクした時", "ブレイクする時"),
        ("ターンの終わりに", "ターンの終わりに"),
        ("ターンの始めに", "ターンの始めに"),
        ("攻撃の終わりに", "攻撃の終わりに"),
    ]

    PRE_TIMING_TOKENS: Tuple[str, ...] = (
        "される時", "する時", "出る時", "置かれる時",
        "離れる時", "唱える時", "召喚する時",
        "破壊される時", "勝つ時", "負ける時",
        "引く時", "捨てる時"
    )

    @classmethod
    def format_zones_list(cls, zones: List[str], joiner: str = "、または") -> str:
        if not zones: return ""
        zone_names = []
        for z in zones:
            if z == "HAND": zone_names.append("手札")
            elif z == "GRAVEYARD": zone_names.append("墓地")
            elif z in ("MANA", "MANA_ZONE"): zone_names.append("マナゾーン")
            elif z == "DECK": zone_names.append("山札")
            elif z in ("SHIELD", "SHIELD_ZONE"): zone_names.append("シールドゾーン")
            elif z == "BATTLE_ZONE": zone_names.append("バトルゾーン")
            else: zone_names.append(cls.get_zone_text(z))
        return joiner.join(zone_names)

    @classmethod
    def get_condition_text(cls, condition_type: str) -> str:
        cls.load_resources()
        return cls._data.get("CONDITION_JAPANESE", {}).get(condition_type, condition_type)
    
    @classmethod
    def get_condition_type_label(cls, condition_type: str) -> str:
        cls.load_resources()
        return cls._data.get("CONDITION_TYPE_LABELS", {}).get(condition_type, condition_type)
    
    @classmethod
    def get_scope_text(cls, scope: str) -> str:
        cls.load_resources()
        return cls._data.get("SCOPE_JAPANESE", {}).get(scope, "")
    
    @classmethod
    def get_trigger_text(cls, trigger: str, is_spell: bool = False) -> str:
        cls.load_resources()
        if is_spell and trigger in cls._data.get("SPELL_TRIGGER_JAPANESE", {}):
            return cls._data["SPELL_TRIGGER_JAPANESE"][trigger]

        triggers = cls._data.get("TRIGGER_JAPANESE", {})
        if trigger in triggers:
            return triggers[trigger]

        return f"[UNKNOWN_TRIGGER: {trigger}]"
    
    @classmethod
    def get_modifier_type_text(cls, modifier_type: str) -> str:
        cls.load_resources()
        return cls._data.get("MODIFIER_TYPE_JAPANESE", {}).get(modifier_type, modifier_type)
    
    @classmethod
    def get_keyword_text(cls, keyword: str) -> str:
        cls.load_resources()
        kw = cls._data.get("KEYWORD_TRANSLATION", {})
        if keyword in kw: return kw[keyword]
        if keyword.lower() in kw: return kw[keyword.lower()]
        if keyword.upper() in kw: return kw[keyword.upper()]
        return keyword

    @classmethod
    def get_delayed_effect_text(cls, effect_id: str) -> str:
        cls.load_resources()
        return cls._data.get("DELAYED_EFFECT_TRANSLATION", {}).get(effect_id, effect_id)
    
    @classmethod
    def get_zone_text(cls, zone: str) -> str:
        cls.load_resources()
        z = cls.normalize_zone_name(zone)
        zones = cls._data.get("ZONE_JAPANESE", {})
        if z in zones:
            return zones[z]
        return f"[UNKNOWN_ZONE: {z}]"
    
    @classmethod
    def get_civilization_text(cls, civ: str) -> str:
        cls.load_resources()
        return cls._data.get("CIVILIZATION_JAPANESE", {}).get(civ, civ)
    
    @classmethod
    def get_card_type_text(cls, card_type: str) -> str:
        cls.load_resources()
        return cls._data.get("CARD_TYPE_JAPANESE", {}).get(card_type, card_type)

    @classmethod
    def normalize_zone_name(cls, zone: str) -> str:
        if not zone: return ""
        z = str(zone).split(".")[-1].upper()
        zone_map = {
            "BATTLE": "BATTLE_ZONE", "MANA": "MANA_ZONE", "SHIELD": "SHIELD_ZONE",
            "BATTLE_ZONE": "BATTLE_ZONE", "MANA_ZONE": "MANA_ZONE", "SHIELD_ZONE": "SHIELD_ZONE",
            "HAND": "HAND", "GRAVEYARD": "GRAVEYARD", "DECK": "DECK",
            "DECK_TOP": "DECK_TOP", "DECK_BOTTOM": "DECK_BOTTOM",
            "BUFFER": "BUFFER", "UNDER_CARD": "UNDER_CARD",
        }
        return zone_map.get(z, z)

    @classmethod
    def normalize_command_alias(cls, command_type: str) -> str:
        cls.load_resources()
        return cls._data.get("COMMAND_ALIASES", {}).get(command_type, command_type)

    @classmethod
    def get_duration_text(cls, duration_key: str, look_count: int = 1) -> str:
        cls.load_resources()
        dt = cls._data.get("DURATION_TRANSLATION", {})
        if duration_key in dt:
            return dt[duration_key]
        if duration_key:
            return duration_key
        # Centralized Fallback logic
        return f"{look_count}ターン" if look_count > 0 else "このターン"

    @classmethod
    def get_duration_text_with_comma(cls, duration_key: str, look_count: int = 1) -> str:
        if not duration_key: return ""
        trans = cls.get_duration_text(duration_key, look_count)
        if trans and trans != duration_key:
            return trans + "、"
        cls.load_resources()
        dt = cls._data.get("DURATION_TRANSLATION", {})
        if duration_key in dt:
            return dt[duration_key] + "、"
        return ""

    @classmethod
    def get_stat_key_label(cls, stat_key: str) -> str:
        cls.load_resources()
        normalized = cls.normalize_stat_key(stat_key)
        sm = cls._data.get("STAT_KEY_MAP", {})
        if normalized in sm:
            name, unit = sm[normalized]
            return f"{name}（{unit}）" if unit else name
        return normalized

    @classmethod
    def normalize_stat_key(cls, stat_key: str) -> str:
        cls.load_resources()
        if not isinstance(stat_key, str): return str(stat_key)
        return cls._data.get("STAT_KEY_ALIASES", {}).get(stat_key, stat_key)
