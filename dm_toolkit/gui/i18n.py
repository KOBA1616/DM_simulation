# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, Optional

# dm_ai_module
m: Optional[Any] = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    pass

TRANSLATIONS: Dict[Any, str] = {}

def load_translations():
    global TRANSLATIONS
    # Load JSON
    # Resolve absolute path relative to this file
    # This file is in dm_toolkit/gui/i18n.py
    # Root is ../../
    json_path = os.path.join(os.path.dirname(__file__), "../../data/locale/ja.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            TRANSLATIONS.update(json.load(f))
    except FileNotFoundError:
        print(f"Warning: Translation file not found at {json_path}")

    # Add Enums if module is available
    if m:
        # Helper to set enum translation using JSON only
        def register_enum_translations(enum_cls):
            if not enum_cls: return
            for member in enum_cls.__members__.values():
                # Map Enum Member -> Translation (if available)
                if member.name in TRANSLATIONS:
                    TRANSLATIONS[member] = TRANSLATIONS[member.name]

                # Also ensure the String Key -> Translation is preserved (it should be from JSON loading)
                # This loop essentially binds the Enum object itself to the string translation
                # so translate(Enum.MEMBER) works.

        # List of Enums to register
        enum_list = [
            'CardType',
            'EffectActionType',
            'ActionType',
            'TriggerType',
            'Civilization',
            'Zone',
            'TargetScope',
            'CommandType',
            'FlowType',
            'MutationType',
            'StatType',
            'GameResult'
        ]

        for enum_name in enum_list:
            if hasattr(m, enum_name):
                register_enum_translations(getattr(m, enum_name))

# Initialize
load_translations()

def translate(key: Any) -> str:
    """Return localized text when available, otherwise echo the key."""
    # Try direct lookup (works for Enums and strings)
    res = TRANSLATIONS.get(key)
    if res is not None:
        return res

    # If key is an Enum, try looking up its name (fallback)
    if hasattr(key, "name"):
         res = TRANSLATIONS.get(key.name)
         if res is not None:
             return res

    # If key is a string and not found, return as is
    return str(key)

def tr(text: Any) -> str:
    return translate(text)
