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
            """Register translations for an Enum-like class.

            Defensive: skip types that do not expose __members__ or have non-iterable members.
            """
            if not enum_cls:
                return

            # Only proceed if enum_cls exposes the Enum API we expect
            members = getattr(enum_cls, '__members__', None)
            if members is None or not hasattr(members, 'values'):
                # Not an Enum-like object; skip with a warning
                try:
                    name = enum_cls.__name__
                except Exception:
                    name = str(enum_cls)
                print(f"Warning: Skipping non-enum type when registering translations: {name}")
                return

            try:
                for member in members.values():
                    # Map Enum Member -> Translation (if available)
                    member_name = getattr(member, 'name', None)
                    if member_name and member_name in TRANSLATIONS:
                        TRANSLATIONS[member] = TRANSLATIONS[member_name]
            except Exception as e:
                try:
                    ename = enum_cls.__name__
                except Exception:
                    ename = str(enum_cls)
                print(f"Warning: Failed to register translations for {ename}: {e}")
                return

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
