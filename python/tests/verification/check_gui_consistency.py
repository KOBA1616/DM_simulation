import sys
import os

# Mock dm_ai_module if not present
try:
    import dm_ai_module
except ImportError:
    pass

# Add repo root to path
sys.path.append(os.getcwd())

from dm_toolkit.consts import COMMAND_TYPES
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.gui.localization import TRANSLATIONS, tr

def check_consistency():
    print("Checking GUI Consistency...")
    print(f"COMMAND_TYPES: {COMMAND_TYPES}")

    missing_config = []
    missing_translation = []

    for cmd_type in COMMAND_TYPES:
        if cmd_type == "NONE": continue

        # Check UI Config
        if cmd_type not in COMMAND_UI_CONFIG:
            missing_config.append(cmd_type)

        # Check Translation
        # logic in localization.py puts them in TRANSLATIONS dict.
        # But wait, localization.py populates TRANSLATIONS based on Enums if module exists.
        # If module doesn't exist, it has a hardcoded map.
        # The hardcoded map in localization.py is:
        # _cmd_map = {'TRANSITION':..., 'MUTATE':..., 'FLOW':..., 'QUERY':..., 'DECIDE':..., 'DECLARE_REACTION':..., 'STAT':..., 'GAME_RESULT':...}
        # It DOES NOT include DRAW_CARD, DISCARD, etc.

        # However, we should check if `tr(cmd_type)` returns the key itself (meaning no translation).
        translated = tr(cmd_type)
        if translated == cmd_type:
            # Maybe it is translated via ActionType mapping?
            # DRAW_CARD is in EffectActionType.
            # But command types are strings here.
            # tr() checks TRANSLATIONS.get(key).
            # If key is "DRAW_CARD", and TRANSLATIONS has "DRAW_CARD" (from ActionType), it works.
            pass

        if cmd_type not in TRANSLATIONS and tr(cmd_type) == cmd_type:
             missing_translation.append(cmd_type)

    if missing_config:
        print(f"FAIL: Missing UI Config for: {missing_config}")
    else:
        print("PASS: All Command Types have UI Config.")

    if missing_translation:
        print(f"FAIL: Missing Translations for: {missing_translation}")
    else:
        print("PASS: All Command Types have Translations.")

    return len(missing_config) + len(missing_translation)

if __name__ == "__main__":
    sys.exit(check_consistency())
