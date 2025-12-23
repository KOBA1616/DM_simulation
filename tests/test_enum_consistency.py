
import pytest
import sys
import os

# Ensure dm_toolkit is in path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import dm_ai_module
    from dm_toolkit.gui.localization import TRANSLATIONS
    from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG
except ImportError:
    pytest.skip("dm_ai_module or dm_toolkit not available, skipping consistency tests", allow_module_level=True)

def test_effect_action_type_consistency():
    """
    Verify that all EffectActionType values defined in C++ are present in:
    1. localization.py (TRANSLATIONS)
    2. action_config.py (ACTION_UI_CONFIG)
    """
    missing_translations = []
    missing_configs = []

    # List of Enum members to potentially ignore if they are purely internal
    # For now, we assume all should be localized/configured or explicitly excluded here.
    ignored_configs = {
        dm_ai_module.EffectActionType.CAST_SPELL, # Often internal or handled specially
        dm_ai_module.EffectActionType.PUT_CREATURE, # Internal
        dm_ai_module.EffectActionType.RESOLVE_BATTLE, # Internal mechanics
    }

    for name, member in dm_ai_module.EffectActionType.__members__.items():
        if member not in TRANSLATIONS:
            # Check if string key exists (legacy)
            if name not in TRANSLATIONS:
                missing_translations.append(name)

        if member not in ACTION_UI_CONFIG:
            if member not in ignored_configs:
                # Check if string key exists
                if name not in ACTION_UI_CONFIG:
                    missing_configs.append(name)

    assert not missing_translations, f"Missing translations for EffectActionType: {missing_translations}"
    assert not missing_configs, f"Missing UI configurations for EffectActionType: {missing_configs}"

def test_action_type_consistency():
    """Verify ActionType values in localization."""
    missing_translations = []

    ignored_translations = {
        dm_ai_module.ActionType.PLAY_CARD_INTERNAL, # Internal
        dm_ai_module.ActionType.RESOLVE_BATTLE,
        dm_ai_module.ActionType.BREAK_SHIELD,
    }

    for name, member in dm_ai_module.ActionType.__members__.items():
        if member not in TRANSLATIONS:
            if member not in ignored_translations and name not in TRANSLATIONS:
                 missing_translations.append(name)

    assert not missing_translations, f"Missing translations for ActionType: {missing_translations}"

def test_trigger_type_consistency():
    """Verify TriggerType values in localization."""
    missing_translations = []

    ignored_translations = {
        dm_ai_module.TriggerType.NONE
    }

    for name, member in dm_ai_module.TriggerType.__members__.items():
        if member not in TRANSLATIONS:
             if member not in ignored_translations and name not in TRANSLATIONS:
                missing_translations.append(name)

    assert not missing_translations, f"Missing translations for TriggerType: {missing_translations}"

def test_card_type_consistency():
    """Verify CardType values in localization."""
    missing_translations = []

    # Check if localization has keys for all CardTypes
    # Note: localization keys might be string names or Enum members

    for name, member in dm_ai_module.CardType.__members__.items():
        if member not in TRANSLATIONS and name not in TRANSLATIONS:
             missing_translations.append(name)

    assert not missing_translations, f"Missing translations for CardType: {missing_translations}"
