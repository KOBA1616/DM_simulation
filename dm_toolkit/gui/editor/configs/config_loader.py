# -*- coding: utf-8 -*-
import json
import os
from dm_toolkit.gui.i18n import tr

class EditorConfigLoader:
    _config_cache = None

    @classmethod
    def load(cls):
        if cls._config_cache:
            return cls._config_cache

        # Try loading from command_ui.json (Preferred)
        try:
            with open('data/configs/command_ui.json', 'r', encoding='utf-8') as f:
                cls._config_cache = json.load(f)
                # Check structure: command_ui.json is usually flat.
                # If we need COMMAND_GROUPS, we might need to load editor_layout.json separately
                # or assume groups are defined elsewhere/hardcoded if not in JSON.
                # For now, we assume this file contains the UI config map directly.
                return cls._config_cache
        except FileNotFoundError:
            pass

        # Fallback to legacy layout
        try:
            with open('data/editor/editor_layout.json', 'r', encoding='utf-8') as f:
                cls._config_cache = json.load(f)
                return cls._config_cache
        except FileNotFoundError:
            print("Warning: Configuration files not found.")
            return {}

    @classmethod
    def get_command_groups(cls):
        # Groups might still be in editor_layout.json or hardcoded in unified_action_form
        # For safety, try to load the legacy file specifically for groups if not in current cache
        cfg = cls.load()
        if 'COMMAND_GROUPS' in cfg:
            return cfg['COMMAND_GROUPS']

        # Fallback load for groups
        try:
            with open('data/editor/editor_layout.json', 'r', encoding='utf-8') as f:
                legacy = json.load(f)
                return legacy.get('COMMAND_GROUPS', {})
        except:
            return {}

    @classmethod
    def get_command_ui_config(cls):
        cfg = cls.load()
        # Handle Legacy Nested format
        if 'COMMAND_UI_CONFIG' in cfg:
            return cfg['COMMAND_UI_CONFIG']
        # Handle Modern Flat format (command_ui.json)
        # We assume keys starting with uppercase are Commands
        # Exclude COMMAND_GROUPS as it's a configuration key, not a command
        return {k: v for k, v in cfg.items() if k.isupper() and isinstance(v, dict) and k != 'COMMAND_GROUPS'}
