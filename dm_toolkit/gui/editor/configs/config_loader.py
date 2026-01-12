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

        # Try loading from external JSON
        try:
            with open('data/editor/editor_layout.json', 'r', encoding='utf-8') as f:
                cls._config_cache = json.load(f)
                return cls._config_cache
        except FileNotFoundError:
            # Fallback (empty or default could be defined here, but for now return empty to signal failure)
            print("Warning: editor_layout.json not found.")
            return {}

    @classmethod
    def get_command_groups(cls):
        cfg = cls.load()
        return cfg.get('COMMAND_GROUPS', {})

    @classmethod
    def get_command_ui_config(cls):
        cfg = cls.load()
        return cfg.get('COMMAND_UI_CONFIG', {})
