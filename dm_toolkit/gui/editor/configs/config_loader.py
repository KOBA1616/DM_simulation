# -*- coding: utf-8 -*-
import json
import os
from dm_toolkit.gui.i18n import tr

class EditorConfigLoader:
    _layout_cache = None
    _ui_config_cache = None

    @classmethod
    def _load_layout(cls):
        if cls._layout_cache:
            return cls._layout_cache
        try:
            with open('data/editor/editor_layout.json', 'r', encoding='utf-8') as f:
                cls._layout_cache = json.load(f)
        except FileNotFoundError:
            print("Warning: editor_layout.json not found.")
            cls._layout_cache = {}
        return cls._layout_cache

    @classmethod
    def _load_ui_config(cls):
        if cls._ui_config_cache:
            return cls._ui_config_cache

        # 1. Try modern flat config
        try:
            with open('data/configs/command_ui.json', 'r', encoding='utf-8') as f:
                cls._ui_config_cache = json.load(f)
                return cls._ui_config_cache
        except FileNotFoundError:
            pass

        # 2. Fallback to legacy nested config
        layout = cls._load_layout()
        cls._ui_config_cache = layout.get('COMMAND_UI_CONFIG', {})
        return cls._ui_config_cache

    @classmethod
    def get_command_groups(cls):
        return cls._load_layout().get('COMMAND_GROUPS', {})

    @classmethod
    def get_command_ui_config(cls):
        return cls._load_ui_config()
