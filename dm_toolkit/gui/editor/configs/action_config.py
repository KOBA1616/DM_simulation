# -*- coding: utf-8 -*-
# Action/Command UI Configuration
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader

# Redirect to loader
COMMAND_GROUPS = EditorConfigLoader.get_command_groups()
COMMAND_UI_CONFIG = EditorConfigLoader.get_command_ui_config()
