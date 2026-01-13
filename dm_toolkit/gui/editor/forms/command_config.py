# -*- coding: utf-8 -*-
# command_config.py (ACTION_UI_CONFIG)
# \x83A\x83N\x83V\x83\x87\x83\x93\x90ݒ\xe8\x82̍\\x90\xac\x83t\x83@\x83C\x83\x8b\x81Bunified_action_form.py \x82\xa8\x82\xe6\x82\xd1 schema_def.py \x82\xaa\x8eg\x97p\x82\xb7\x82\xe9\x83f\x81[\x83^\x82\xf0\x92\xe8\x8b`\x82\xb5\x82܂\xb7\x81B
# \x88ێ\x9d: AI\x82\xaa\x95ҏW\x82\xb7\x82\xe9\x8dۂ\xcd unified_action_form.py \x82\xe2 schema_def.py \x82ƃZ\x83b\x83g\x82ŗ\x9d\x89\xf0\x82\xb7\x82\xe9\x95K\x97v\x82\xaa\x82\xa0\x82\xe8\x82܂\xb7\x81B
# \x8fd\x97v: \x82\xb1\x82̃t\x83@\x83C\x83\x8b\x82͍폜\x82\xb5\x82Ȃ\xa2\x82ł\xad\x82\xbe\x82\xb3\x82\xa2\x81B\x93\xae\x93I\x83t\x83H\x81[\x83\x80\x90\xb6\x90\xac\x82̍\\x90\xac\x8a\xee\x94Ղł\xb7\x81B

import json
import os

class CommandDef:
    """
    Configuration class for a Command Type.
    Separates the definition logic from the generated configuration dictionary.
    """
    def __init__(self, key, config_dict=None):
        self.key = key
        self.config = config_dict or {}

    def build_config(self):
        """Generates the dictionary entry for COMMAND_UI_CONFIG."""
        # The JSON structure is already close to what we need, but we map labels manually for now
        # to match original build_config behavior if needed.
        # Actually, if we just load JSON directly into COMMAND_UI_CONFIG, we don't need this class much.

        # Flatten labels for the UnifiedActionForm consumption
        conf = self.config.copy()
        labels = conf.get('labels', {})
        for k, v in labels.items():
            if k == "amount": conf["label_amount"] = v
            elif k == "mutation_kind": conf["label_mutation_kind"] = v
            elif k == "str_param": conf["label_str_param"] = v

        # Handle Output Key Label
        outputs = conf.get('outputs', {})
        if outputs:
            conf["outputs"] = outputs

        return conf

def load_command_config():
    config_path = None
    # 1. Environment Variable
    env_path = os.environ.get('DM_COMMAND_UI_CONFIG_PATH')
    if env_path and os.path.exists(env_path):
        config_path = env_path
    else:
        # 2. Relative Path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(current_dir, '..', '..', '..', 'data', 'configs', 'command_ui.json'),
            # repository root fallback (forms -> editor -> gui -> dm_toolkit -> repo root)
            os.path.join(current_dir, '..', '..', '..', '..', 'data', 'configs', 'command_ui.json'),
            os.path.join(os.getcwd(), 'data', 'configs', 'command_ui.json')
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break

    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading command_ui.json: {e}")
    return {}

# Load JSON
_raw_config = load_command_config()

# Generate Dictionary Export
COMMAND_UI_CONFIG = {}

# If JSON load failed or is empty (fallback), we might want to keep the old hardcoded definitions
# temporarily or just accept empty config. Ideally, we should have defaults.
# For now, we assume JSON exists as we just created it.

for key, val in _raw_config.items():
    cmd_def = CommandDef(key, val)
    COMMAND_UI_CONFIG[key] = cmd_def.build_config()

# Fallback for NONE if missing
if "NONE" not in COMMAND_UI_CONFIG:
    COMMAND_UI_CONFIG["NONE"] = {"visible": []}
