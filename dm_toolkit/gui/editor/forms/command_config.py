# -*- coding: utf-8 -*-
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
