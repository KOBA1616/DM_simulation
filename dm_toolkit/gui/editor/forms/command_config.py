# -*- coding: utf-8 -*-
# command_config.py
# コマンド設定の構成ファイル（旧名: ACTION_UI_CONFIG）。unified_action_form.py および schema_def.py が使用するデータを定義します。
# 再発防止: ファイル内「アクション(Action)」表記はすべて「コマンド(Command)」に移行済み。
# 備考: AIが編集する際は unified_action_form.py や schema_def.py とセットで同期する必要があります。
# 重要: このファイルは削除しないでください。動的フォーム生成の構成基盤です。

from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader

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
    # 再発防止: 設定ロード経路を EditorConfigLoader に一本化して、
    # forms 側での重複したパス探索実装を再導入しない。
    return EditorConfigLoader.get_command_ui_config()

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
