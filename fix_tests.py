import os
import glob
import re

TEST_DIR = "python/tests/unit"

for filepath in glob.glob(f"{TEST_DIR}/test_text_generator*.py"):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # If the file contains `_format_game_action_command`, we replace it with a helper that creates a context and calls _format_command
    if "_format_game_action_command" in content or "_format_buffer_command" in content or "_format_mutate_action" in content:
        # We inject a helper
        helper = """
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
def _call_format_command(cmd_type, action, is_spell, look_count=0, add_count=0, target_str="", unit="", input_key="", input_usage="", sample=None):
    from dm_toolkit.gui.editor.text_generator import CardTextGenerator
    action = action.copy() if action else {}
    action["type"] = cmd_type
    if look_count: action["look_count"] = look_count
    if add_count: action["add_count"] = add_count
    if input_key: action["input_value_key"] = input_key
    if input_usage: action["input_usage"] = input_usage
    # For testing, we mock the resolution
    ctx = TextGenerationContext({}, sample)
    ctx.is_spell = is_spell
    return CardTextGenerator._format_command(action, ctx)

def _call_format_buffer(cmd_type, action, is_spell, look_count=0):
    return _call_format_command(cmd_type, action, is_spell, look_count)

def _call_format_mutate(cmd_type, action, is_spell, look_count=0, add_count=0, target_str="", unit=""):
    return _call_format_command(cmd_type, action, is_spell, look_count, add_count, target_str, unit)
"""
        # Inject imports
        if "from dm_toolkit.gui.editor.formatters.context import TextGenerationContext" not in content:
            content = content.replace("from dm_toolkit.gui.editor.text_generator import CardTextGenerator", "from dm_toolkit.gui.editor.text_generator import CardTextGenerator\n" + helper)

        content = re.sub(r"CardTextGenerator\._format_game_action_command\(", "_call_format_command(", content)
        content = re.sub(r"CardTextGenerator\._format_buffer_command\(", "_call_format_buffer(", content)
        content = re.sub(r"CardTextGenerator\._format_mutate_action\(", "_call_format_mutate(", content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print("Tests updated.")
