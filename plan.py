import re

with open("dm_toolkit/gui/editor/formatters/utils.py", "r") as f:
    content = f.read()

new_func = """
def get_command_max_cost(command: Dict[str, Any], default: Any = None) -> Any:
    \"\"\"
    Safely extract the max_cost from a command dictionary or its target_filter.
    \"\"\"
    max_cost_src = command.get('max_cost')
    if max_cost_src is None and 'target_filter' in command:
        max_cost_src = (command.get('target_filter') or {}).get('max_cost')

    if max_cost_src is not None and not isinstance(max_cost_src, dict):
        return max_cost_src

    return default
"""

if "get_command_max_cost" not in content:
    content += new_func
    with open("dm_toolkit/gui/editor/formatters/utils.py", "w") as f:
        f.write(content)

with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "r") as f:
    se_content = f.read()

se_content = se_content.replace(
    "from dm_toolkit.gui.editor.formatters.utils import get_command_amount",
    "from dm_toolkit.gui.editor.formatters.utils import get_command_amount, get_command_max_cost"
)

old_block = """        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)"""

new_block = """        max_cost_src = get_command_max_cost(command)
        val1 = max_cost_src if max_cost_src is not None else get_command_amount(command, default=0)"""

se_content = se_content.replace(old_block, new_block)

with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "w") as f:
    f.write(se_content)

print("utils.py and special_effect_formatters.py updated.")
