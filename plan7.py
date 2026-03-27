import re

# Fix zone_formatters.py
with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "r") as f:
    zf_content = f.read()

zf_content = zf_content.replace(
    "is_all = (amount == 0 and not input_key)",
    "input_key = command.get('input_value_key') or command.get('input_link')\n        is_all = (amount == 0 and not input_key)"
)

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "w") as f:
    f.write(zf_content)

# Fix game_action_formatters.py
with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "r") as f:
    gaf_content = f.read()

gaf_content = gaf_content.replace(
    "if input_key:",
    "input_key = command.get('input_value_key') or command.get('input_link')\n        if input_key:"
)

with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "w") as f:
    f.write(gaf_content)

print("Fixed missing input_key definitions.")
