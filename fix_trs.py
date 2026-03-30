import re

with open("dm_toolkit/gui/editor/services/target_resolution_service.py", "r") as f:
    content = f.read()

content = content.replace("@classmethod\n    def format_modifier_target", "    @classmethod\n    def format_modifier_target")

with open("dm_toolkit/gui/editor/services/target_resolution_service.py", "w") as f:
    f.write(content)
