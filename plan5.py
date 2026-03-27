import re

with open("dm_toolkit/gui/editor/formatters/legacy_action_formatters.py", "r") as f:
    laf_content = f.read()

laf_content = laf_content.replace(
    "from dm_toolkit.gui.editor.formatters.text_utils import TextUtils\n",
    ""
)

# Replace the def apply_conjugation function completely just to clean it up since it's unused now
laf_content = re.sub(r"    @staticmethod\n    def apply_conjugation\(command: Dict\[str, Any\], text: str\) -> str:\n        optional = command.get\(\"optional\", False\)\n        return TextUtils\.apply_conjugation\(text, optional\)", "", laf_content)

with open("dm_toolkit/gui/editor/formatters/legacy_action_formatters.py", "w") as f:
    f.write(laf_content)
