import re

# Update command_formatter_base.py
with open("dm_toolkit/gui/editor/formatters/command_formatter_base.py", "r") as f:
    base_content = f.read()

new_base = """from typing import Dict, Any, List, Optional
import dm_toolkit.consts as consts
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

class CommandFormatterBase:
    \"\"\"Base class for all command text formatters.\"\"\"

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        \"\"\"
        Generate Japanese text for the given command.
        Must be implemented by subclasses.
        \"\"\"
        raise NotImplementedError("Subclasses must implement format()")

    @classmethod
    def format_with_optional(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        \"\"\"
        Formats the command and automatically applies optional conjugation if needed.
        \"\"\"
        text = cls.format(command, ctx)
        optional = bool(command.get("optional", False))
        return TextUtils.apply_conjugation(text, optional)

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False, **kwargs) -> tuple[str, str]:
        \"\"\"
        Helper to delegate target resolution to TargetFormatter.
        \"\"\"
        from dm_toolkit.gui.editor.formatters.target_formatter import TargetFormatter
        return TargetFormatter.format_target(action, is_spell, **kwargs)
"""

with open("dm_toolkit/gui/editor/formatters/command_formatter_base.py", "w") as f:
    f.write(new_base)

# Replace all calls from CommandFormatterRegistry.get_formatter(...).format(...) to format_with_optional
# This requires replacing the caller inside CardTextGenerator
with open("dm_toolkit/gui/editor/text_generator.py", "r") as f:
    tg_content = f.read()

tg_content = tg_content.replace(
    "formatted_text = formatter_cls.format(command_copy, ctx)",
    "formatted_text = formatter_cls.format_with_optional(command_copy, ctx)"
)
with open("dm_toolkit/gui/editor/text_generator.py", "w") as f:
    f.write(tg_content)

print("command_formatter_base.py and text_generator.py updated.")
