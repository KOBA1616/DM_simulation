from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class CommandListFormatter:
    """Utility class for formatting recursive lists of commands like IF blocks or SELECT_OPTION choices."""

    @staticmethod
    def format_list(commands: List[Any], ctx: TextGenerationContext, joiner: str = "、") -> str:
        """
        Formats a list of commands and joins them with the specified joiner.
        """
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        texts = []
        for cmd in commands:
            if isinstance(cmd, dict):
                cmd_text = CardTextGenerator._format_command(cmd, ctx)
                if cmd_text:
                    texts.append(cmd_text)
        return joiner.join(texts)

    @staticmethod
    def format_block(options: List[List[Any]], ctx: TextGenerationContext, bullet: str = "> ") -> str:
        """
        Formats a list of command chains (like in SELECT_OPTION) into a bulleted string list.
        """
        lines = []
        for opt_chain in options:
            chain_text = CommandListFormatter.format_list(opt_chain, ctx, joiner=" ")
            if chain_text:
                lines.append(f"{bullet}{chain_text}")
        return "\n".join(lines)
