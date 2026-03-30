from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class CommandListFormatter:
    """Utility class for formatting recursive lists of commands like IF blocks or SELECT_OPTION choices."""

    @staticmethod
    def get_bullet(indent_level: int) -> str:
        bullets = ["■", "・", "-", "▶", "▷"]
        return bullets[indent_level % len(bullets)]

    @staticmethod
    def format_list(commands: List[Any], ctx: TextGenerationContext, joiner: str = "、", use_tree: bool = False) -> str:
        """
        Formats a list of commands. If use_tree is True or ctx.indent_level > 0, it formats them as a structured tree
        using indentation and bullets based on ctx.indent_level. Otherwise, it uses the specified joiner.
        """
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        # Auto-enable tree rendering if we are deeply nested (e.g. > 1) or have many commands
        # For top level single conditional statements with 1-2 actions, we prefer inline.
        auto_tree = getattr(ctx, 'indent_level', 0) > 1 or (getattr(ctx, 'indent_level', 0) > 0 and len(commands) > 2)
        effective_use_tree = use_tree if use_tree is not None else auto_tree

        # Save old list for nested restorations, and set the new context
        old_list = ctx.current_commands_list
        ctx.current_commands_list = commands

        texts = []
        for cmd in commands:
            if isinstance(cmd, dict):
                cmd_text = CardTextGenerator._format_command(cmd, ctx)
                if cmd_text:
                    if effective_use_tree:
                        indent_str = "  " * ctx.indent_level
                        bullet = CommandListFormatter.get_bullet(ctx.indent_level)
                        # Add tree structure only if not already formatted with current bullet
                        if not cmd_text.strip().startswith(bullet):
                            cmd_text = f"{indent_str}{bullet} {cmd_text}"
                    texts.append(cmd_text)

        # Restore old context
        ctx.current_commands_list = old_list

        if effective_use_tree:
            return "\n".join(texts)
        return joiner.join(texts)

    @staticmethod
    def format_block(options: List[List[Any]], ctx: TextGenerationContext, bullet: str = None) -> str:
        """
        Formats a list of command chains (like in SELECT_OPTION) into a bulleted string list.
        Uses recursive indentation tracking from ctx.indent_level.
        """
        lines = []
        original_indent = getattr(ctx, 'indent_level', 0)

        # We increase indent for the options block
        ctx.indent_level = original_indent + 1

        auto_bullet = bullet if bullet is not None else CommandListFormatter.get_bullet(original_indent) + " "
        indent_str = "  " * original_indent

        for opt_chain in options:
            # We process options as joined sentences rather than tree-formatted items
            # unless there's deeper nesting within the chain itself.
            chain_text = CommandListFormatter.format_list(opt_chain, ctx, joiner=" ")
            if chain_text:
                lines.append(f"{indent_str}{auto_bullet}{chain_text}")

        ctx.indent_level = original_indent
        return "\n".join(lines)
