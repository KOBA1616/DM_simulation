from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.input_link_ast import InputLinkASTBuilder
import copy

class StaticAbilityFormatter:
    """Formatter specifically for PASSIVE_CONST or un-triggered static abilities."""

    @classmethod
    def format(cls, effect: Dict[str, Any], ctx: TextGenerationContext, cond_text: str = "") -> str:
        # Import inside the function to avoid circular imports
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        action_texts = []
        raw_items = []
        commands = effect.get("commands", [])
        commands_with_labels = InputLinkASTBuilder.infer_command_labels(commands)

        for i, command in enumerate(commands_with_labels):
            if ctx and hasattr(ctx, "error_reporter"):
                with ctx.error_reporter.path_segment(f"commands[{i}]"):
                    command_for_text = copy.deepcopy(command) if isinstance(command, dict) else command
                    raw_items.append(command_for_text)
                    action_texts.append(CardTextGenerator._format_command(command_for_text, ctx))
            else:
                command_for_text = copy.deepcopy(command) if isinstance(command, dict) else command
                raw_items.append(command_for_text)
                action_texts.append(CardTextGenerator._format_command(command_for_text, ctx))

        full_action_text = CardTextGenerator._merge_action_texts(raw_items, action_texts)
        return f"{cond_text}{full_action_text}"
