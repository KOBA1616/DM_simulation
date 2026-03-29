from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.condition_formatter import ConditionFormatter
from dm_toolkit.gui.i18n import tr

class LogicFormatterUtils:
    @classmethod
    def get_cond_text(cls, action: Dict[str, Any], default_text: str = "もし条件を満たすなら") -> str:
        cond_detail = action.get('condition', {}) or action.get('target_filter', {})
        cond_text = ConditionFormatter.format_condition_text(cond_detail, action)

        if not cond_text:
            cond_text = default_text

        return cond_text


@register_formatter("IF")
class IfFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.formatters.command_list_formatter import CommandListFormatter
        cond_text = LogicFormatterUtils.get_cond_text(command, "もし条件を満たすなら")

        if_true_cmds = command.get('if_true', [])

        original_indent = getattr(ctx, 'indent_level', 0)
        ctx.indent_level = original_indent + 1

        # Determine if we should use tree format
        use_tree = original_indent > 0 or len(if_true_cmds) > 1

        actions_text = CommandListFormatter.format_list(if_true_cmds, ctx, joiner="。そうしたら、", use_tree=use_tree)
        ctx.indent_level = original_indent

        if actions_text:
            if "\n" in actions_text:
                return f'{cond_text}、次を行う。\n{actions_text}'
            else:
                return f'{cond_text}、{actions_text}'
        else:
            return f'（{cond_text}）'

@register_formatter("IF_ELSE")
class IfElseFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.formatters.command_list_formatter import CommandListFormatter
        cond_text = LogicFormatterUtils.get_cond_text(command, "もし条件を満たすなら")

        if_true_cmds = command.get('if_true', [])
        if_false_cmds = command.get('if_false', [])

        original_indent = getattr(ctx, 'indent_level', 0)
        ctx.indent_level = original_indent + 1

        # Determine if we should use tree format
        use_tree = original_indent > 0 or max(len(if_true_cmds), len(if_false_cmds)) > 1

        true_actions_text = CommandListFormatter.format_list(if_true_cmds, ctx, joiner="。そうしたら、", use_tree=use_tree)
        false_actions_text = CommandListFormatter.format_list(if_false_cmds, ctx, joiner="。そうしなかったら、", use_tree=use_tree)
        ctx.indent_level = original_indent

        result_parts = []
        if true_actions_text:
            if "\n" in true_actions_text:
                result_parts.append(f'{cond_text}、次を行う。\n' + true_actions_text)
            else:
                result_parts.append(f'{cond_text}、{true_actions_text}')

        if false_actions_text:
            if "\n" in false_actions_text:
                result_parts.append(f'そうしなかったら、次を行う。\n' + false_actions_text)
            else:
                prefix = "" if false_actions_text.startswith("そうしなかったら") else "そうしなかったら、"
                result_parts.append(f'{prefix}{false_actions_text}')

        if result_parts:
            # If using tree format, join with newlines instead of circles
            if "\n" in true_actions_text or "\n" in false_actions_text:
                return '\n'.join(result_parts)
            return '。'.join(result_parts) + '。'
        else:
            return f'（条件分岐: {cond_text}）'

@register_formatter("ELSE")
class ElseFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        return '（そうでなければ）'
