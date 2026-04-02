from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordRegistry

@register_formatter("REVOLUTION_CHANGE")
class RevolutionChangeCommandFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        formatter_cls = SpecialKeywordRegistry.get_formatter("revolution_change")
        if formatter_cls and hasattr(formatter_cls, "format_revolution_change_text"):
            tf = command.get("target_filter") or command.get("filter") or {}
            cond_text = formatter_cls.format_revolution_change_text(tf) if tf else "クリーチャー"
            return f"革命チェンジ：{cond_text}"
        return "革命チェンジ"
