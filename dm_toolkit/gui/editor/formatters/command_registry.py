from typing import Dict, Any, Type, Optional
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class CommandFormatterRegistry:
    """Registry for command formatters."""

    _formatters: Dict[str, Type[CommandFormatterBase]] = {}

    @classmethod
    def register(cls, command_type: str, formatter_cls: Type[CommandFormatterBase]) -> None:
        """Register a formatter class for a specific command type."""
        cls._formatters[command_type] = formatter_cls

    @classmethod
    def get_formatter(cls, command_type: str) -> Optional[Type[CommandFormatterBase]]:
        """Retrieve the formatter class for the given command type."""
        return cls._formatters.get(command_type)

    @classmethod
    def has_formatter(cls, command_type: str) -> bool:
        """Check if a formatter is registered for the command type."""
        return command_type in cls._formatters

    @classmethod
    def format_command(cls, command: Dict[str, Any], ctx: Any = None) -> str:
        """Centralized dispatch method for formatting commands."""
        if not command:
            return ""

        from dm_toolkit.gui.editor.text_resources import CardTextResources
        from dm_toolkit.gui.i18n import tr
        import copy

        cmd_type = command.get("type") or command.get("name") or "NONE"
        command_ro = copy.deepcopy(command)

        if cmd_type == "SHIELD_TRIGGER":
            return "S・トリガー"

        cmd_type = CardTextResources.normalize_command_alias(cmd_type, command_ro)

        if ctx is None:
            # 再発防止: 直接 format_command() を呼んでも update_metadata が None を前提にせず、
            # 既存フォーマッタが安全に動作するよう既定コンテキストを補う。
            ctx = TextGenerationContext(command_ro)

        formatter_cls = cls.get_formatter(cmd_type)
        if formatter_cls:
            if hasattr(formatter_cls, "update_metadata"):
                formatter_cls.update_metadata(command_ro, ctx)
            return formatter_cls.format_with_optional(command_ro, ctx)

        # In case the normalized alias is not found, try original cmd_type
        # Some aliases might have specific logic in fallback formatters.
        cmd_type_original = command.get("type") or command.get("name") or "NONE"
        formatter_cls_orig = cls.get_formatter(cmd_type_original)
        if formatter_cls_orig and cmd_type_original != cmd_type:
             if hasattr(formatter_cls_orig, "update_metadata"):
                 formatter_cls_orig.update_metadata(command_ro, ctx)
             return formatter_cls_orig.format_with_optional(command_ro, ctx)

        return f"({tr(cmd_type)})"

# Decorator for easy registration
def register_formatter(command_type: str):
    def wrapper(cls: Type[CommandFormatterBase]):
        CommandFormatterRegistry.register(command_type, cls)
        return cls
    return wrapper
