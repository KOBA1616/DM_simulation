# -*- coding: utf-8 -*-
from typing import List, Dict, Any

from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class ClauseJoiner:
    """
    Handles structurally combining condition clauses as an Abstract Syntax Tree (AST)
    with natural language particles ("〜時、", "〜なら、", "〜中、").
    Supports nesting (AND/OR) and parallel conditions cleanly.
    """

    @classmethod
    def join_condition_ast(cls, condition_ast: Dict[str, Any], ctx: TextGenerationContext, is_root: bool = True) -> str:
        """
        Evaluates a condition AST recursively, formats the leaf nodes,
        and joins them with appropriate logical operators and suffixes.
        The grammatical suffix is only applied at the root level.
        """
        if not condition_ast:
            return ""

        cond_type = condition_ast.get("type", "NONE")

        # Handle composite logical nodes (AND/OR nesting)
        if cond_type == "OR" or cond_type == "AND":
            sub_conds = condition_ast.get("conditions", [])
            # Recursively evaluate sub-conditions without applying suffixes yet
            formatted_subs = [cls.join_condition_ast(sc, ctx, is_root=False) for sc in sub_conds if sc]
            formatted_subs = [s for s in formatted_subs if s] # filter empties

            if not formatted_subs:
                return ""

            # Join with appropriate conjunction
            joiner = "、または" if cond_type == "OR" else "、かつ"
            joined_text = joiner.join(formatted_subs)

            if is_root:
                # Apply suffix logic to the joined composite clause using the last sub-condition's type
                last_type = sub_conds[-1].get("type", "NONE") if sub_conds else "NONE"
                return cls._apply_suffix(joined_text, last_type)
            return joined_text

        # Handle a single leaf condition
        leaf_text = cls._format_leaf(condition_ast, ctx)
        if not leaf_text:
            return ""

        if is_root:
            return cls._apply_suffix(leaf_text, cond_type)
        return leaf_text

    @classmethod
    def _format_leaf(cls, condition: Dict[str, Any], ctx: TextGenerationContext) -> str:
        """Formats a single condition node using the registry."""
        from dm_toolkit.gui.editor.formatters.condition_registry import ConditionFormatterRegistry
        cond_type = condition.get("type", "NONE")
        formatter_cls = ConditionFormatterRegistry.get_formatter(cond_type)
        if formatter_cls:
            return formatter_cls.format(condition, ctx)
        return ""

    @classmethod
    def _apply_suffix(cls, text: str, condition_type: str) -> str:
        """Appends the appropriate grammatical suffix based on the condition type."""
        text = text.rstrip("、:。")

        from dm_toolkit.gui.editor.formatters.condition_registry import ConditionFormatterRegistry
        formatter_cls = ConditionFormatterRegistry.get_formatter(condition_type)

        suffix = ""
        if formatter_cls:
             # Ensure safety against classes that might have overridden and removed get_suffix
             suffix_func = getattr(formatter_cls, "get_suffix", None)
             if suffix_func:
                 suffix = suffix_func()

        if not suffix:
             if text.endswith("ない") or text.endswith("る") or text.endswith("た") or text.endswith("い"):
                  suffix = "なら、"
             else:
                  suffix = "なら、"

        # Ensure we don't duplicate logic if text already handles it in some edge case
        if text.endswith("なら"):
             suffix = "、"

        if text.endswith("中") and suffix == "中、":
             suffix = "、"

        if text.endswith("時") and suffix == "時、":
             suffix = "、"

        if text.endswith("ば") and suffix == "なら、":
             suffix = "、"

        return f"{text}{suffix}"
