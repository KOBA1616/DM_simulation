from typing import Dict, Any, Callable, Type
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.i18n import tr

class ModifierFormatterBase:
    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        raise NotImplementedError

class CostModifierFormatter(ModifierFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: Any) -> None:
        if ctx:
            ctx.metadata["costs_reduced"] = True

    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        if not isinstance(modifier, dict):
            if value > 0:
                return f"{cond}{target}のコストを{value}少なくする。"
            elif value < 0:
                return f"{cond}{target}のコストを{abs(value)}多くする。"
            return f"{cond}{target}のコストを修正する。"

        norm_mod = modifier.copy()
        if "value" not in norm_mod:
             norm_mod["value"] = value

        return text_gen_cls._format_unified_cost_modifier(norm_mod, prefix=cond, target_phrase=f"{target}のコストを", ctx=ctx)

class PowerModifierFormatter(ModifierFormatterBase):
    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        sign = "+" if value >= 0 else ""
        if value == 0:
            return f"{cond}{target}のパワーは不変。"
        return f"{cond}{target}のパワーを{sign}{value}する。"

class GrantKeywordFormatter(ModifierFormatterBase):
    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        str_val = modifier.get('mutation_kind') or modifier.get('str_val', '')

        if not str_val:
            return f"{cond}{target}に能力を与える。"

        keyword = CardTextResources.get_keyword_text(str_val)

        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = CardTextResources.get_duration_text_with_comma(duration_key)

        amt = modifier.get('value') if modifier.get('value') not in (None, 0) else modifier.get('amount', 0)
        if not isinstance(amt, int) or amt <= 0:
            amt = None

        subject = f"{cond}{target}"
        return text_gen_cls._format_keyword_grant_text(subject, str_val, keyword, duration_text, amount=amt)

class SetKeywordFormatter(ModifierFormatterBase):
    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        str_val = modifier.get("mutation_kind", "") or modifier.get("str_val", "")
        if str_val:
            keyword = CardTextResources.get_keyword_text(str_val)
            return f"{cond}{target}は「{keyword}」を得る。"
        return f"{cond}{target}は能力を得る。"

class AddRestrictionFormatter(ModifierFormatterBase):
    @classmethod
    def format(cls, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        keyword = modifier.get("mutation_kind", "") or modifier.get("str_val", "")
        return f"{cond}{scope_prefix}{CardTextResources.get_keyword_text(keyword)}を与える。"


class ModifierFormatterRegistry:
    _registry: Dict[str, Type[ModifierFormatterBase]] = {}

    @classmethod
    def update_metadata(cls, mtype: str, command: Dict[str, Any], ctx: Any) -> None:
        formatter = cls.get_formatter(mtype)
        if formatter and hasattr(formatter, "update_metadata"):
            formatter.update_metadata(command, ctx)

    @classmethod
    def register(cls, mtype: str, formatter: Type[ModifierFormatterBase]) -> None:
        cls._registry[mtype] = formatter

    @classmethod
    def get_formatter(cls, mtype: str) -> Type[ModifierFormatterBase]:
        return cls._registry.get(mtype)

    @classmethod
    def format(cls, mtype: str, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], text_gen_cls: Any, ctx: Any = None) -> str:
        formatter = cls.get_formatter(mtype)
        if formatter:
            return formatter.format(cond, target, scope_prefix, value, modifier, text_gen_cls, ctx)
        return f"{cond}{scope_prefix}常在効果: {tr(mtype)}"

# Register default formatters
ModifierFormatterRegistry.register("COST_MODIFIER", CostModifierFormatter)
ModifierFormatterRegistry.register("POWER_MODIFIER", PowerModifierFormatter)
ModifierFormatterRegistry.register("GRANT_KEYWORD", GrantKeywordFormatter)
ModifierFormatterRegistry.register("SET_KEYWORD", SetKeywordFormatter)
ModifierFormatterRegistry.register("ADD_RESTRICTION", AddRestrictionFormatter)