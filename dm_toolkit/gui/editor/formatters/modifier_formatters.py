from typing import Dict, Any, Callable, Type
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.metadata_flags import SemanticMetadataFlags
from dm_toolkit.gui.i18n import tr

class ModifierFormatterBase:
    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        raise NotImplementedError

class CostModifierFormatter(ModifierFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: Any) -> None:
        if ctx:
            ctx.metadata[SemanticMetadataFlags.COSTS_REDUCED.value] = True

    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        if str(value).upper() in ("INFINITY", "∞"):
            return f"{cond}{target}のコストを∞にする。"
        if str(value) == "*":
            return f"{cond}{target}のコストを＊にする。"

        if not isinstance(modifier, dict):
            if isinstance(value, (int, float)) and value > 0:
                return f"{cond}{target}のコストを{value}少なくする。"
            elif isinstance(value, (int, float)) and value < 0:
                return f"{cond}{target}のコストを{abs(value)}多くする。"
            return f"{cond}{target}のコストを修正する。"

        norm_mod = modifier.copy()
        if "value" not in norm_mod:
             norm_mod["value"] = value

        from dm_toolkit.gui.editor.formatters.cost_modifier_formatter import CostModifierFormatter as CMF
        return CMF._format_unified_cost_modifier(norm_mod, prefix=cond, target_phrase=f"{target}のコストを", ctx=ctx)

class PowerModifierFormatter(ModifierFormatterBase):
    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        if str(value).upper() in ("INFINITY", "∞"):
            return f"{cond}{target}のパワーを∞にする。"
        if str(value) == "*":
            return f"{cond}{target}のパワーを＊にする。"

        if isinstance(value, (int, float)) and value > 0:
            return f"{cond}{target}のパワーを+{value}する。"
        elif isinstance(value, (int, float)) and value < 0:
            return f"{cond}{target}のパワーを{abs(value)}低下させる。"
        else:
            return f"{cond}{target}のパワーは不変。"

from enum import Enum

class CharacteristicModifierType(Enum):
    GRANT = "GRANT"
    REMOVE = "REMOVE"
    RESTRICT = "RESTRICT"
    SET = "SET"

class CharacteristicModifierBase(ModifierFormatterBase):
    @classmethod
    def format_characteristic(cls, behavior: CharacteristicModifierType, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        str_val = modifier.get('mutation_kind') or modifier.get('str_val', '')
        keyword = CardTextResources.get_keyword_text(str_val) if str_val else ""

        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = CardTextResources.get_duration_text_with_comma(duration_key)

        amt = modifier.get('value') if modifier.get('value') not in (None, 0) else modifier.get('amount', 0)
        if not isinstance(amt, int) or amt <= 0:
            amt = None

        subject = f"{target}"

        # Determine duration placement
        if duration_text and not duration_text.endswith("、"):
            duration_text += "、"

        prefix = f"{cond}{duration_text}"

        # Determine the action based on behavior
        if behavior == CharacteristicModifierType.GRANT:
            if not str_val:
                return f"{prefix}{target}に能力を与える。"
            # Pass an empty duration_text since we handle it in prefix
            return cls._format_keyword_grant_text(subject, str_val, keyword, "", amount=amt, cond_prefix=prefix)

        elif behavior == CharacteristicModifierType.SET:
            if keyword:
                return f"{prefix}{target}は「{keyword}」を得る。"
            return f"{prefix}{target}は能力を得る。"

        elif behavior == CharacteristicModifierType.REMOVE:
            if keyword:
                return f"{prefix}{target}は「{keyword}」を失う。"
            return f"{prefix}{target}から能力を失わせる。"

        elif behavior == CharacteristicModifierType.RESTRICT:
            if keyword:
                return f"{prefix}{target}に{keyword}を与える。"
            return f"{prefix}{target}に制限を与える。"

        return f"{prefix}{target}の能力を変更する。"

    @classmethod
    def _format_keyword_grant_text(cls, target_str: str, key_id: str, display_text: str, duration_text: str, amount: int = None, skip_selection: bool = False, cond_prefix: str = "") -> str:
        """Helper to format keyword granting text.

        amount=None or 0: apply to all matching targets (no selection).
        amount>0: select N targets (N体選び).
        skip_selection=True: target already determined by input link.
        cond_prefix: A prefix like 'このターン、' that goes before the sentence.
        """
        restriction_keys = [
            'CANNOT_ATTACK', 'CANNOT_BLOCK', 'CANNOT_ATTACK_OR_BLOCK', 'CANNOT_ATTACK_AND_BLOCK'
        ]
        is_restriction = (key_id in restriction_keys) or (str(key_id).upper() in restriction_keys)

        # Normalize duration_text end
        if duration_text and not duration_text.endswith('、'):
            duration_text += "、"

        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        selection_prefix = TargetResolutionService.format_target_selection_prefix(target_str, amount, skip_selection)

        # Adjust placement of prefix - prefix comes first
        if selection_prefix:
            prefix = f"{cond_prefix}{selection_prefix}"
        else:
            prefix = cond_prefix

        # If we selected targets, the subject usually becomes "そのクリーチャー"
        # However if it's all targets (amount=0/None), subject remains target_str
        has_selection = bool(selection_prefix)
        subject_str = "そのクリーチャー" if has_selection or skip_selection else target_str

        if is_restriction:
            template = "{modifier}{duration}{target}は{keyword}。"
            return template.format(
                modifier=prefix,
                duration=duration_text,
                target=subject_str,
                keyword=display_text
            )

        template = "{modifier}{duration}{target}に「{keyword}」を与える。"
        return template.format(
            modifier=prefix,
            duration=duration_text,
            target=subject_str,
            keyword=display_text
        )

class GrantKeywordFormatter(CharacteristicModifierBase):
    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        return cls.format_characteristic(CharacteristicModifierType.GRANT, cond, target, value, modifier, ctx)

class SetKeywordFormatter(CharacteristicModifierBase):
    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        return cls.format_characteristic(CharacteristicModifierType.SET, cond, target, value, modifier, ctx)

class AddRestrictionFormatter(CharacteristicModifierBase):
    @classmethod
    def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        return cls.format_characteristic(CharacteristicModifierType.RESTRICT, cond, target, value, modifier, ctx)


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
    def format(cls, mtype: str, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:
        formatter = cls.get_formatter(mtype)
        if formatter:
            return formatter.format(cond, target, value, modifier, ctx)
        return f"{cond}{target}常在効果: {tr(mtype)}"

# Register default formatters
ModifierFormatterRegistry.register("COST_MODIFIER", CostModifierFormatter)
ModifierFormatterRegistry.register("POWER_MODIFIER", PowerModifierFormatter)
ModifierFormatterRegistry.register("GRANT_KEYWORD", GrantKeywordFormatter)
ModifierFormatterRegistry.register("SET_KEYWORD", SetKeywordFormatter)
ModifierFormatterRegistry.register("ADD_RESTRICTION", AddRestrictionFormatter)