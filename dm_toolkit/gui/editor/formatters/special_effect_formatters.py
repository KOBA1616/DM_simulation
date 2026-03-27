from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
import dm_toolkit.consts as consts
from dm_toolkit.gui.i18n import tr

@register_formatter("MEKRAID")
class MekraidFormatter(CommandFormatterBase):


    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)
        look_count = command.get('look_count') if command.get('look_count') is not None else 0
        val2 = look_count if look_count > 0 else 3
        select_count = command.get('select_count', 1)
        input_key = command.get('input_value_key', '')
        input_usage = command.get('input_value_usage') or command.get('input_usage')

        use_token = str(val1)
        if input_key and input_usage == 'MAX_COST':
            use_token = 'その数'
        elif val1 == 0 and input_usage == 'MAX_COST':
            use_token = 'その数'

        count_str = '1体' if select_count == 1 else f'{select_count}体まで'
        template = CardTextResources.SPECIAL_EFFECT_TEMPLATES.get("MEKRAID", "")
        return template.format(use_token=use_token, val2=val2, count_str=count_str)

@register_formatter("FRIEND_BURST")
class FriendBurstFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        str_val = command.get('str_val', '')
        if not str_val:
            races = command.get('filter', {}).get('races', [])
            if races:
                str_val = races[0]
        template = CardTextResources.SPECIAL_EFFECT_TEMPLATES.get("FRIEND_BURST", "")
        return template.format(str_val=str_val)

@register_formatter("APPLY_MODIFIER")
class ApplyModifierFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        str_param = command.get('mutation_kind') or command.get('str_param') or command.get('str_val') or ''
        duration_key = command.get('duration') or command.get('input_value_key', '')
        input_key = command.get('input_value_key', '')
        input_usage = command.get('input_value_usage') or command.get('input_usage')
        is_target_linked = bool(input_key) and (not input_usage or input_usage == 'TARGET')

        duration_text = ''
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + '、'
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + '、'

        effect_text = CardTextResources.get_keyword_text(str_param) if str_param else '（効果）'
        if isinstance(effect_text, str):
            effect_text = effect_text.strip() or '（効果）'

        amt = command.get('amount')

        if str_param == 'COST':
            amt_val = amt if isinstance(amt, int) else 0
            if is_target_linked:
                select_phrase = ''
            elif isinstance(amt, int) and amt > 0:
                select_phrase = f'{target_str}を{amt}{unit}は、'
            else:
                select_phrase = f'{target_str}を選び、'
            return f'{select_phrase}{duration_text}そのクリーチャーにコスト修正（{amt_val}）を与える。'

        return CardTextGenerator._format_keyword_grant_text(target_str, str_param, effect_text, duration_text, amt, skip_selection=is_target_linked)

@register_formatter("ADD_KEYWORD")
class AddKeywordFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)

        str_val = command.get('str_val', '')
        duration_key = command.get('duration') or command.get('input_value_key', '')
        input_key = command.get('input_value_key', '')
        input_usage = command.get('input_value_usage') or command.get('input_usage')
        is_target_linked = bool(input_key) and (not input_usage or input_usage == 'TARGET')

        duration_text = ''
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + '、'
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + '、'

        keyword = CardTextResources.get_keyword_text(str_val)

        if command.get('explicit_self'):
            target_str = 'このカード'

        filt = command.get('filter') or command.get('target_filter') or {}
        if isinstance(filt, dict) and 'zones' in filt and filt.get('zones'):
            if 'SHIELD_ZONE' in filt.get('zones') or 'SHIELD' in filt.get('zones'):
                target_str = 'カード'

        amt = command.get('amount')
        if amt is None:
            amt = val1 if isinstance(val1, int) else 0

        return CardTextGenerator._format_keyword_grant_text(target_str, str_val, keyword, duration_text, amt, skip_selection=is_target_linked)

@register_formatter("MUTATE")
class MutateFormatter(CommandFormatterBase):
    @classmethod
    def _mutate_tap(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if val1 == 0:
            return f"{target_str}をすべてタップする。"
        return f"{target_str}を{val1}{unit}選び、タップする。"

    @classmethod
    def _mutate_untap(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if val1 == 0:
            return f"{target_str}をすべてアンタップする。"
        return f"{target_str}を{val1}{unit}選び、アンタップする。"

    @classmethod
    def _mutate_power(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        sign = "+" if val1 >= 0 else ""
        return f"{duration_text}{target_str}のパワーを{sign}{val1}する。"

    @classmethod
    def _mutate_add_keyword(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        keyword = CardTextResources.get_keyword_text(str_param)
        return CardTextGenerator._format_keyword_grant_text(target_str, str_param, keyword, duration_text, val1, skip_selection=is_target_linked)

    @classmethod
    def _mutate_remove_keyword(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        keyword = CardTextResources.get_keyword_text(str_param)
        return f"{duration_text}{target_str}の「{keyword}」を無視する。"

    @classmethod
    def _mutate_add_passive(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        if str_param:
            kw = CardTextResources.get_keyword_text(str_param)
            return f"{duration_text}{target_str}に「{kw}」を与える。"
        return f"{duration_text}{target_str}にパッシブ効果を与える。"

    @classmethod
    def _mutate_add_cost(cls, target_str: str, val1: int, unit: str, duration_text: str, str_param: str, is_target_linked: bool) -> str:
        return f"{duration_text}{target_str}にコスト修正を追加する。"

    MUTATE_KIND_HANDLERS = {
        consts.MutationKind.TAP: _mutate_tap.__func__,
        consts.MutationKind.UNTAP: _mutate_untap.__func__,
        consts.MutationKind.POWER_MOD: _mutate_power.__func__,
        consts.MutationKind.GIVE_POWER: _mutate_power.__func__,
        consts.MutationKind.ADD_KEYWORD: _mutate_add_keyword.__func__,
        consts.MutationKind.GIVE_ABILITY: _mutate_add_keyword.__func__,
        consts.MutationKind.REMOVE_KEYWORD: _mutate_remove_keyword.__func__,
        consts.MutationKind.ADD_PASSIVE_EFFECT: _mutate_add_passive.__func__,
        consts.MutationKind.ADD_MODIFIER: _mutate_add_passive.__func__,
        consts.MutationKind.ADD_COST_MODIFIER: _mutate_add_cost.__func__,
    }
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)

        mkind = command.get('mutation_kind', '')
        str_param = command.get('str_val', '')
        input_key = command.get('input_value_key', '')
        input_usage = command.get('input_value_usage') or command.get('input_usage')
        is_target_linked = bool(input_key) and (not input_usage or input_usage == 'TARGET')

        duration_key = command.get('duration') or command.get('input_value_key', '')
        duration_text = ''
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + '、'
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + '、'

        lookup_key = mkind
        if isinstance(mkind, str):
            if hasattr(consts.MutationKind, mkind):
                try:
                    lookup_key = getattr(consts.MutationKind, mkind)
                except Exception:
                    lookup_key = mkind

        handler = cls.MUTATE_KIND_HANDLERS.get(lookup_key)
        if handler:
            try:
                return handler(cls, target_str, val1, unit, duration_text, str_param, is_target_linked)
            except Exception:
                return f'状態変更({tr(mkind)}): {target_str} (値:{val1})'
        return f'状態変更({tr(mkind)}): {target_str} (値:{val1})'

@register_formatter("SUMMON_TOKEN")
class SummonTokenFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)
        token_id = command.get('token_id') if command.get('token_id') is not None else ''
        count = val1 if val1 > 0 else 1
        token_name = 'トークン'
        if token_id:
            translated = tr(token_id)
            if translated == token_id and '_' in token_id and token_id.isupper():
                token_name = 'トークン'
            else:
                token_name = translated
        return f'{token_name}を{count}体出す。'

@register_formatter("REGISTER_DELAYED_EFFECT")
class RegisterDelayedEffectFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        max_cost_src = command.get('max_cost')
        if max_cost_src is None and 'target_filter' in command:
            max_cost_src = (command.get('target_filter') or {}).get('max_cost')
        val1 = max_cost_src if max_cost_src is not None and not isinstance(max_cost_src, dict) else get_command_amount(command, default=0)
        str_val = command.get('str_val', '')
        effect_text = CardTextResources.get_delayed_effect_text(str_val)
        if effect_text == str_val:
            duration = val1 if val1 > 0 else 1
            return f'遅延効果（{str_val}）を{duration}ターン登録する。'
        return effect_text
