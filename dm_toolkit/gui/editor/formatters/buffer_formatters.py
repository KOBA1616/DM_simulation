from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter
from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount, get_command_amount_with_fallback
from dm_toolkit.gui.i18n import tr

@register_formatter("LOOK_TO_BUFFER")
class LookToBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        try:
            val1 = int(get_command_amount_with_fallback(command, default=int(command.get('value1', 0))))
        except (TypeError, ValueError):
            val1 = 0

        src_zone = tr(command.get('from_zone', 'DECK'))
        amt = val1 if val1 > 0 else 1
        return f'{src_zone}から{amt}枚を見る。'

@register_formatter("REVEAL_TO_BUFFER")
class RevealToBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        try:
            val1 = int(get_command_amount_with_fallback(command, default=int(command.get('value1', 0))))
        except (TypeError, ValueError):
            val1 = 0

        src_zone = tr(command.get('from_zone', 'DECK'))
        amt = val1 if val1 > 0 else 1
        return f'{src_zone}から{amt}枚を表向きにしてバッファに置く。'

@register_formatter("SELECT_FROM_BUFFER")
class SelectFromBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        try:
            val1 = int(get_command_amount_with_fallback(command, default=int(command.get('value1', 0))))
        except (TypeError, ValueError):
            val1 = 0

        filter_def = command.get('filter') or command.get('target_filter') or {}
        civs = filter_def.get('civilizations', []) if filter_def else []
        types = filter_def.get('types', []) if filter_def else []
        races = filter_def.get('races', []) if filter_def else []

        civ_part = ''
        if civs:
            civ_part = '/'.join((CardTextResources.get_civilization_text(c) for c in civs)) + 'の'

        if races:
            type_part = '/'.join(races)
        elif 'ELEMENT' in types:
            type_part = 'エレメント'
        elif 'SPELL' in types and 'CREATURE' not in types:
            type_part = '呪文'
        elif 'CREATURE' in types:
            type_part = 'クリーチャー'
        elif types:
            type_part = '/'.join((tr(t) for t in types if t))
        else:
            type_part = 'カード'

        if val1 <= 0:
            qty_part = 'すべて'
        else:
            qty_part = f'{val1}枚'

        return f'見た{civ_part}{type_part}{qty_part}を選ぶ。'

@register_formatter("PLAY_FROM_BUFFER")
class PlayFromBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        return f'選んだカード（{target_str}）を使う。'

@register_formatter("MOVE_BUFFER_TO_ZONE")
class MoveBufferToZoneFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        try:
            val1 = int(get_command_amount_with_fallback(command, default=int(command.get('value1', 0))))
        except (TypeError, ValueError):
            val1 = 0

        to_zone = tr(command.get('to_zone', 'HAND'))
        filter_def = command.get('filter') or command.get('target_filter') or {}
        civs = filter_def.get('civilizations', []) if filter_def else []
        types = filter_def.get('types', []) if filter_def else []
        races = filter_def.get('races', []) if filter_def else []
        has_filter = bool(civs or types or races)

        if has_filter:
            civ_part = ''
            if civs:
                civ_part = '/'.join((CardTextResources.get_civilization_text(c) for c in civs)) + 'の'
            if races:
                type_part = '/'.join(races)
            elif 'ELEMENT' in types:
                type_part = 'エレメント'
            elif 'SPELL' in types and 'CREATURE' not in types:
                type_part = '呪文'
            elif 'CREATURE' in types:
                type_part = 'クリーチャー'
            elif types:
                type_part = '/'.join((tr(t) for t in types if t))
            else:
                type_part = 'カード'

            qty = QuantityFormatter.format_quantity(val1, "枚", up_to=False, is_all=(val1==0))
            return f'見た{civ_part}{type_part}{qty}を選び、{to_zone}に置く。'

        if val1 > 0:
            qty = QuantityFormatter.format_quantity(val1, "枚", up_to=False, is_all=False)
            return f'{qty}を{to_zone}に置く。'
        return f'選んだカードをすべて{to_zone}に置く。'

@register_formatter("MOVE_BUFFER_REMAIN_TO_ZONE")
class MoveBufferRemainToZoneFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        to_zone = tr(command.get('to_zone', 'DECK_BOTTOM'))
        return f'残りを{to_zone}に置く。'
