from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.i18n import tr

class LogicFormatterUtils:
    @classmethod
    def get_cond_text(cls, action: Dict[str, Any], default_text: str = "もし条件を満たすなら") -> str:
        cond_detail = action.get('condition', {}) or action.get('target_filter', {})
        cond_text = ''

        def _handle_opponent_draw_count(d):
            val = d.get('value', 0)
            return f'相手がカードを{val}枚目以上引いたなら'

        def _handle_compare_stat(d):
            key = d.get('stat_key', '')
            op = d.get('op', '=')
            val = d.get('value', 0)
            stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ''))
            if op == '>=':
                op_text = f'{val}{unit}以上'
            elif op == '<=':
                op_text = f'{val}{unit}以下'
            elif op == '=' or op == '==':
                op_text = f'{val}{unit}'
            elif op == '>':
                op_text = f'{val}{unit}より多い'
            elif op == '<':
                op_text = f'{val}{unit}より少ない'
            else:
                op_text = f'{val}{unit}'
            return f'自分の{stat_name}が{op_text}なら'

        def _handle_shield_count(d):
            val = d.get('value', 0)
            op = d.get('op', '>=')
            op_text = '以上' if op == '>=' else '以下' if op == '<=' else ''
            if op == '=':
                op_text = ''
            return f'自分のシールドが{val}つ{op_text}なら'

        def _handle_compare_input(d, action_local):
            val = d.get('value', 0)
            op = d.get('op', '>=')
            input_key = action_local.get('input_value_key', '')
            input_desc_map = {'spell_count': '墓地の呪文の数', 'card_count': 'カードの数', 'creature_count': 'クリーチャーの数', 'element_count': 'エレメントの数'}
            input_desc = input_desc_map.get(input_key, input_key if input_key else '入力値')
            if op == '>=':
                try:
                    op_text = f'{int(val) + 1}以上'
                except Exception:
                    op_text = f'{val}以上'
            elif op == '<=':
                op_text = f'{val}以下'
            elif op == '=' or op == '==':
                op_text = f'{val}'
            elif op == '>':
                op_text = f'{val}より多い'
            elif op == '<':
                op_text = f'{val}より少ない'
            else:
                op_text = f'{val}'
            return f'{input_desc}が{op_text}なら'

        def _handle_civ_match(d):
            return 'マナゾーンに同じ文明があれば'

        def _handle_played_without_mana(d):
            return '指定した対象をコストを支払わずに出していれば'

        def _handle_mana_civ_count(d):
            val = d.get('value', 0)
            op = d.get('op', '>=')
            op_text = '以上' if op == '>=' else '以下' if op == '<=' else 'と同じ' if op == '=' else ''
            return f'自分のマナゾーンにある文明の数が{val}{op_text}なら'

        COND_HANDLERS = {
            'OPPONENT_DRAW_COUNT': lambda d: _handle_opponent_draw_count(d),
            'COMPARE_STAT': lambda d: _handle_compare_stat(d),
            'SHIELD_COUNT': lambda d: _handle_shield_count(d),
            'COMPARE_INPUT': lambda d: _handle_compare_input(d, action),
            'CIVILIZATION_MATCH': lambda d: _handle_civ_match(d),
            'PLAYED_WITHOUT_MANA_TARGET': lambda d: _handle_played_without_mana(d),
            'MANA_CIVILIZATION_COUNT': lambda d: _handle_mana_civ_count(d)
        }

        if isinstance(cond_detail, dict):
            cond_type = cond_detail.get('type', 'NONE')
            handler = COND_HANDLERS.get(cond_type)
            if handler:
                try:
                    cond_text = handler(cond_detail)
                except Exception:
                    cond_text = ''

        if not cond_text:
            cond_text = default_text

        return cond_text


@register_formatter("IF")
class IfFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        cond_text = LogicFormatterUtils.get_cond_text(command, "もし条件を満たすなら")

        if_true_cmds = command.get('if_true', [])
        if_true_texts = []
        for cmd in if_true_cmds:
            if isinstance(cmd, dict):
                cmd_text = CardTextGenerator._format_command(cmd, ctx)
                if cmd_text:
                    if_true_texts.append(cmd_text)

        if if_true_texts:
            actions_text = '、'.join(if_true_texts)
            return f'{cond_text}、{actions_text}'
        else:
            return f'（{cond_text}）'

@register_formatter("IF_ELSE")
class IfElseFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        cond_text = LogicFormatterUtils.get_cond_text(command, "もし条件を満たすなら")

        if_true_cmds = command.get('if_true', [])
        if_false_cmds = command.get('if_false', [])

        if_true_texts = []
        for cmd in if_true_cmds:
            if isinstance(cmd, dict):
                cmd_text = CardTextGenerator._format_command(cmd, ctx)
                if cmd_text:
                    if_true_texts.append(cmd_text)

        if_false_texts = []
        for cmd in if_false_cmds:
            if isinstance(cmd, dict):
                cmd_text = CardTextGenerator._format_command(cmd, ctx)
                if cmd_text:
                    if_false_texts.append(cmd_text)

        result_parts = []
        if if_true_texts:
            result_parts.append(f'{cond_text}、' + '、'.join(if_true_texts))
        if if_false_texts:
            result_parts.append('そうでなければ、' + '、'.join(if_false_texts))

        if result_parts:
            return '。'.join(result_parts) + '。'
        else:
            return f'（条件分岐: {cond_text}）'

@register_formatter("ELSE")
class ElseFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        return '（そうでなければ）'
