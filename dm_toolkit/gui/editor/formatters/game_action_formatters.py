from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.target_scope_resolver import TargetScopeResolver
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.i18n import tr

@register_formatter("LOOK_AND_ADD")
class LookAndAddFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        add_count = command.get("add_count") if command.get("add_count") is not None else 0

        look_count = look_count if look_count > 0 else 3
        add_count = add_count if add_count > 0 else 1
        rest_zone = command.get("rest_zone", "DECK_BOTTOM")

        rest_text = ""
        if rest_zone == "DECK_BOTTOM":
            rest_text = "残りを好きな順序で山札の下に置く。"
        elif rest_zone == "GRAVEYARD":
            rest_text = "残りを墓地に置く。"
        else:
            rest_text = f"残りを{tr(rest_zone)}に置く。"

        filter_text = ""
        if target_str != "カード":
            filter_text = f"{target_str}を"

        return f"自分の山札の上から{look_count}枚を見る。その中から{filter_text}{add_count}{unit}手札に加え、{rest_text}"

@register_formatter("SEARCH_DECK")
class SearchDeckFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)

        dest_zone = command.get("destination_zone", "HAND")
        if not dest_zone:
            dest_zone = "HAND"
        zone_str = CardTextResources.get_zone_text(dest_zone)
        count = look_count if look_count > 0 else 1

        if dest_zone == "HAND":
            action_phrase = "手札に加える"
        elif dest_zone == "MANA_ZONE":
            action_phrase = "マナゾーンに置く"
        elif dest_zone == "GRAVEYARD":
            action_phrase = "墓地に置く"
        else:
            action_phrase = f"{zone_str}に置く"

        template = f"自分の山札を見る。その中から{target_str}を{count}{unit}選び、{action_phrase}。その後、山札をシャッフルする。"
        if count == 1:
            template = f"自分の山札を見る。その中から{target_str}を1{unit}選び、{action_phrase}。その後、山札をシャッフルする。"
        return template

@register_formatter("PUT_CREATURE")
class PutCreatureFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        action_local = command.copy()
        filter_local = (command.get("filter") or command.get("target_filter") or {}).copy()
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        count = look_count if look_count > 0 else 1
        filter_zones = filter_local.get("zones", [])
        src_text = ""

        # Prioritize explicit from_zone
        from_z = command.get("from_zone") or command.get("source_zone")
        if from_z and from_z != "NONE":
            src_text = CardTextResources.get_zone_text(from_z) + "から"
            filter_local.pop("zones", None)
        elif filter_zones:
            znames = [CardTextResources.get_zone_text(z) for z in filter_zones]
            src_text = "または".join(znames) + "から"
            filter_local["zones"] = []

        action_local["filter"] = filter_local
        action_local["target_filter"] = filter_local
        put_target_str, put_unit = cls._resolve_target(action_local, ctx.is_spell)

        return f"{src_text}{put_target_str}を{count}{put_unit}バトルゾーンに出す。"

@register_formatter("SHUFFLE_DECK")
class ShuffleDeckFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        return "山札をシャッフルする。"

@register_formatter("BOOST_MANA")
class BoostManaFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        count = look_count if look_count > 0 else 1
        return f"自分のマナを{count}つ増やす。"

@register_formatter("BREAK_SHIELD")
class BreakShieldFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        count = look_count if look_count > 0 else 1
        tgt = target_str
        if tgt in ("", "カード", "自分のカード", "それ"):
            tgt = "シールド"
        if not command.get("target_group") and not command.get("scope") or (command.get("target_group") or command.get("scope")) == "NONE":
            if "相手" not in tgt:
                tgt = "相手の" + tgt
        return f"{tgt}を{count}つブレイクする。"

@register_formatter("ADD_SHIELD")
class AddShieldFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amt = look_count if look_count > 0 else 1
        if "山札" in target_str or target_str == "カード":
            return f"山札の上から{amt}枚をシールド化する。"
        return f"{target_str}を{amt}つシールド化する。"

@register_formatter("SHIELD_BURN")
class ShieldBurnFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amt = look_count if look_count > 0 else 1
        return f"相手のシールドを{amt}つ選び、墓地に置く。"

@register_formatter("REVEAL_CARDS")
class RevealCardsFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        deck_owner = "相手の" if (command.get("target_group") or command.get("scope", "NONE")) in ["OPPONENT", "PLAYER_OPPONENT"] else ""
        input_key = command.get("input_value_key") or command.get("input_link") or ""
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        if input_key:
            return f"{deck_owner}山札の上から、その数だけ表向きにする。"
        return f"{deck_owner}山札の上から{look_count}枚を表向きにする。"

@register_formatter("COUNT_CARDS")
class CountCardsFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        if not target_str or target_str == "カード":
            from dm_toolkit.gui.i18n import tr
            return f"({tr('COUNT_CARDS')})"
        return f"{target_str}の数を数える。"

@register_formatter("LOCK_SPELL")
class LockSpellFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        scope = command.get("target_group") or command.get("scope", "NONE")
        noun = TargetScopeResolver.resolve_noun(scope)
        if noun:
            player_str = noun
        else:
            player_str, _ = cls._resolve_target(command, ctx.is_spell)

        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        duration_key = command.get("duration", "") or ""
        if duration_key:
            duration_str = CardTextResources.get_duration_text(duration_key)
        else:
            duration_str = f"{look_count}ターン" if look_count > 0 else "このターン"

        return f"{player_str}は{duration_str}の間、呪文を唱えられない。"

@register_formatter("SPELL_RESTRICTION")
@register_formatter("CANNOT_PUT_CREATURE")
@register_formatter("CANNOT_SUMMON_CREATURE")
@register_formatter("PLAYER_CANNOT_ATTACK")
class ActionRestrictionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        atype = command.get("type") or command.get("name") or "NONE"
        scope = command.get("target_group") or command.get("scope", "NONE")
        noun = TargetScopeResolver.resolve_noun(scope)
        if noun:
            player_str = noun
        else:
            player_str, _ = cls._resolve_target(command, ctx.is_spell)

        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        duration_key = command.get("duration", "") or ""
        if duration_key:
            duration_str = CardTextResources.get_duration_text(duration_key)
        else:
            duration_str = f"{look_count}ターン" if look_count > 0 else "このターン"

        input_key = command.get("input_value_key") or command.get("input_link") or ""

        if atype == 'SPELL_RESTRICTION':
            filt = command.get('filter', {}) or {}
            exact_cost = filt.get('exact_cost') if isinstance(filt, dict) else None
            if input_key:
                action_text = '入力値と同じコストの呪文を唱えられない'
            elif exact_cost is not None:
                action_text = f'コスト{exact_cost}の呪文を唱えられない'
            else:
                action_text = '呪文を唱えられない'
        elif atype == 'CANNOT_PUT_CREATURE':
            action_text = 'クリーチャーを出せない'
        elif atype == 'CANNOT_SUMMON_CREATURE':
            action_text = 'クリーチャーを召喚できない'
        else:
            action_text = '攻撃できない'

        return f"{player_str}は{duration_str}の間、{action_text}。"

@register_formatter("STAT")
class StatFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        key = command.get('stat')
        amount = command.get('amount')
        if amount is None: amount = 0
        if key:
            stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(key), (None, None))
            if stat_name:
                return f'統計更新: {stat_name} += {amount}'
        return f'統計更新: {tr(str(key))} += {amount}'

@register_formatter("GET_GAME_STAT")
class GetGameStatFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        key = command.get('result') or ''
        stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(key, (None, None))
        if stat_name:
            if ctx.sample is not None:
                try:
                    val = CardTextGenerator._compute_stat_from_sample(key, ctx.sample)
                    if val is not None:
                        return f'{stat_name}（例: {val}{stat_unit}）'
                except Exception:
                    pass
            return f'{stat_name}'
        return f'（{tr(key)}を参照）'

@register_formatter("FLOW")
class FlowFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        ftype = command.get('flow_type', '')
        flow_value = command.get('amount')
        if flow_value is None: flow_value = 0
        if ftype == 'PHASE_CHANGE':
            phase_name = CardTextResources.PHASE_MAP.get(flow_value, str(flow_value))
            return f'{phase_name}フェーズへ移行する。'
        if ftype == 'TURN_CHANGE':
            return 'ターンを終了する。'
        if ftype == 'SET_ACTIVE_PLAYER':
            return '手番を変更する。'
        return f'進行制御({tr(ftype)}): {flow_value}'

@register_formatter("GAME_RESULT")
class GameResultFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        result = command.get('result', '')
        return f'ゲームを終了する（{tr(result)}）。'

@register_formatter("DECLARE_NUMBER")
class DeclareNumberFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        min_val = command.get('min_value', 1)
        max_val = command.get('amount', 10)
        return f'数字を1つ宣言する（{min_val}～{max_val}）。'

@register_formatter("DECIDE")
class DecideFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        selected_option = command.get('selected_option_index')
        if isinstance(selected_option, int) and selected_option >= 0:
            return f'選択肢{selected_option}を確定する。'
        indices = command.get('selected_indices') or []
        if isinstance(indices, list) and indices:
            return f'選択（{indices}）を確定する。'
        return '選択を確定する。'

@register_formatter("DECLARE_REACTION")
class DeclareReactionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        if command.get('pass'):
            return 'リアクション: パスする。'
        reaction_index = command.get('reaction_index')
        if isinstance(reaction_index, int):
            return f'リアクションを宣言する（インデックス {reaction_index}）。'
        return 'リアクションを宣言する。'

@register_formatter("ATTACH")
class AttachFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        return f'{target_str}をカードの下に重ねる。'

@register_formatter("MOVE_TO_UNDER_CARD")
class MoveToUnderCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amount = look_count if look_count > 0 else 1
        if amount == 1:
            return f'{target_str}をカードの下に重ねる。'
        return f'{target_str}を{amount}{unit}カードの下に重ねる。'

@register_formatter("RESET_INSTANCE")
class ResetInstanceFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        return f'{target_str}の状態を初期化する（効果を無視する）。'

@register_formatter("SELECT_TARGET")
class SelectTargetFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val1 = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amount = val1 if val1 > 0 else 1
        return f'{target_str}を{amount}{unit}選ぶ。'

@register_formatter("COST_REFERENCE")
class CostReferenceFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        ref_mode = command.get('ref_mode', '')
        return f'（コスト参照: {tr(ref_mode)}）'

@register_formatter("SEND_SHIELD_TO_GRAVE")
class SendShieldToGraveFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amt = look_count if look_count > 0 else 1
        scope = command.get("target_group") or command.get("scope", "NONE")
        if scope == 'OPPONENT' or scope == 'PLAYER_OPPONENT':
            return f'相手のシールドを{amt}つ選び、墓地に置く。'
        return f'{target_str}を{amt}つ墓地に置く。'

@register_formatter("SEARCH_DECK_BOTTOM")
class SearchDeckBottomFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        val1 = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amt = val1 if val1 > 0 else 1
        return f'山札の下から{amt}枚を探す。'

@register_formatter("SEND_TO_DECK_BOTTOM")
class SendToDeckBottomFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val1 = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        amt = val1 if val1 > 0 else 1
        return f'{target_str}を{amt}{unit}山札の下に置く。'

@register_formatter("RESOLVE_BATTLE")
class ResolveBattleFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        return f'{target_str}とバトルさせる。'

@register_formatter("MODIFY_POWER")
class ModifyPowerFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val = command.get('amount', 0)
        sign = '+' if val >= 0 else ''
        return f'{target_str}のパワーを{sign}{val}する。'

@register_formatter("SELECT_NUMBER")
class SelectNumberFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        min_val = command.get("min_value", 1)
        max_val = command.get("amount", 6)
        if min_val > 0 and max_val > 0:
            return f'{min_val}～{max_val}の数字を1つ選ぶ。'
        return "数字を1つ選ぶ。"

@register_formatter("SELECT_OPTION")
class SelectOptionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.formatters.command_list_formatter import CommandListFormatter
        options = command.get('options', [])
        lines = []
        amount = command.get('amount') if command.get('amount') is not None else 1
        flags = command.get('flags', []) or []
        optional = command.get('optional', False)
        if isinstance(flags, list) and 'ALLOW_DUPLICATES' in flags:
            optional = True
        suffix = '（同じものを選んでもよい）' if optional else ''
        lines.append(f'次の中から{amount}回選ぶ。{suffix}')

        block_text = CommandListFormatter.format_block(options, ctx, bullet="> ")
        if block_text:
            lines.append(block_text)

        return '\n'.join(lines)

@register_formatter("QUERY")
class QueryFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        mode = command.get('query_mode') or ''
        input_key = command.get("input_value_key") or command.get("input_link") or ""
        input_usage = command.get("input_value_usage") or command.get("input_usage") or ""
        stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(mode), (None, None))

        if stat_name:
            base = f'{stat_name}{stat_unit}を数える。'
            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                if usage_label:
                    base += f'（{usage_label}）'
            return base

        if str(mode) == 'CARDS_MATCHING_FILTER' or str(mode) == 'COUNT_CARDS' or (not mode):
            filter_def = command.get('filter', {})
            zones = filter_def.get('zones', [])
            if target_str and target_str != 'カード':
                base = f'{target_str}の数を数える。'
            elif zones:
                zone_names = [tr(z) for z in zones]
                if len(zone_names) == 1:
                    base = f'{zone_names[0]}のカードの枚数を数える。'
                else:
                    base = f"{'または'.join(zone_names)}のカードの枚数を数える。"
            else:
                base = 'カードの数を数える。'
            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                if usage_label:
                    base += f'（{usage_label}）'
            return base

        if str(mode) == 'SELECT_OPTION':
            sel_count = command.get('amount', 1)

            # Pass sample/context to descriptive formatting if needed
            from dm_toolkit.gui.editor.text_generator import CardTextGenerator
            filter_txt = FilterTextFormatter.format_filter_text(command.get('filter', {}))

            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                cnt_txt = '指定数'
                if usage_label:
                    cnt_txt = f'入力値（{usage_label}）'
                if filter_txt:
                    return f'{filter_txt}から{cnt_txt}選ぶ。'
                return f'条件に合うカードから{cnt_txt}選ぶ。'
            if filter_txt:
                return f'{filter_txt}から{sel_count}枚選ぶ。'
            return f'条件に合うカードから{sel_count}枚選ぶ。'

        base = f'質問: {tr(mode)}'
        if input_key:
            usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
            if usage_label:
                base += f'（{usage_label}）'
        return base

@register_formatter("IGNORE_ABILITY")
class IgnoreAbilityFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        scope = TargetScopeResolver.resolve_action_scope(command)
        target_str_lock = TargetScopeResolver.resolve_noun(scope)
        if not target_str_lock:
            target_str_lock, _ = cls._resolve_target(command, ctx.is_spell)

        look_count = command.get("look_count") if command.get("look_count") is not None else get_command_amount(command, default=0)
        duration_key = command.get('duration', '') or ''
        duration_text = CardTextResources.get_duration_text(duration_key) if duration_key else f'{look_count}ターン' if look_count > 0 else 'このターン'

        filt = command.get('filter', {}) or {}
        types = []
        if isinstance(filt, dict):
            types = filt.get('types', []) or []
        type_txt = 'カード'
        if types:
            type_txt = '・'.join([tr(t) for t in types])

        input_key = command.get("input_value_key") or command.get("input_link") or ""

        if input_key:
            return f'{target_str_lock}のコスト入力値と同じ{type_txt}の能力は{duration_text}の間、無視される。'
        if isinstance(filt, dict) and filt.get('exact_cost') is not None:
            return f"{target_str_lock}のコスト{filt.get('exact_cost')}の{type_txt}の能力は{duration_text}の間、無視される。"
        return f'{target_str_lock}の{type_txt}の能力は{duration_text}の間、無視される。'
