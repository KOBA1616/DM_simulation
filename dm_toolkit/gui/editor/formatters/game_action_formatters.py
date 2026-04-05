from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount, get_command_amount_with_fallback
from dm_toolkit.gui.editor.formatters.metadata_flags import SemanticMetadataFlags
from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
from dm_toolkit.gui.i18n import tr

@register_formatter("LOOK_AND_ADD")
class LookAndAddFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        look_count = get_command_amount_with_fallback(command, default=1)
        add_count = command.get("add_count") if command.get("add_count") is not None else 0

        look_count = look_count if command.get("look_count") is not None else get_command_amount_with_fallback(command, default=3)
        add_count = add_count if add_count > 0 else 1

        dest_zone = command.get("destination_zone", "HAND")
        rest_zone = command.get("rest_zone", "DECK_BOTTOM")

        from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter
        dest_zone_list = [dest_zone]
        dest_zone_str = ZoneFormatter.format_zone_list(dest_zone_list, context="to", joiner="、または")

        rest_text = ""
        # Check if there is an input link that resolves to "残りを"
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)
        remainder_prefix = linked_text if "残り" in linked_text else "残りを"

        if rest_zone == "DECK_BOTTOM":
            rest_text = f"{remainder_prefix}好きな順序で山札の下に置く。"
        elif rest_zone == "GRAVEYARD":
            rest_text = f"{remainder_prefix}墓地に置く。"
        else:
            rest_text = f"{remainder_prefix}{tr(rest_zone)}に置く。"

        filter_text = ""
        if target_str != "カード":
            filter_text = f"{target_str}を"

        # Handle modifiers for the destination action
        face_modifier = "表向きにして" if command.get("face_up") else "裏向きにして" if command.get("face_down") else ""
        tapped_modifier = "タップして" if command.get("enter_tapped") else ""
        combined_modifier = face_modifier + tapped_modifier

        # Determine appropriate verb based on the destination zone
        action_verb = "加え" if dest_zone == "HAND" else "置き" if dest_zone in ["MANA_ZONE", "GRAVEYARD", "SHIELD_ZONE", "DECK_BOTTOM", "DECK_TOP", "BUFFER", "UNDER_CARD"] else "出し" if dest_zone == "BATTLE_ZONE" else "置き"

        return f"自分の山札の上から{look_count}枚を見る。その中から{filter_text}{add_count}{unit}{dest_zone_str}{combined_modifier}{action_verb}、{rest_text}"

@register_formatter("SEARCH_DECK")
class SearchDeckFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        dest_zone = command.get("destination_zone", "HAND")
        if not dest_zone:
            dest_zone = "HAND"

        # Delegate logic to TransitionFormatter to avoid custom action phrases
        from dm_toolkit.gui.editor.formatters.zone_formatters import TransitionFormatter
        transition_cmd = command.copy()
        transition_cmd["from_zone"] = "DECK"
        transition_cmd["to_zone"] = dest_zone

        transition_text = TransitionFormatter.format(transition_cmd, ctx)

        # Transition formatter will output something like "山札から{target}を...選び、{zone}に置く。" or "{from_z}から...".
        # However, DM text convention for SEARCH_DECK usually starts with "自分の山札を見る。その中から..."
        # and ends with "その後、山札をシャッフルする。"

        # We can format the transition string, but to maintain the exact convention:
        # We replace "山札から" with "自分の山札を見る。その中から"
        if transition_text.startswith("山札から"):
            transition_text = "自分の山札を見る。その中から" + transition_text[4:]
        elif transition_text.startswith("自分の山札から"):
            transition_text = "自分の山札を見る。その中から" + transition_text[7:]
        elif "山札から" in transition_text:
            transition_text = transition_text.replace("山札から", "自分の山札を見る。その中から")

        if not transition_text.endswith("。"):
            transition_text += "。"

        return f"{transition_text}その後、山札をシャッフルする。"

@register_formatter("PUT_CREATURE")
class PutCreatureFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        action_local = command.copy()
        filter_local = (command.get("filter") or command.get("target_filter") or {}).copy()
        look_count = get_command_amount_with_fallback(command, default=1)
        count = look_count
        filter_zones = filter_local.get("zones", [])
        src_text = ""

        # Prioritize explicit from_zone
        from_z = command.get("from_zone") or command.get("source_zone")
        if isinstance(from_z, list):
            if from_z and "NONE" not in from_z:
                from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter
                src_text = ZoneFormatter.format_zone_list(from_z, context="from", joiner="、または")
                filter_local.pop("zones", None)
        elif from_z and from_z != "NONE":
            src_text = CardTextResources.get_zone_text(from_z) + "から"
            filter_local.pop("zones", None)
        elif filter_zones:
            znames = [CardTextResources.get_zone_text(z) for z in filter_zones]
            src_text = "または".join(znames) + "から"
            filter_local["zones"] = []

        action_local["filter"] = filter_local
        action_local["target_filter"] = filter_local
        put_target_str, put_unit = cls._resolve_target(action_local, ctx)

        placement_text = ""
        placement_target = command.get("placement_target")
        if placement_target:
            from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
            placement_target_desc, _ = TargetResolutionService.build_subject(placement_target)
            placement_text = f"{placement_target_desc}の上に"

        tapped_modifier = "タップして" if command.get("enter_tapped") else ""

        return f"{src_text}{put_target_str}を{count}{put_unit}{placement_text}バトルゾーンに{tapped_modifier}出す。"

@register_formatter("SHUFFLE_DECK")
class ShuffleDeckFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        return "山札をシャッフルする。"

@register_formatter("BOOST_MANA")
class BoostManaFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        look_count = get_command_amount_with_fallback(command, default=1)
        count = look_count
        return f"自分のマナを{count}つ増やす。"

@register_formatter("SEND_SHIELD_TO_GRAVE")
@register_formatter("ADD_SHIELD")
@register_formatter("BREAK_SHIELD")
class ShieldActionFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        action_type = command.get("type", "")
        if action_type == "BREAK_SHIELD":
            ctx.metadata[SemanticMetadataFlags.SHIELDS_BROKEN.value] = True
        elif action_type == "ADD_SHIELD":
            ctx.metadata[SemanticMetadataFlags.SHIELDS_ADDED.value] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        action_type = command.get("type", "")
        to_zone = command.get("to_zone", "GRAVEYARD")

        target_str, unit = cls._resolve_target(command, ctx)
        look_count = get_command_amount_with_fallback(command, default=1)
        amt = look_count

        # Determine prefix and target for shield commands
        tgt = target_str

        if action_type == "BREAK_SHIELD":
            if tgt in ("", "カード", "自分のカード", "それ"):
                tgt = "シールド"
            scope = TargetResolutionService.resolve_action_scope(command)
            if scope == "NONE":
                prefix = TargetResolutionService.resolve_prefix("OPPONENT")
                if not tgt.startswith(prefix):
                    tgt = prefix + tgt
            return f"{tgt}を{amt}つブレイクする。"

        elif action_type == "ADD_SHIELD":
            face_modifier = ""
            if command.get("face_up"):
                face_modifier = "表向きにして"
            elif command.get("face_down"):
                face_modifier = "裏向きにして"

            if "山札" in target_str or target_str == "カード":
                return f"山札の上から{amt}枚を{face_modifier}シールド化する。"
            return f"{target_str}を{amt}つ{face_modifier}シールド化する。"

        else:
            # Handle SEND_SHIELD_TO_GRAVE and general manipulating shield actions
            scope = command.get("target_group") or command.get("scope", "NONE")
            if tgt == "カード":
                tgt = "シールド"

            if scope == 'OPPONENT' or scope == 'PLAYER_OPPONENT':
                tgt = "相手のシールド"
                if amt > 0:
                    tgt += f"を{amt}つ選び、"
                else:
                    tgt += "を"
            else:
                if tgt == "シールド":
                    tgt = f"{tgt}を{amt}つ"
                else:
                    tgt = f"{tgt}を{amt}つ選び、"

            if to_zone == "HAND":
                return f'{tgt}手札に加える。'
            elif to_zone == "MANA_ZONE":
                return f'{tgt}マナゾーンに置く。'
            else:
                return f'{tgt}墓地に置く。'

@register_formatter("SHIELD_BURN")
class ShieldBurnFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        look_count = get_command_amount_with_fallback(command, default=1)
        amt = look_count
        return f"相手のシールドを{amt}つ選び、墓地に置く。"

@register_formatter("DESTROY")
class DestroyFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        ctx.metadata[SemanticMetadataFlags.DESTROYS.value] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        target_str, unit = cls._resolve_target(command, ctx)
        val1 = command.get("look_count") if command.get("look_count") is not None else get_command_amount_with_fallback(command, default=0)

        # テンプレートでなく、直接処理してしまうか、あるいは TextResources のテンプレートを利用する。
        # 既存の text_resources.json に "DESTROY": "{target}を{value1}{unit}破壊する。" があるが、フォーマッタで処理できる
        amt = val1
        if isinstance(val1, int) and val1 <= 0:
            amt = 1
        return f"{target_str}を{amt}{unit}破壊する。"

@register_formatter("REVEAL_CARDS")
class RevealCardsFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        deck_owner = "相手の" if (command.get("target_group") or command.get("scope", "NONE")) in ["OPPONENT", "PLAYER_OPPONENT"] else ""
        look_count = get_command_amount_with_fallback(command, default=1)
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)
        if linked_text:
            return f"{deck_owner}山札の上から、{linked_text}だけ表向きにする。"
        return f"{deck_owner}山札の上から{look_count}枚を表向きにする。"

@register_formatter("COUNT_CARDS")
class CountCardsFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        if not target_str or target_str == "カード":
            from dm_toolkit.gui.i18n import tr
            return f"({tr('COUNT_CARDS')})"
        return f"{target_str}の数を数える。"

@register_formatter("LOCK_SPELL")
class LockSpellFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        player_str = cls._resolve_player_noun(command, ctx)

        look_count = get_command_amount_with_fallback(command, default=1)
        duration_key = command.get("duration", "") or ""
        duration_str = CardTextResources.get_duration_text(duration_key, look_count)

        return f"{player_str}は{duration_str}の間、呪文を唱えられない。"

@register_formatter("SPELL_RESTRICTION")
@register_formatter("CANNOT_PUT_CREATURE")
@register_formatter("CANNOT_SUMMON_CREATURE")
@register_formatter("PLAYER_CANNOT_ATTACK")
@register_formatter("LIMIT_PUT_CREATURE_PER_TURN")
class ActionRestrictionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        atype = command.get("type") or command.get("name") or "NONE"
        player_str = cls._resolve_player_noun(command, ctx)

        look_count = get_command_amount_with_fallback(command, default=1)
        duration_key = command.get("duration", "") or ""
        duration_str = CardTextResources.get_duration_text(duration_key, look_count)

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)

        if atype == 'SPELL_RESTRICTION':
            filt = command.get('filter', {}) or {}
            exact_cost = filt.get('exact_cost') if isinstance(filt, dict) else None
            if linked_text:
                action_text = f'{linked_text}のコストの呪文を唱えられない'
            elif exact_cost is not None:
                action_text = f'コスト{exact_cost}の呪文を唱えられない'
            else:
                action_text = '呪文を唱えられない'
        elif atype == 'CANNOT_PUT_CREATURE':
            action_text = 'クリーチャーを出せない'
        elif atype == 'CANNOT_SUMMON_CREATURE':
            action_text = 'クリーチャーを召喚できない'
        elif atype == 'LIMIT_PUT_CREATURE_PER_TURN':
            amount = command.get('amount', 1)
            action_text = f'各ターン、クリーチャーを{amount}体までしか出せない'
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
        key = command.get('result') or ''
        stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(key, (None, None))
        if stat_name:
            if ctx.sample is not None:
                try:
                    from dm_toolkit.gui.editor.text_generator import CardTextGenerator
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
        target_str, unit = cls._resolve_target(command, ctx)
        return f'{target_str}をカードの下に重ねる。'

@register_formatter("MOVE_TO_UNDER_CARD")
class MoveToUnderCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        look_count = get_command_amount_with_fallback(command, default=1)
        amount = look_count
        if amount == 1:
            return f'{target_str}をカードの下に重ねる。'
        return f'{target_str}を{amount}{unit}カードの下に重ねる。'

@register_formatter("RESET_INSTANCE")
class ResetInstanceFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        reset_mode = command.get('str_param') or command.get('mutation_kind') or 'DEFAULT'

        if reset_mode == 'UNSEAL':
            return f'{target_str}の封印を外す。'
        elif reset_mode == 'RECONSTRUCT':
            return f'{target_str}を再構築する。'
        elif reset_mode == 'IGNORE_EFFECTS':
            return f'{target_str}に付与された効果を無視する。'
        elif reset_mode == 'RESET_MODIFIERS':
            return f'{target_str}の継続的効果をリセットする。'

        return f'{target_str}の状態を初期化する。'

@register_formatter("SELECT_TARGET")
class SelectTargetFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        ctx.metadata[SemanticMetadataFlags.TARGETS.value] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        val1 = get_command_amount_with_fallback(command, default=1)
        amount = val1
        choosing_player = command.get("choosing_player")
        if choosing_player:
            from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
            prefix = TargetResolutionService.resolve_noun(choosing_player)
            if prefix:
                return f'{prefix}は{target_str}を{amount}{unit}選ぶ。'
        return f'{target_str}を{amount}{unit}選ぶ。'

@register_formatter("COST_REFERENCE")
class CostReferenceFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        ref_mode = command.get('ref_mode', '')
        return f'（コスト参照: {tr(ref_mode)}）'

@register_formatter("SEARCH_DECK_BOTTOM")
class SearchDeckBottomFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        val1 = get_command_amount_with_fallback(command, default=1)
        amt = val1
        return f'山札の下から{amt}枚を探す。'

@register_formatter("SEND_TO_DECK_BOTTOM")
class SendToDeckBottomFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        val1 = get_command_amount_with_fallback(command, default=1)
        amount = val1
        amt = val1
        return f'{target_str}を{amt}{unit}山札の下に置く。'

@register_formatter("RESOLVE_BATTLE")
class ResolveBattleFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        battle_context_id = command.get("id")
        if battle_context_id:
            ctx.metadata["battle_context_id"] = battle_context_id

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        source_target = command.get("source_target")
        destination_target = command.get("destination_target")

        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService

        dest_str = ""
        if destination_target:
            dest_str, _ = TargetResolutionService.build_subject(destination_target)
        else:
            dest_str, _ = cls._resolve_target(command, ctx)

        if not dest_str:
            dest_str = "対象"

        if source_target:
            source_str, _ = TargetResolutionService.build_subject(source_target)
            if source_str and source_str != "カード" and source_str != "自身":
                return f'{source_str}と{dest_str}をバトルさせる。'

        return f'{dest_str}とバトルさせる。'

@register_formatter("MODIFY_POWER")
class ModifyPowerFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        val = command.get('amount', 0)
        sign = '+' if val >= 0 else ''
        return f'{target_str}のパワーを{sign}{val}する。'

@register_formatter("SELECT_NUMBER")
class SelectNumberFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        choosing_player = command.get("choosing_player")
        prefix = ""
        if choosing_player:
            from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
            prefix = TargetResolutionService.resolve_noun(choosing_player)
            if prefix:
                prefix = f'{prefix}は'

        max_value_link = command.get("max_value_link")
        if max_value_link:
            # Re-use the InputLinkFormatter logic to describe the linked name
            # To handle max_value_link properly, we inject it into the command dict as input_link
            # so resolve_linked_value_text can process it correctly since it looks up input_value_key
            temp_command = command.copy()
            temp_command["input_value_key"] = max_value_link
            temp_command["input_value_usage"] = "MAX_VALUE"

            link_name = InputLinkFormatter.resolve_linked_value_text(temp_command, default=max_value_link, context_commands=ctx.current_commands_list)

            # Remove leading "その" and trailing words to isolate the core word
            if link_name.startswith("その"):
                link_name = link_name[2:]

            # Clean up trailing logic if it is "のコスト" -> "コスト"
            if link_name.endswith("のコスト"):
                link_name = "コスト"
            elif link_name.endswith("のパワー"):
                link_name = "パワー"

            # "コスト以下の数字を1つ選ぶ。"
            return f'{prefix}{link_name}以下の数字を1つ選ぶ。'

        min_val = command.get("min_value", 1)
        max_val = command.get("amount", 6)
        if min_val > 0 and max_val > 0:
            return f'{prefix}{min_val}～{max_val}の数字を1つ選ぶ。'
        return f'{prefix}数字を1つ選ぶ。'

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

        # DM 特有の「次のうちいずれかXつを選ぶ」フォーマット
        if amount == 1:
            lines.append(f'次のうちいずれか1つを選ぶ。')
        else:
            lines.append(f'次のうちいずれか{amount}つを選ぶ。{suffix}')

        # 箇条書き（・）でフォーマット
        block_text = CommandListFormatter.format_block(options, ctx, bullet="▶")
        if block_text:
            lines.append(block_text)

        return '\n'.join(lines)

@register_formatter("QUERY")
class QueryFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        mode = command.get('str_param') or command.get('query_mode') or ''

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)

        input_usage = command.get("input_value_usage") or command.get("input_usage") or ""
        stat_name, stat_unit = CardTextResources.STAT_KEY_MAP.get(str(mode), (None, None))

        if stat_name:
            base = f'{stat_name}{stat_unit}を数える。'
            if linked_text:
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
            input_key = command.get('input_value_key') or command.get('input_link')
            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                if usage_label:
                    base += f'（{usage_label}）'
            return base

        if str(mode) == 'SELECT_OPTION':
            sel_count = command.get('amount', 1)

            # Pass sample/context to descriptive formatting if needed
            filter_txt = FilterTextFormatter.format_filter_text(command.get('filter', {}))

            input_key = command.get('input_value_key') or command.get('input_link')
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
        input_key = command.get('input_value_key') or command.get('input_link')
        if input_key:
            usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
            if usage_label:
                base += f'（{usage_label}）'
        return base

@register_formatter("IGNORE_ABILITY")
class IgnoreAbilityFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str_lock = cls._resolve_player_noun(command, ctx)

        filt = command.get('filter', {}) or {}
        types = []
        if isinstance(filt, dict):
            types = filt.get('types', []) or []
        type_txt = 'カード'
        if types:
            type_txt = '・'.join([tr(t) for t in types])

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)

        if linked_text:
            return f'{target_str_lock}のコスト{linked_text}の{type_txt}の能力は無視される。'
        if isinstance(filt, dict) and filt.get('exact_cost') is not None:
            return f"{target_str_lock}のコスト{filt.get('exact_cost')}の{type_txt}の能力は無視される。"
        return f'{target_str_lock}の{type_txt}の能力は無視される。'
