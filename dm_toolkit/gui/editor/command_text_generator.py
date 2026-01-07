# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
from dm_toolkit.gui.localization import tr
import dm_ai_module as dm

class CommandTextGenerator:
    """
    Generates localized Japanese text descriptions for GameCommand objects.
    """

    @classmethod
    def generate_text(cls, command: dm.GameCommand, game_state: Optional[dm.GameState] = None) -> str:
        if not command:
            return ""

        try:
            cmd_type = command.get_type()

            if cmd_type == dm.CommandType.TRANSITION:
                return cls._format_transition(command, game_state)
            elif cmd_type == dm.CommandType.MUTATE:
                return cls._format_mutate(command, game_state)
            elif cmd_type == dm.CommandType.ATTACH:
                return cls._format_attach(command, game_state)
            elif cmd_type == dm.CommandType.FLOW:
                return cls._format_flow(command, game_state)
            elif cmd_type == dm.CommandType.QUERY:
                return cls._format_query(command, game_state)
            elif cmd_type == dm.CommandType.DECIDE:
                return cls._format_decide(command, game_state)
            elif cmd_type == dm.CommandType.DECLARE_REACTION:
                return cls._format_reaction(command, game_state)
            elif cmd_type == dm.CommandType.STAT:
                return cls._format_stat(command, game_state)
            elif cmd_type == dm.CommandType.GAME_RESULT:
                return cls._format_game_result(command, game_state)
            else:
                return f"未対応のコマンド: {cmd_type}"
        except Exception as e:
            return f"コマンドテキスト生成エラー: {e}"

    @classmethod
    def _get_card_name(cls, instance_id: int, game_state: Optional[dm.GameState]) -> str:
        # Note: Ideally we need card_db to get the name.
        # Without card_db, we can only show the instance ID.
        # If game_state is provided, we might be able to inspect more, but currently just ID.
        if not game_state:
            return f"カード#{instance_id}"

        try:
            instance = game_state.get_card_instance(instance_id)
            if not instance:
                return f"カード#{instance_id}"
            # TODO: If we have a mechanism to lookup names globally or pass card_db, use it.
            # For now, return ID representation.
            return f"カード<{instance.card_id}>(ID:{instance_id})"
        except:
            return f"カード#{instance_id}"

    @classmethod
    def _format_transition(cls, cmd: dm.TransitionCommand, state: Optional[dm.GameState]) -> str:
        target = cls._get_card_name(cmd.card_instance_id, state)
        from_z = tr(str(cmd.from_zone).split('.')[-1])
        to_z = tr(str(cmd.to_zone).split('.')[-1])
        return f"{target}を{from_z}から{to_z}へ移動"

    @classmethod
    def _format_mutate(cls, cmd: dm.MutateCommand, state: Optional[dm.GameState]) -> str:
        mtype = cmd.mutation_type
        target = cls._get_card_name(cmd.target_instance_id, state)
        val = cmd.int_value
        sval = cmd.str_value

        if mtype == dm.MutationType.TAP:
            return f"{target}をタップ"
        elif mtype == dm.MutationType.UNTAP:
            return f"{target}をアンタップ"
        elif mtype == dm.MutationType.POWER_MOD:
            sign = "+" if val >= 0 else ""
            return f"{target}のパワーを{sign}{val}修正"
        elif mtype == dm.MutationType.ADD_KEYWORD:
            return f"{target}に「{tr(sval)}」を与える"
        elif mtype == dm.MutationType.REMOVE_KEYWORD:
            return f"{target}から「{tr(sval)}」を削除"
        elif mtype == dm.MutationType.ADD_PASSIVE_EFFECT:
            return f"{target}にパッシブ効果を追加"
        elif mtype == dm.MutationType.ADD_COST_MODIFIER:
            return f"{target}にコスト修正を追加"

        return f"状態変更({tr(mtype)}): {target} (値:{val}/{sval})"

    @classmethod
    def _format_attach(cls, cmd: dm.AttachCommand, state: Optional[dm.GameState]) -> str:
        base = cls._get_card_name(cmd.target_base_card_id, state)
        card_id = cmd.card_to_attach_id
        return f"カード<{card_id}>を{base}の下に重ねる"

    @classmethod
    def _format_flow(cls, cmd: dm.FlowCommand, state: Optional[dm.GameState]) -> str:
        ftype = cmd.flow_type
        val = cmd.new_value

        if ftype == dm.FlowType.PHASE_CHANGE:
            p_name = tr(str(dm.Phase(val)).split('.')[-1])
            return f"フェーズ開始: {p_name}"
        elif ftype == dm.FlowType.TURN_CHANGE:
            return f"ターン変更: プレイヤー{val}"
        elif ftype == dm.FlowType.STEP_CHANGE:
            return f"ステップ変更: {val}"
        elif ftype == dm.FlowType.SET_ACTIVE_PLAYER:
            return f"手番プレイヤー変更: {val}"

        return f"進行制御({tr(ftype)}): {val}"

    @classmethod
    def _format_query(cls, cmd: dm.QueryCommand, state: Optional[dm.GameState]) -> str:
        return f"クエリ: {tr(str(cmd.query_type).split('.')[-1])}（対象数: {len(cmd.valid_targets)}）"

    @classmethod
    def _format_decide(cls, cmd: dm.DecideCommand, state: Optional[dm.GameState]) -> str:
        if cmd.selected_option_index >= 0:
            return f"決定: 選択肢{cmd.selected_option_index}"
        return f"決定: インデックス {cmd.selected_indices}"

    @classmethod
    def _format_reaction(cls, cmd: dm.DeclareReactionCommand, state: Optional[dm.GameState]) -> str:
        is_pass = getattr(cmd, "pass")
        if is_pass:
            return f"リアクション: パス (プレイヤー{cmd.player_id})"
        return f"リアクション: 使用 (インデックス{cmd.reaction_index}, プレイヤー{cmd.player_id})"

    @classmethod
    def _format_stat(cls, cmd: dm.StatCommand, state: Optional[dm.GameState]) -> str:
        stype = cmd.stat
        amount = cmd.amount
        stype_str = tr(str(stype).split('.')[-1])
        return f"統計更新: {stype_str} += {amount}"

    @classmethod
    def _format_game_result(cls, cmd: dm.GameResultCommand, state: Optional[dm.GameState]) -> str:
        return f"ゲーム終了: {tr(cmd.result)}"
