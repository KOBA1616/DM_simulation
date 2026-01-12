# -*- coding: utf-8 -*-
from typing import Any
from ..i18n import tr
from .card_helpers import get_card_name_by_instance

m: Any = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    pass

def describe_command(cmd: Any, game_state: Any, card_db: Any) -> str:
    """Generate a localized string description for a GameCommand."""
    if not m:
        return "GameCommand（ネイティブモジュール未ロード）"

    cmd_type = cmd.get_type()

    if cmd_type == m.CommandType.TRANSITION:
        # TransitionCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.card_instance_id)
        return f"[{tr('TRANSITION')}] {name} (P{c.owner_id}): {tr(c.from_zone)} -> {tr(c.to_zone)}"

    elif cmd_type == m.CommandType.MUTATE:
        # MutateCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.target_instance_id)
        mutation = tr(c.mutation_type)
        val = ""
        if c.mutation_type == m.MutationType.POWER_MOD:
            val = f"{c.int_value:+}"
        elif c.mutation_type == m.MutationType.ADD_KEYWORD:
            val = c.str_value

        return f"[{tr('MUTATE')}] {name}: {mutation} {val}".strip()

    elif cmd_type == m.CommandType.FLOW:
        # FlowCommand
        c = cmd
        flow = tr(c.flow_type)
        val = c.new_value
        if c.flow_type == m.FlowType.PHASE_CHANGE:
            # Cast int to Phase enum if possible
            try:
                val = tr(m.Phase(c.new_value))
            except:
                pass
        return f"[{tr('FLOW')}] {flow}: {val}"

    elif cmd_type == m.CommandType.QUERY:
        c = cmd
        return f"[{tr('QUERY')}] {tr(c.query_type)}"

    elif cmd_type == m.CommandType.DECIDE:
        c = cmd
        return f"[{tr('DECIDE')}] 選択肢: {c.selected_option_index}, 対象数: {len(c.selected_indices)}"

    elif cmd_type == m.CommandType.STAT:
        c = cmd
        return f"[{tr('STAT')}] {tr(c.stat)} += {c.amount}"

    elif cmd_type == m.CommandType.GAME_RESULT:
        c = cmd
        return f"[{tr('GAME_RESULT')}] {tr(c.result)}"

    return f"未対応コマンド: {cmd_type}"
