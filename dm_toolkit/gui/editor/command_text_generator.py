# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
from dm_toolkit.gui.localization import tr
import dm_ai_module as dm

class CommandTextGenerator:
    """
    Generates text descriptions for GameCommand objects.
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
                return f"Unknown Command: {cmd_type}"
        except Exception as e:
            return f"Error formatting command: {e}"

    @classmethod
    def _get_card_name(cls, instance_id: int, game_state: Optional[dm.GameState]) -> str:
        if not game_state:
            return f"Card#{instance_id}"

        try:
            instance = game_state.get_card_instance(instance_id)
            if not instance:
                return f"Card#{instance_id}"
            return f"Card<{instance.card_id}>"
        except:
            return f"Card#{instance_id}"

    @classmethod
    def _format_transition(cls, cmd: dm.TransitionCommand, state: Optional[dm.GameState]) -> str:
        # TransitionCommand handles card movement
        target = cls._get_card_name(cmd.card_instance_id, state)
        from_z = str(cmd.from_zone).split('.')[-1]
        to_z = str(cmd.to_zone).split('.')[-1]
        return f"Move {target} from {from_z} to {to_z}"

    @classmethod
    def _format_mutate(cls, cmd: dm.MutateCommand, state: Optional[dm.GameState]) -> str:
        mtype = cmd.mutation_type
        target = cls._get_card_name(cmd.target_instance_id, state)
        val = cmd.int_value
        sval = cmd.str_value

        # MutationType is module-level enum in dm_ai_module
        if mtype == dm.MutationType.TAP:
            return f"Tap {target}"
        elif mtype == dm.MutationType.UNTAP:
            return f"Untap {target}"
        elif mtype == dm.MutationType.POWER_MOD:
            sign = "+" if val >= 0 else ""
            return f"Power {sign}{val} to {target}"
        elif mtype == dm.MutationType.ADD_KEYWORD:
            return f"Grant '{tr(sval)}' to {target}"
        elif mtype == dm.MutationType.REMOVE_KEYWORD:
            return f"Remove '{tr(sval)}' from {target}"
        elif mtype == dm.MutationType.ADD_PASSIVE_EFFECT:
            return f"Add Passive to {target}"
        elif mtype == dm.MutationType.ADD_COST_MODIFIER:
            return f"Add Cost Modifier to {target}"

        return f"Mutate {mtype}: {target} ({val}/{sval})"

    @classmethod
    def _format_attach(cls, cmd: dm.AttachCommand, state: Optional[dm.GameState]) -> str:
        base = cls._get_card_name(cmd.target_base_card_id, state)
        card_id = cmd.card_to_attach_id
        return f"Attach Card<{card_id}> to {base}"

    @classmethod
    def _format_flow(cls, cmd: dm.FlowCommand, state: Optional[dm.GameState]) -> str:
        ftype = cmd.flow_type
        val = cmd.new_value

        # FlowType is module-level enum
        if ftype == dm.FlowType.PHASE_CHANGE:
            p_name = str(dm.Phase(val)).split('.')[-1]
            return f"Phase Start: {p_name}"
        elif ftype == dm.FlowType.TURN_CHANGE:
            return f"Turn Change: Player {val}"
        elif ftype == dm.FlowType.STEP_CHANGE:
            return f"Step Change: {val}"
        elif ftype == dm.FlowType.SET_ACTIVE_PLAYER:
            return f"Set Active Player: {val}"

        return f"Flow {ftype}: {val}"

    @classmethod
    def _format_query(cls, cmd: dm.QueryCommand, state: Optional[dm.GameState]) -> str:
        return f"Query: {cmd.query_type} (Targets: {len(cmd.valid_targets)})"

    @classmethod
    def _format_decide(cls, cmd: dm.DecideCommand, state: Optional[dm.GameState]) -> str:
        if cmd.selected_option_index >= 0:
            return f"Decision: Option {cmd.selected_option_index}"
        return f"Decision: Indices {cmd.selected_indices}"

    @classmethod
    def _format_reaction(cls, cmd: dm.DeclareReactionCommand, state: Optional[dm.GameState]) -> str:
        is_pass = getattr(cmd, "pass")
        if is_pass:
            return f"Reaction: Pass (Player {cmd.player_id})"
        return f"Reaction: Use Index {cmd.reaction_index} (Player {cmd.player_id})"

    @classmethod
    def _format_stat(cls, cmd: dm.StatCommand, state: Optional[dm.GameState]) -> str:
        stype = cmd.stat
        amount = cmd.amount
        stype_str = str(stype).split('.')[-1]
        return f"Stat Update: {stype_str} += {amount}"

    @classmethod
    def _format_game_result(cls, cmd: dm.GameResultCommand, state: Optional[dm.GameState]) -> str:
        return f"Game End: {cmd.result}"
