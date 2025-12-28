# -*- coding: utf-8 -*-
import sys
import os
from typing import Any, List, Optional, Callable, Dict, Union
from types import ModuleType

# dm_ai_module may be an optional compiled extension; annotate as Optional
dm_ai_module: Optional[ModuleType] = None
try:
    import dm_ai_module  # type: ignore
    dm_ai_module = dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.types import GameState, CardDB, Action, PlayerID, Tensor, NPArray

class EngineCompat:
    """
    Compatibility layer for dm_ai_module.
    Handles missing functions, renamed attributes, and robust API calls.
    """

    @staticmethod
    def is_available() -> bool:
        return dm_ai_module is not None

    @staticmethod
    def _check_module() -> None:
        if not dm_ai_module:
            raise ImportError("dm_ai_module is not loaded.")

    # -------------------------------------------------------------------------
    # GameState Attribute Wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def get_game_state_attribute(state: GameState, attr_name: str, default: Any = None) -> Any:
        """Safely retrieve an attribute from GameState, checking aliases."""
        val = getattr(state, attr_name, None)
        if val is not None:
            return val

        # Define aliases or legacy names if any
        aliases: Dict[str, List[str]] = {
            'active_player_id': ['active_player'],
            'current_phase': ['phase'],
            'waiting_for_user_input': [], # Add known aliases if any
            'pending_query': [],
            'effect_buffer': [],
            'command_history': [],
            'turn_number': []
        }

        if attr_name in aliases:
            for alias in aliases[attr_name]:
                val = getattr(state, alias, None)
                if val is not None:
                    return val

        return default

    @staticmethod
    def get_turn_number(state: GameState) -> Any:
        return EngineCompat.get_game_state_attribute(state, 'turn_number', '?')

    @staticmethod
    def get_current_phase(state: GameState) -> Any:
        return EngineCompat.get_game_state_attribute(state, 'current_phase', 'UNKNOWN')

    @staticmethod
    def get_active_player_id(state: GameState) -> int:
        return int(EngineCompat.get_game_state_attribute(state, 'active_player_id', 0))

    @staticmethod
    def is_waiting_for_user_input(state: GameState) -> bool:
        return bool(EngineCompat.get_game_state_attribute(state, 'waiting_for_user_input', False))

    @staticmethod
    def get_pending_query(state: GameState) -> Any:
        return EngineCompat.get_game_state_attribute(state, 'pending_query', None)

    @staticmethod
    def get_effect_buffer(state: GameState) -> List[Any]:
        return list(EngineCompat.get_game_state_attribute(state, 'effect_buffer', []))

    @staticmethod
    def get_command_history(state: GameState) -> List[Any]:
        return list(EngineCompat.get_game_state_attribute(state, 'command_history', []))

    @staticmethod
    def get_player(state: GameState, player_index: int) -> Any:
        players = getattr(state, 'players', None)
        if players and len(players) > player_index:
            return players[player_index]
        # Fallback for older bindings that might expose player0/player1 directly
        if player_index == 0:
            return getattr(state, 'player0', None)
        elif player_index == 1:
            return getattr(state, 'player1', None)
        return None

    # -------------------------------------------------------------------------
    # Action Object Wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def get_action_slot_index(action: Action) -> int:
        return getattr(action, 'slot_index', -1)

    @staticmethod
    def get_action_source_id(action: Action) -> int:
        return getattr(action, 'source_instance_id', -1)

    # -------------------------------------------------------------------------
    # API Call Wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def EffectResolver_resume(state: GameState, card_db: CardDB, selection: Union[int, List[int], Any]) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module.EffectResolver, 'resume'):
            dm_ai_module.EffectResolver.resume(state, card_db, selection)
        else:
            print("Warning: dm_ai_module.EffectResolver.resume not found.")

    @staticmethod
    def EffectResolver_resolve_action(state: GameState, action: Action, card_db: CardDB) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        # Prefer action.execute (which will run attached Command when present)
        try:
            if hasattr(action, 'execute') and callable(getattr(action, 'execute')):
                try:
                    # Some action.execute accept (state, db) others only (state,)
                    try:
                        action.execute(state, card_db)
                    except TypeError:
                        action.execute(state)
                    return
                except Exception:
                    # Fall through to resolver if wrapper fails
                    pass
        except Exception:
            pass
        if hasattr(dm_ai_module.EffectResolver, 'resolve_action'):
            dm_ai_module.EffectResolver.resolve_action(state, action, card_db)
        else:
            print("Warning: dm_ai_module.EffectResolver.resolve_action not found.")

    @staticmethod
    def PhaseManager_next_phase(state: GameState, card_db: CardDB) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module.PhaseManager, 'next_phase'):
            dm_ai_module.PhaseManager.next_phase(state, card_db)
        else:
            print("Warning: dm_ai_module.PhaseManager.next_phase not found.")

    @staticmethod
    def PhaseManager_start_game(state: GameState, card_db: CardDB) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module.PhaseManager, 'start_game'):
            dm_ai_module.PhaseManager.start_game(state, card_db)
        else:
            print("Warning: dm_ai_module.PhaseManager.start_game not found.")

    @staticmethod
    def ActionGenerator_generate_legal_actions(state: GameState, card_db: CardDB) -> List[Action]:
        """
        Deprecated: Prefer ActionGenerator_generate_legal_commands for new code.
        Returns raw Actions from the engine.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module.ActionGenerator, 'generate_legal_actions'):
            return list(dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db))
        return []

    @staticmethod
    def ActionGenerator_generate_legal_commands(state: GameState, card_db: CardDB) -> List[Any]:
        """Return a list of ICommand-like objects for the given state.

        Uses the Python compatibility helper `dm_toolkit.commands_new.generate_legal_commands`
        to wrap engine actions into ICommand interfaces.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None
        from dm_toolkit.commands_new import generate_legal_commands
        return generate_legal_commands(state, card_db)

    @staticmethod
    def ExecuteCommand(state: GameState, cmd: Any, card_db: CardDB = None) -> None:
        """Execute a command-like object on the provided GameState.

        Tries `dm_ai_module.CommandSystem.execute_command` if applicable,
        then fallback to other methods.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None

        # Prepare dictionary if input is a dict or has to_dict
        cmd_dict = None
        if isinstance(cmd, dict):
            cmd_dict = cmd
        elif hasattr(cmd, 'to_dict'):
            cmd_dict = cmd.to_dict()

        # 1. Try C++ CommandSystem if it's a wrapped Command with `to_dict`
        # and has a valid type for the engine.
        if cmd_dict:
            try:
                type_str = cmd_dict.get('type')

                # STRICT VALIDATION: Only proceed if type exists in C++ CommandType
                if type_str and hasattr(dm_ai_module.CommandType, type_str):
                    if hasattr(dm_ai_module, 'CommandSystem') and hasattr(dm_ai_module.CommandSystem, 'execute_command'):
                        # Map dict to CommandDef
                        cmd_def = dm_ai_module.CommandDef()
                        cmd_def.type = getattr(dm_ai_module.CommandType, type_str)

                        # Common fields
                        cmd_def.amount = int(cmd_dict.get('amount', 0))
                        cmd_def.str_param = str(cmd_dict.get('str_param', ''))
                        cmd_def.optional = bool(cmd_dict.get('optional', False))

                        # Populate instance_id / target_instance_id / owner_id
                        if 'instance_id' in cmd_dict: cmd_def.instance_id = int(cmd_dict['instance_id'])
                        if 'target_instance' in cmd_dict: cmd_def.target_instance = int(cmd_dict['target_instance'])
                        if 'owner_id' in cmd_dict: cmd_def.owner_id = int(cmd_dict['owner_id'])
                        elif 'player_id' in cmd_dict: cmd_def.owner_id = int(cmd_dict['player_id'])

                        # Map Zones: Need to convert string to Zone enum if binding requires it
                        if hasattr(dm_ai_module, 'Zone'):
                            fz = cmd_dict.get('from_zone')
                            if fz and hasattr(dm_ai_module.Zone, fz): cmd_def.from_zone = getattr(dm_ai_module.Zone, fz)
                            tz = cmd_dict.get('to_zone')
                            if tz and hasattr(dm_ai_module.Zone, tz): cmd_def.to_zone = getattr(dm_ai_module.Zone, tz)
                        else:
                            # Fallback if Zone not exposed or string allowed
                            cmd_def.from_zone = str(cmd_dict.get('from_zone', ''))
                            cmd_def.to_zone = str(cmd_dict.get('to_zone', ''))

                        cmd_def.mutation_kind = str(cmd_dict.get('mutation_kind', ''))
                        cmd_def.input_value_key = str(cmd_dict.get('input_value_key', ''))
                        cmd_def.output_value_key = str(cmd_dict.get('output_value_key', ''))

                        # Filter Mapping
                        filter_dict = cmd_dict.get('target_filter')
                        if filter_dict:
                            f = dm_ai_module.FilterDef()
                            if 'zones' in filter_dict: f.zones = filter_dict['zones']
                            if 'types' in filter_dict: f.types = filter_dict['types']

                            # Safe property assignment
                            if 'owner' in filter_dict and hasattr(f, 'owner'): f.owner = filter_dict['owner']
                            if 'count' in filter_dict and hasattr(f, 'count'): f.count = filter_dict['count']

                            cmd_def.target_filter = f

                        # Target Scope Mapping
                        scope_str = cmd_dict.get('target_group')
                        if scope_str and hasattr(dm_ai_module.TargetScope, scope_str):
                            cmd_def.target_group = getattr(dm_ai_module.TargetScope, scope_str)

                        # Execution Context
                        source_id = -1
                        player_id = state.active_player_id

                        # Use legacy action context if available
                        if hasattr(cmd, '_action'):
                            act = cmd._action
                            if hasattr(act, 'source_instance_id'):
                                source_id = act.source_instance_id
                            if hasattr(act, 'target_player') and act.target_player != 255:
                                player_id = act.target_player

                        # Phase 4.3: Ensure required fields are present for CommandSystem
                        # If cmd_def has instance_id, use it for source_id if source_id is -1
                        if source_id == -1 and cmd_def.instance_id > 0:
                            source_id = cmd_def.instance_id

                        # Execute
                        ctx = {}
                        dm_ai_module.CommandSystem.execute_command(state, cmd_def, source_id, player_id, ctx)
                        return # Success: Return only if executed by CommandSystem
            except Exception as e:
                # Fallthrough on error to try legacy path
                pass

        # Fallback Logic (Legacy Path)
        try:
            if hasattr(state, 'execute_command'):
                try:
                    state.execute_command(cmd)
                    return
                except Exception:
                    pass

            if hasattr(cmd, 'execute') and callable(getattr(cmd, 'execute')):
                try:
                    # Some execute signatures accept (state, db)
                    try:
                        cmd.execute(state, card_db)
                    except TypeError:
                        cmd.execute(state)
                    return
                except Exception:
                    pass

            # If it's an Action-like object, use EffectResolver
            if hasattr(cmd, 'type'):
                if hasattr(dm_ai_module.EffectResolver, 'resolve_action'):
                    dm_ai_module.EffectResolver.resolve_action(state, cmd, card_db)
                    return
        except Exception:
            pass

        # Last resort: no-op with warning
        try:
            print('Warning: ExecuteCommand could not execute given command/object:', cmd)
        except Exception:
            pass

    @staticmethod
    def JsonLoader_load_cards(filepath: str) -> Optional[CardDB]:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'JsonLoader') and hasattr(dm_ai_module.JsonLoader, 'load_cards'):
            res = dm_ai_module.JsonLoader.load_cards(filepath)
            if res is None:
                return None
            return res  # type: ignore
        return None

    @staticmethod
    def register_batch_inference_numpy(callback: Callable[[List[Any]], Any]) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'register_batch_inference_numpy'):
            dm_ai_module.register_batch_inference_numpy(callback)
        else:
            print("Warning: dm_ai_module.register_batch_inference_numpy not found.")

    @staticmethod
    def TensorConverter_convert_to_tensor(state: GameState, player_id: int, card_db: CardDB) -> Union[List[float], Any]:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'TensorConverter') and hasattr(dm_ai_module.TensorConverter, 'convert_to_tensor'):
             return dm_ai_module.TensorConverter.convert_to_tensor(state, player_id, card_db)
        return []

    @staticmethod
    def get_pending_effects_info(state: GameState) -> List[Any]:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'get_pending_effects_info'):
            return list(dm_ai_module.get_pending_effects_info(state))
        return []

    @staticmethod
    def create_parallel_runner(card_db: CardDB, sims: int, batch_size: int) -> Any:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'ParallelRunner'):
            return dm_ai_module.ParallelRunner(card_db, sims, batch_size)
        return None

    @staticmethod
    def ParallelRunner_play_games(runner: Any, initial_states: List[GameState], evaluator_func: Callable, temperature: float, verbose: bool, threads: int) -> List[Any]:
        if runner and hasattr(runner, 'play_games'):
            return list(runner.play_games(initial_states, evaluator_func, temperature, verbose, threads))
        raise RuntimeError("ParallelRunner invalid or play_games not found")
