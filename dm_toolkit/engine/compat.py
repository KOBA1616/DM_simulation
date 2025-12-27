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
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module.ActionGenerator, 'generate_legal_actions'):
            return list(dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db))
        return []

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
