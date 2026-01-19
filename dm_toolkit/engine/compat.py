# -*- coding: utf-8 -*-
import sys
import os
from typing import Any, List, Optional, Callable, Dict, Union
from types import ModuleType

# dm_ai_module may be an optional compiled extension; annotate as Optional.
# Prefer the dm_toolkit shim module so unit tests can patch it reliably.
dm_ai_module: Optional[ModuleType] = None
try:
    from dm_toolkit import dm_ai_module as _shim_dm_ai_module  # type: ignore
    dm_ai_module = _shim_dm_ai_module
except ImportError:
    try:
        import dm_ai_module as _native_dm_ai_module  # type: ignore
        dm_ai_module = _native_dm_ai_module
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
        # Phase 1 (Specs/AGENTS.md Policy): Route action through unified execution when possible
        # Prefer action.execute first (may already encapsulate command behavior)
        try:
            if hasattr(action, 'execute') and callable(getattr(action, 'execute')):
                try:
                    try:
                        action.execute(state, card_db)
                    except TypeError:
                        action.execute(state)
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt unified conversion to Command dict and execute via EngineCompat
        try:
            from dm_toolkit.unified_execution import ensure_executable_command
            cmd = ensure_executable_command(action)
            # If conversion is inconclusive (NONE or legacy_warning), defer to native resolver
            if isinstance(cmd, dict) and (cmd.get('type') in (None, 'NONE') or cmd.get('legacy_warning')):
                raise RuntimeError('Inconclusive unified conversion; use native resolver')
            EngineCompat.ExecuteCommand(state, cmd, card_db)
            return
        except Exception:
            pass

        # Legacy fallback: call native resolver directly if available
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
        # Disabled to eliminate Action-based flows. Use dm_toolkit.commands.generate_legal_commands instead.
        raise RuntimeError("ActionGenerator_generate_legal_actions is deprecated. Use generate_legal_commands.")

    @staticmethod
    def ActionGenerator_generate_legal_commands(state: GameState, card_db: CardDB) -> List[Any]:
        """Return a list of ICommand-like objects for the given state.

        Uses the Python compatibility helper `dm_toolkit.commands.generate_legal_commands`
        to wrap engine actions into ICommand interfaces.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None
        from dm_toolkit.commands import generate_legal_commands
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
                print('EngineCompat: cmd_dict detected:', cmd_dict)
                type_str = cmd_dict.get('type')
                print('EngineCompat: type_str =', type_str)

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

                        # Map Zones (best-effort): try Zone enum, otherwise fall back to string.
                        fz = cmd_dict.get('from_zone')
                        tz = cmd_dict.get('to_zone')

                        zone_alias = {
                            # Legacy / UI strings
                            'MANA_ZONE': 'MANA',
                            'BATTLE_ZONE': 'BATTLE',
                            'SHIELD_ZONE': 'SHIELD',
                        }
                        fz_norm = zone_alias.get(str(fz), str(fz)) if fz is not None else ''
                        tz_norm = zone_alias.get(str(tz), str(tz)) if tz is not None else ''

                        # Ensure we always assign strings if CommandDef expects strings,
                        # or enums if it expects enums.
                        # Based on pybind11 errors "arg0: str", CommandDef.from_zone is a string.
                        # So we should pass the normalized string name of the zone.

                        # We use the normalized strings (MANA, HAND, etc.)
                        cmd_def.from_zone = str(fz_norm or '')
                        cmd_def.to_zone = str(tz_norm or '')

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
                        # If a target_filter is present but instance_id wasn't assigned,
                        # attempt a Python-side target resolution as a fallback.
                        ctx = {}
                        filter_dict = cmd_dict.get('target_filter') or {}
                        assigned_any = False

                        def _resolve_zone_instances(zname: str):
                            # Normalize common legacy names
                            if zname in ('BATTLE_ZONE', 'BATTLE'):
                                return getattr(state.players[player_id], 'battle_zone', [])
                            if zname in ('HAND',):
                                return getattr(state.players[player_id], 'hand', [])
                            if zname in ('MANA_ZONE', 'MANA'):
                                return getattr(state.players[player_id], 'mana_zone', [])
                            if zname in ('SHIELD_ZONE', 'SHIELD'):
                                return getattr(state.players[player_id], 'shield_zone', [])
                            if zname in ('DECK',):
                                return getattr(state.players[player_id], 'deck', [])
                            return []

                        if filter_dict and not getattr(cmd_def, 'instance_id', 0):
                            print('EngineCompat: resolving filter zones', filter_dict.get('zones'))
                            zones = filter_dict.get('zones') or []
                            instances = []
                            for z in zones:
                                print('EngineCompat: resolving zone', z)
                                for inst in _resolve_zone_instances(str(z)):
                                    print('EngineCompat: found instance', getattr(inst, 'instance_id', None))
                                    instances.append(inst)

                            # If instances found, call CommandSystem per-instance
                            if instances:
                                for inst in instances:
                                    try:
                                        cmd_def.instance_id = int(getattr(inst, 'instance_id', getattr(inst, 'id', 0) or 0))
                                        dm_ai_module.CommandSystem.execute_command(state, cmd_def, cmd_def.instance_id or source_id, player_id, ctx)
                                        assigned_any = True
                                    except Exception:
                                        pass
                                if assigned_any:
                                    print('EngineCompat: executed per-instance for command')
                                    return

                        # Default single-shot execute if no per-instance fallback
                        print('EngineCompat: calling CommandSystem.execute_command single-shot')
                        dm_ai_module.CommandSystem.execute_command(state, cmd_def, source_id, player_id, ctx)
                        print('EngineCompat: called CommandSystem.execute_command')
                        return # Success: Return only if executed by CommandSystem
            except Exception as e:
                    print('EngineCompat: exception during CommandSystem mapping', e)
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
        # Final Python-side fallback: handle a few high-priority commands by mutating state directly
        try:
            if isinstance(cmd, dict) or cmd_dict:
                cd = cmd_dict or cmd
                ctype = cd.get('type')
                player_id = getattr(state, 'active_player_id', 0)
                print('EngineCompat: Python-fallback player_id =', player_id, 'ctype=', ctype)
                # Helper to resolve instances
                def _resolve_zone_instances_local(zname: str):
                    if zname in ('BATTLE_ZONE', 'BATTLE'):
                        return getattr(state.players[player_id], 'battle_zone', [])
                    if zname in ('HAND',):
                        return getattr(state.players[player_id], 'hand', [])
                    if zname in ('MANA_ZONE', 'MANA'):
                        return getattr(state.players[player_id], 'mana_zone', [])
                    if zname in ('SHIELD_ZONE', 'SHIELD'):
                        return getattr(state.players[player_id], 'shield_zone', [])
                    if zname in ('DECK',):
                        return getattr(state.players[player_id], 'deck', [])
                    return []

                if ctype == 'REPLACE_CARD_MOVE':
                    # Python-side fallback for replacement move semantics.
                    zone_attr_map = {
                        'BATTLE': 'battle_zone',
                        'BATTLE_ZONE': 'battle_zone',
                        'HAND': 'hand',
                        'MANA': 'mana_zone',
                        'MANA_ZONE': 'mana_zone',
                        'GRAVEYARD': 'graveyard',
                        'SHIELD': 'shield_zone',
                        'SHIELD_ZONE': 'shield_zone',
                        'DECK': 'deck',
                        'DECK_BOTTOM': 'deck',
                    }

                    def _normalize_zone(z: str) -> str:
                        zstr = str(z or '').upper()
                        return zstr

                    def _get_zone_list_for(player, zone_key: str):
                        attr = zone_attr_map.get(zone_key)
                        if not attr:
                            return None
                        return getattr(player, attr, None)

                    def _detach_instance(inst_id: int):
                        players = getattr(state, 'players', [])
                        for pid, pl in enumerate(players):
                            for zkey, attr in zone_attr_map.items():
                                zone_list = getattr(pl, attr, None)
                                if not zone_list:
                                    continue
                                for obj in list(zone_list):
                                    cid = getattr(obj, 'instance_id', getattr(obj, 'id', None))
                                    if cid == inst_id:
                                        try:
                                            zone_list.remove(obj)
                                        except Exception:
                                            pass
                                        return obj, pid
                        return None, None

                    def _place(card_obj, pid: int, dest_zone: str):
                        players = getattr(state, 'players', [])
                        if pid is None or pid < 0 or pid >= len(players):
                            return False
                        zlist = _get_zone_list_for(players[pid], dest_zone)
                        if zlist is None:
                            return False
                        try:
                            # Deck bottom semantics: append
                            zlist.append(card_obj)
                            return True
                        except Exception:
                            return False

                    dest_zone = _normalize_zone(cd.get('to_zone') or 'DECK_BOTTOM')
                    original_zone = _normalize_zone(cd.get('original_to_zone') or cd.get('from_zone') or '')
                    instance_id = cd.get('instance_id') or cd.get('target_instance') or cd.get('source_instance_id')
                    try:
                        amount = int(cd.get('amount', 1) or 1)
                    except Exception:
                        amount = 1

                    moved = 0
                    if instance_id is not None:
                        card_obj, pid = _detach_instance(int(instance_id))
                        if card_obj is not None:
                            if _place(card_obj, pid, dest_zone):
                                moved += 1

                    # If we still need to move cards and a filter exists, try pulling from that zone
                    if moved < amount:
                        filter_dict = cd.get('target_filter') or {}
                        owner = cd.get('owner_id', cd.get('player_id', player_id))
                        players = getattr(state, 'players', [])
                        if 0 <= int(owner) < len(players):
                            zones = filter_dict.get('zones') or ([original_zone] if original_zone else [])
                            for z in zones:
                                z_norm = _normalize_zone(z)
                                zlist = _get_zone_list_for(players[int(owner)], z_norm)
                                if not zlist:
                                    continue
                                while zlist and moved < amount:
                                    obj = zlist.pop(0)
                                    if _place(obj, int(owner), dest_zone):
                                        moved += 1
                                    else:
                                        # If placement failed, put the card back to avoid loss
                                        try:
                                            zlist.insert(0, obj)
                                        except Exception:
                                            pass
                                        break
                                if moved >= amount:
                                    break

                    if moved > 0:
                        print(f'EngineCompat: REPLACE_CARD_MOVE moved {moved} card(s) from {original_zone or "UNKNOWN"} to {dest_zone}')
                        return

                if ctype in ('TAP', 'UNTAP', 'RETURN_TO_HAND'):
                    filter_dict = cd.get('target_filter') or {}
                    zones = filter_dict.get('zones') or []
                    targets = []
                    for z in zones:
                        for inst in _resolve_zone_instances_local(str(z)):
                            targets.append(inst)

                    if targets:
                        print('EngineCompat: Python-fallback found targets count', len(targets))
                        if ctype == 'TAP':
                            for inst in targets:
                                try:
                                    setattr(inst, 'is_tapped', True)
                                except Exception:
                                    pass
                            return
                        if ctype == 'UNTAP':
                            for inst in targets:
                                try:
                                    setattr(inst, 'is_tapped', False)
                                except Exception:
                                    pass
                            return
                        if ctype == 'RETURN_TO_HAND':
                            # Move from battle_zone to hand for matching instances
                            for inst in list(targets):
                                try:
                                    # remove from zone lists if present
                                    bz = getattr(state.players[player_id], 'battle_zone', [])
                                    if inst in bz:
                                        bz.remove(inst)
                                    hand = getattr(state.players[player_id], 'hand', [])
                                    hand.append(inst)
                                except Exception:
                                    pass
                            return

        except Exception:
            pass

        try:
            print('Warning: ExecuteCommand could not execute given command/object:', cmd)
        except Exception:
            pass

    @staticmethod
    def JsonLoader_load_cards(filepath: str) -> Optional[CardDB]:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'JsonLoader') and hasattr(dm_ai_module.JsonLoader, 'load_cards'):
            # Suppress C++ stderr warnings about JSON parsing
            import os
            import sys
            devnull = open(os.devnull, 'w')
            old_stderr = sys.stderr
            try:
                sys.stderr = devnull
                res = dm_ai_module.JsonLoader.load_cards(filepath)
            finally:
                sys.stderr = old_stderr
                devnull.close()
            if res is None:
                return None
            return res  # type: ignore
        return None

    @staticmethod
    def load_cards_robust(filepath: str) -> Dict[int, Any]:
        """
        Attempts to load cards using the native loader first, then falls back to standard JSON loading.
        Returns a dictionary mapping CardID (int) to card data.
        
        NOTE: Python fallback is preferred due to known C++ JSON parsing compatibility issues.
        The C++ loader is deprecated and will be removed in a future version.
        """
        # Fallback to pure Python json load (PREFERRED PATH)
        try:
            import json
            if not os.path.exists(filepath):
                # Try finding relative to project root if path is relative
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                alt_path = os.path.join(base_dir, filepath)
                if os.path.exists(alt_path):
                    filepath = alt_path

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert array format to dict format if needed
            if isinstance(data, list):
                card_dict = {}
                for card in data:
                    if isinstance(card, dict) and 'id' in card:
                        card_id = card['id']
                        card_dict[card_id] = card
                return card_dict
            else:
                # Already in dict format
                return data
        except Exception as e:
            print(f"Error loading cards: {e}", flush=True)

        # Fallback: Try native loader only if Python loading failed
        try:
            db = EngineCompat.JsonLoader_load_cards(filepath)
            if db:
                return db
        except Exception:
            pass

        return {}

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
