# -*- coding: utf-8 -*-
import sys
import os
import logging
from typing import Any, List, Optional, Callable, Dict, Union, cast, Tuple
import enum
from types import ModuleType

from dm_toolkit.debug.effect_tracer import get_tracer, TraceEventType

# dm_ai_module may be an optional compiled extension; annotate as Optional.
dm_ai_module: Optional[ModuleType] = None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.dm_types import GameState, CardDB, PlayerID, Tensor, NPArray

# At runtime, prefer concrete engine types from dm_ai_module when available
try:
    if dm_ai_module is not None:
        try:
            GameState = getattr(dm_ai_module, 'GameState', GameState)
        except Exception:
            pass
except Exception:
    pass

# Module logger (default to null handler to avoid spamming test output)
logger = logging.getLogger('dm_toolkit.engine.compat')
try:
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
except Exception:
    pass

class EngineCompat:
    """
    Compatibility layer for dm_ai_module.
    Handles missing functions, renamed attributes, and robust API calls.
    """

    _native_db_cache: Optional[Any] = None
    _native_enabled: bool = True

    @staticmethod
    def is_available() -> bool:
        return dm_ai_module is not None

    @staticmethod
    def set_native_enabled(enabled: bool) -> None:
        EngineCompat._native_enabled = enabled

    @staticmethod
    def _check_module() -> None:
        if not dm_ai_module:
            raise ImportError("dm_ai_module is not loaded.")

    @staticmethod
    def get_execution_context(state: GameState) -> Dict[str, Any]:
        """Wraps dm_ai_module.get_execution_context"""
        if dm_ai_module and hasattr(dm_ai_module, 'get_execution_context'):
            try:
                return cast(Dict[str, Any], dm_ai_module.get_execution_context(state))
            except Exception:
                pass
        # Fallback if state has it directly (Python stub)
        if hasattr(state, 'execution_context'):
             # Check if it's an object with .variables or a dict
             ctx = getattr(state, 'execution_context')
             if hasattr(ctx, 'variables'):
                 return cast(Dict[str, Any], ctx.variables)
             if isinstance(ctx, dict):
                 return cast(Dict[str, Any], ctx)
        return {}

    @staticmethod
    def get_command_details(cmd: Any) -> str:
        """Wraps dm_ai_module.get_command_details"""
        if dm_ai_module and hasattr(dm_ai_module, 'get_command_details'):
            try:
                return cast(str, dm_ai_module.get_command_details(cmd))
            except Exception:
                pass
        return str(cmd)

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
        val = EngineCompat.get_game_state_attribute(state, 'current_phase', None)
        # Normalize numeric/native enum to the Python-side Phase enum when available
        try:
            if dm_ai_module is not None and hasattr(dm_ai_module, 'Phase'):
                PhaseEnum = dm_ai_module.Phase
                phase_enum_is_real = isinstance(PhaseEnum, enum.EnumMeta)

                # If this GameState wraps a native backing, prefer the native
                # object's `current_phase` as the authoritative source. Only
                # attempt to convert to a Phase enum when the `Phase` exposed
                # by the module is a real Enum type; otherwise preserve raw
                # string/int values to maintain test expectations.
                try:
                    native_obj = getattr(state, '_native', None)
                    if native_obj is not None:
                        raw_native = getattr(native_obj, 'current_phase', None)
                        if raw_native is not None and phase_enum_is_real:
                            # Convert native representation to PhaseEnum when possible
                            try:
                                if isinstance(raw_native, PhaseEnum):
                                    return raw_native
                            except Exception:
                                pass
                            try:
                                if hasattr(raw_native, 'value'):
                                    return PhaseEnum(int(getattr(raw_native, 'value')))
                            except Exception:
                                pass
                            try:
                                if isinstance(raw_native, str):
                                    s = raw_native
                                    name = s.split('.', 1)[1] if s.startswith('Phase.') else s
                                    name = name.strip()
                                    if hasattr(PhaseEnum, name):
                                        return getattr(PhaseEnum, name)
                            except Exception:
                                pass
                            try:
                                return PhaseEnum(int(raw_native))
                            except Exception:
                                pass

                except Exception:
                    pass

                # Only attempt enum conversions when PhaseEnum is a real Enum
                try:
                    if phase_enum_is_real:
                        # Already the enum
                        try:
                            if isinstance(val, PhaseEnum):
                                return val
                        except Exception:
                            pass
                        # If original value is a string, preserve it (tests expect textual alias)
                        try:
                            if isinstance(val, str):
                                return val
                        except Exception:
                            pass

                        # If value has .value (enum-like), try to convert
                        try:
                            if hasattr(val, 'value'):
                                return PhaseEnum(int(getattr(val, 'value')))
                        except Exception:
                            pass

                        # Integer-like -> enum
                        try:
                            if val is None:
                                # nothing to coerce
                                pass
                            else:
                                ival = int(val)
                                return PhaseEnum(ival)
                        except Exception:
                            pass

                        # String-like -> enum (accept 'Phase.MAIN' or 'MAIN')
                        try:
                            if isinstance(val, str):
                                s = val
                                name = s.split('.', 1)[1] if s.startswith('Phase.') else s
                                name = name.strip()
                                if hasattr(PhaseEnum, name):
                                    return getattr(PhaseEnum, name)
                        except Exception:
                            pass
                except Exception:
                    pass

        except Exception:
            pass

        # If nothing could be determined, return a conservative textual fallback
        if val is None:
            return 'UNKNOWN'
        return val

    @staticmethod
    def dump_state_debug(state: GameState, max_samples: int = 3) -> Dict[str, Any]:
        """Produce a compact, serializable debug snapshot of GameState.

        Returns a dict with basic attrs, native presence, and per-player zone
        counts and a small sample (type + truncated repr) for quick diagnosis.
        """
        out: Dict[str, Any] = {}
        try:
            native_obj = getattr(state, '_native', None)
            out['native_present'] = native_obj is not None
        except Exception:
            out['native_present'] = False

        # Heuristic: some bindings expose pybind internals or do not set `_native`.
        # If we didn't detect `_native`, inspect common alternatives:
        try:
            if not out.get('native_present') and dm_ai_module is not None:
                # 1) Attributes with names like '_pybind' likely indicate pybind-wrapping
                for a in dir(state):
                    try:
                        if a.startswith('_pybind') or a.startswith('_conduit'):
                            out['native_present'] = True
                            break
                    except Exception:
                        continue

            # 2) Inspect player zones for native types coming from the extension module
            if not out.get('native_present') and hasattr(state, 'players') and dm_ai_module is not None:
                try:
                    modname = getattr(dm_ai_module, '__name__', None)
                    pl = getattr(state, 'players', [])
                    for p in pl:
                        if out.get('native_present'):
                            break
                        for zone_name in ('deck', 'hand', 'mana_zone', 'shield_zone'):
                            try:
                                zone = getattr(p, zone_name, None)
                                if zone is None:
                                    continue
                                seq = list(zone)
                                for c in seq[:1]:
                                    t = type(c)
                                    try:
                                        # If the object's type's module matches the native module, it's native
                                        if modname and getattr(t, '__module__', '') == modname:
                                            out['native_present'] = True
                                            break
                                    except Exception:
                                        pass
                                if out.get('native_present'):
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
        except Exception:
            pass

        try:
            out['active_player_id'] = EngineCompat.get_active_player_id(state)
        except Exception:
            out['active_player_id'] = None

        try:
            out['current_phase'] = str(EngineCompat.get_current_phase(state))
        except Exception:
            out['current_phase'] = None

        try:
            pe_info = EngineCompat.get_pending_effects_info(state)
            out['pending_effects_count'] = len(pe_info)
            if pe_info:
                # Add brief summary of first few
                out['pending_effects_summary'] = [str(x[0]) for x in pe_info[:3]]
        except Exception:
            out['pending_effects_count'] = 0

        players = []
        try:
            plst = getattr(state, 'players', None) or []
            for p in plst:
                pdata: Dict[str, Any] = {}
                try:
                    pdata['deck_count'] = len(getattr(p, 'deck', []))
                except Exception:
                    pdata['deck_count'] = None
                try:
                    pdata['hand_count'] = len(getattr(p, 'hand', []))
                except Exception:
                    pdata['hand_count'] = None
                try:
                    pdata['mana_count'] = len(getattr(p, 'mana_zone', [])) if hasattr(p, 'mana_zone') else None
                except Exception:
                    pdata['mana_count'] = None
                try:
                    pdata['shield_count'] = len(getattr(p, 'shield_zone', []))
                except Exception:
                    pdata['shield_count'] = None

                def _sample(zone_obj: Any) -> List[Dict[str, Any]]:
                    samples: List[Dict[str, Any]] = []
                    try:
                        seq = list(zone_obj)
                    except Exception:
                        try:
                            seq = list(getattr(zone_obj, '__iter__', lambda: [])())
                        except Exception:
                            seq = []
                    for i, c in enumerate(seq[:max_samples]):
                        try:
                            tname = type(c).__name__
                            r = repr(c)
                            if len(r) > 200:
                                r = r[:200] + '...'
                            samples.append({'idx': i, 'type': tname, 'repr': r})
                        except Exception:
                            samples.append({'idx': i, 'type': 'ERROR', 'repr': ''})
                    return samples

                try:
                    pdata['deck_samples'] = _sample(getattr(p, 'deck', []))
                except Exception:
                    pdata['deck_samples'] = []
                try:
                    pdata['hand_samples'] = _sample(getattr(p, 'hand', []))
                except Exception:
                    pdata['hand_samples'] = []
                try:
                    pdata['mana_samples'] = _sample(getattr(p, 'mana_zone', [])) if hasattr(p, 'mana_zone') else []
                except Exception:
                    pdata['mana_samples'] = []
                try:
                    pdata['shield_samples'] = _sample(getattr(p, 'shield_zone', []))
                except Exception:
                    pdata['shield_samples'] = []

                players.append(pdata)
        except Exception:
            pass

        out['players'] = players
        return out

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
        try:
            players = getattr(state, 'players', None)
            if players is not None:
                # Check for length if it's a list/sequence
                if hasattr(players, '__len__'):
                    if player_index < len(players):
                        return players[player_index]
                # Some native bindings might support indexing but not len() directly via python protocols efficiently?
                # But typically list-like bindings support len.
                # Try direct indexing as fallback
                try:
                    return players[player_index]
                except (IndexError, TypeError):
                    pass

            # Fallback for older bindings that might expose player0/player1 directly
            if player_index == 0:
                return getattr(state, 'player0', None)
            elif player_index == 1:
                return getattr(state, 'player1', None)
        except Exception:
            pass
        return None

    # -------------------------------------------------------------------------
    # Action Object Wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def get_action_slot_index(action: Any) -> int:
        return getattr(action, 'slot_index', -1)

    @staticmethod
    def get_action_source_id(action: Any) -> int:
        return getattr(action, 'source_instance_id', -1)

    # -------------------------------------------------------------------------
    # API Call Wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def EffectResolver_resume(state: GameState, card_db: CardDB, selection: Union[int, List[int], Any]) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module.EffectResolver, 'resume'):
            dm_ai_module.EffectResolver.resume(state, real_db, selection)
        else:
            logger.warning("dm_ai_module.EffectResolver.resume not found.")

    @staticmethod
    def EffectResolver_resolve_action(state: GameState, action: Any, card_db: CardDB) -> None:
        # Improved logging with structured data
        log_data: Dict[str, Any] = {"action_str": str(action)}
        try:
            from dm_toolkit.action_to_command import map_action
            log_data["command_map"] = map_action(action)
        except Exception:
            pass
        get_tracer().log_event(TraceEventType.EFFECT_RESOLUTION, "Resolving Action", log_data)

        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        # Phase 1 (Specs/AGENTS.md Policy): Route action through unified execution when possible
        # Prefer action.execute first (may already encapsulate command behavior)
        try:
            if hasattr(action, 'execute') and callable(getattr(action, 'execute')):
                try:
                    try:
                        action.execute(state, real_db)
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
        # REMOVED: GenericCardSystem (EffectResolver.resolve_action) is incomplete.
        # All actions must be convertible to commands via ensure_executable_command.
        logger.warning("EffectResolver_resolve_action: Action could not be converted to Command: %s", action)

    @staticmethod
    def PhaseManager_next_phase(state: GameState, card_db: CardDB) -> None:
        get_tracer().log_event(TraceEventType.STATE_CHANGE, "PhaseManager.next_phase", {"current_phase": str(getattr(state, 'current_phase', 'UNKNOWN'))})
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module.PhaseManager, 'next_phase'):
            try:
                # Normalize `state.current_phase` to native Phase enum when possible
                try:
                    if dm_ai_module is not None and hasattr(dm_ai_module, 'Phase'):
                        PhaseEnum = dm_ai_module.Phase
                        cur_val = getattr(state, 'current_phase', None)
                        try:
                            # If it's already an enum of the native type, leave it
                            if not (isinstance(cur_val, PhaseEnum)):
                                # If cur_val is string like 'Phase.MAIN' or 'MAIN', map to enum
                                if isinstance(cur_val, str):
                                    name = cur_val.split('.', 1)[1] if cur_val.startswith('Phase.') else cur_val
                                    name = name.strip()
                                    if hasattr(PhaseEnum, name):
                                        try:
                                            setattr(state, 'current_phase', getattr(PhaseEnum, name))
                                        except Exception:
                                            pass
                                else:
                                    # Try int coercion
                                    try:
                                        if cur_val is not None:
                                            setattr(state, 'current_phase', PhaseEnum(int(cur_val)))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except RuntimeError:
                    raise
                except Exception:
                    pass

                # Defensive: detect if phase does not advance to avoid infinite loops
                try:
                    before = EngineCompat.get_current_phase(state)
                except Exception:
                    before = None

                dm_ai_module.PhaseManager.next_phase(state, real_db)

                try:
                    after = EngineCompat.get_current_phase(state)
                except Exception:
                    after = None

                # If native call made no observable change, retry a few times then force progress
                try:
                    def _phase_name(x: Any) -> str:
                        try:
                            return str(x)
                        except Exception:
                            try:
                                return cast(str, getattr(x, 'name', repr(x)))
                            except Exception:
                                return repr(x)

                    if _phase_name(before) == _phase_name(after):
                        # Try a small number of retries in case native has transient issues
                        retried = False
                        for _ in range(3):
                            try:
                                dm_ai_module.PhaseManager.next_phase(state, real_db)
                            except Exception:
                                pass
                            try:
                                after = EngineCompat.get_current_phase(state)
                            except Exception:
                                after = None
                            if _phase_name(before) != _phase_name(after):
                                retried = True
                                break

                        if not retried:
                            # As a last resort, compute and assign the next phase ourselves to break loops.
                            try:
                                if dm_ai_module is not None and hasattr(dm_ai_module, 'Phase'):
                                    PhaseEnum = dm_ai_module.Phase
                                    # helper to get int value from various representations
                                    def _to_int(x: Any) -> Optional[int]:
                                        try:
                                            if x is None:
                                                return None
                                            if isinstance(x, PhaseEnum):
                                                return int(x)
                                            if isinstance(x, str):
                                                s = x.split('.', 1)[1] if x.startswith('Phase.') else x
                                                s = s.strip()
                                                if hasattr(PhaseEnum, s):
                                                    return int(getattr(PhaseEnum, s))
                                            return int(x)
                                        except Exception:
                                            return None

                                    before_int = _to_int(before)
                                    if before_int is not None:
                                        # Use robust cycling based on defined enum values
                                        sorted_values = sorted([p.value for p in PhaseEnum])
                                        if sorted_values:
                                            # Heuristic for standard DM phases: END(5) -> MANA(2)
                                            if before_int == 5 and 2 in sorted_values:
                                                forced_next = PhaseEnum(2)
                                            elif before_int in sorted_values:
                                                curr_idx = sorted_values.index(before_int)
                                                next_val = sorted_values[(curr_idx + 1) % len(sorted_values)]
                                                forced_next = PhaseEnum(next_val)
                                            else:
                                                # Fallback: go to first defined phase
                                                forced_next = PhaseEnum(sorted_values[0])

                                            try:
                                                setattr(state, 'current_phase', forced_next)
                                            except Exception:
                                                pass
                                        try:
                                            native_obj = getattr(state, '_native', None)
                                            if native_obj is not None:
                                                try:
                                                    setattr(native_obj, 'current_phase', forced_next)
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                        # update after view
                                        try:
                                            after = EngineCompat.get_current_phase(state)
                                        except Exception:
                                            after = None
                                        try:
                                            logger.warning("EngineCompat: forced phase advance %s -> %s", before, forced_next)
                                        except Exception:
                                            pass
                                else:
                                    # Fallback for integer phases when Phase enum is missing (e.g. Python stub)
                                    try:
                                        before_val = int(before) if before is not None else 0
                                        # Heuristic for standard DM phases 2(MANA)->3(MAIN)->4(ATTACK)->5(END)->2
                                        if before_val >= 2 and before_val < 5:
                                            forced_next = before_val + 1
                                        elif before_val == 5:
                                            forced_next = 2
                                        else:
                                            # Blind increment if unknown range
                                            forced_next = before_val + 1

                                        setattr(state, 'current_phase', forced_next)

                                        # Sync native if present (rare case where native present but no Phase enum)
                                        try:
                                            native_obj = getattr(state, '_native', None)
                                            if native_obj is not None:
                                                setattr(native_obj, 'current_phase', forced_next)
                                        except Exception:
                                            pass

                                        after = EngineCompat.get_current_phase(state)
                                        logger.warning("EngineCompat: forced phase advance (int) %s -> %s", before, forced_next)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except RuntimeError:
                    raise
                except Exception:
                    pass

                # Attempt to synchronize native backing if present
                try:
                    if dm_ai_module is not None and hasattr(dm_ai_module, 'Phase'):
                        PhaseEnum = dm_ai_module.Phase
                        # Read raw assigned values
                        raw_state_val = getattr(state, 'current_phase', None)
                        native_obj = getattr(state, '_native', None)
                        raw_native_val = getattr(native_obj, 'current_phase', None) if native_obj is not None else None

                        def _to_phase(x: Any) -> Optional[Any]:
                            try:
                                if x is None:
                                    return None
                                if isinstance(x, PhaseEnum):
                                    return x
                                if hasattr(x, 'value'):
                                    return cast(Any, PhaseEnum(int(getattr(x, 'value'))))
                                if isinstance(x, str):
                                    nm = x.split('.', 1)[1] if x.startswith('Phase.') else x
                                    nm = nm.strip()
                                    if hasattr(PhaseEnum, nm):
                                        return cast(Any, getattr(PhaseEnum, nm))
                                return cast(Any, PhaseEnum(int(x)))
                            except Exception:
                                return None

                        p_state = _to_phase(raw_state_val)
                        p_native = _to_phase(raw_native_val)

                        # Prefer native value if available
                        chosen = p_native or p_state
                        if chosen is not None:
                            try:
                                # set both representations to the chosen enum where possible
                                try:
                                    setattr(state, 'current_phase', chosen)
                                except Exception:
                                    pass
                                if native_obj is not None:
                                    try:
                                        setattr(native_obj, 'current_phase', chosen)
                                    except Exception:
                                        pass
                                # Update 'after' textual view for downstream checks
                                after = EngineCompat.get_current_phase(state)
                            except Exception:
                                pass
                        else:
                            # If neither converted, emit diagnostics to stderr for debugging
                            try:
                                logger.debug("EngineCompat: PhaseManager_next_phase: could not normalize phases (state=%s, native=%s)", raw_state_val, raw_native_val)
                            except Exception:
                                pass
                except Exception:
                    pass

                # If phase didn't change (compare normalized names), increment per-state guard counter and raise after threshold
                def _phase_name(x: Any) -> str:
                    try:
                        return str(x)
                    except Exception:
                        try:
                            return cast(str, getattr(x, 'name', repr(x)))
                        except Exception:
                            return repr(x)

                if _phase_name(before) == _phase_name(after):
                    # attach counter to state object for persistence across calls
                    cnt = getattr(state, '_phase_nochange_count', 0) or 0
                    cnt += 1
                    try:
                        setattr(state, '_phase_nochange_count', cnt)
                    except Exception:
                        pass

                    # Emit a concise diagnostic every 5 occurrences to avoid spam
                    try:
                        reported = getattr(state, '_phase_nochange_reported', 0) or 0
                        if cnt % 5 == 0 and reported < cnt:
                            try:
                                native_obj = getattr(state, '_native', None)
                                logger.debug("EngineCompat: phase no-change detected count=%s before=%s native_present=%s", cnt, before, (native_obj is not None))
                            except Exception:
                                pass
                            try:
                                setattr(state, '_phase_nochange_reported', cnt)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Fail fast earlier to break test hang and gather stack trace
                    # Reduced threshold from 15 to 5 for faster debugging
                    if cnt > 5:
                        raise RuntimeError(f"PhaseManager.next_phase: phase did not advance after {cnt} attempts (before={before})")
                else:
                    # reset counter on progress
                    try:
                        if hasattr(state, '_phase_nochange_count'):
                            setattr(state, '_phase_nochange_count', 0)
                    except Exception:
                        pass
            except RuntimeError:
                raise
        else:
            logger.warning("dm_ai_module.PhaseManager.next_phase not found.")

    @staticmethod
    def PhaseManager_start_game(state: GameState, card_db: CardDB) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module.PhaseManager, 'start_game'):
            dm_ai_module.PhaseManager.start_game(state, real_db)
        else:
            logger.warning("dm_ai_module.PhaseManager.start_game not found.")

    @staticmethod
    def PhaseManager_check_game_over(state: GameState) -> Tuple[bool, Optional[Any]]:
        """
        Safe wrapper around dm_ai_module.PhaseManager.check_game_over.

        Returns a tuple (is_over: bool, result: Optional[GameResult/Object]).
        The underlying binding has varied signatures across versions (one-arg
        returning bool/tuple, or two-arg where a GameResult is provided by
        reference). This wrapper tries the common forms and normalizes the
        output to (bool, result_obj).
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None
        try:
            pm = getattr(dm_ai_module, 'PhaseManager', None)
            if pm is None or not hasattr(pm, 'check_game_over'):
                return False, None

            # Prefer two-arg form when GameResult type is available
            if hasattr(dm_ai_module, 'GameResult'):
                try:
                    gr = dm_ai_module.GameResult()
                    res = dm_ai_module.PhaseManager.check_game_over(state, gr)
                    # Interpret result
                    if isinstance(res, tuple) and len(res) == 2:
                        return res
                    if isinstance(res, bool):
                        return res, gr
                    # Some bindings mutate gr and return None/other; inspect gr
                    try:
                        is_over = getattr(gr, 'is_over', None)
                        if is_over is None:
                            is_over = getattr(gr, 'result', None)
                        if isinstance(is_over, bool):
                            return is_over, gr
                    except Exception:
                        pass
                    return bool(res), gr
                except TypeError:
                    # Binding likely expects single-arg form
                    res2 = dm_ai_module.PhaseManager.check_game_over(state)
                    if isinstance(res2, tuple) and len(res2) == 2:
                        return res2
                    return bool(res2), None
            else:
                # No GameResult type exposed; try single-arg call
                res = dm_ai_module.PhaseManager.check_game_over(state)
                if isinstance(res, tuple) and len(res) == 2:
                    return res
                return bool(res), None
        except Exception:
            return False, None

    @staticmethod
    def ActionGenerator_generate_legal_commands(state: GameState, card_db: CardDB) -> List[Any]:
        """
        Deprecated: Prefer ActionGenerator_generate_legal_commands for new code.
        Returns raw Actions from the engine.
        """
        # Disabled to eliminate Action-based flows. Use dm_toolkit.commands.generate_legal_commands instead.
        raise RuntimeError("ActionGenerator_generate_legal_commands is deprecated. Use generate_legal_commands.")

    @staticmethod
    def ActionGenerator_generate_legal_commands(state: GameState, card_db: CardDB) -> List[Any]:
        """Return a list of ICommand-like objects for the given state.

        Uses the Python compatibility helper `dm_toolkit.commands.generate_legal_commands`
        to wrap engine actions into ICommand interfaces.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        from dm_toolkit import commands_v2 as commands
        return commands.generate_legal_commands(state, real_db)

    @staticmethod
    def ExecuteCommand(state: GameState, cmd: Any, card_db: Optional[Any] = None) -> None:
        """Execute a command-like object on the provided GameState.

        Tries `dm_ai_module.CommandSystem.execute_command` if applicable,
        then fallback to other methods.
        """
        EngineCompat._check_module()
        assert dm_ai_module is not None

        # Phase 3: Direct CommandDef execution
        if dm_ai_module and hasattr(dm_ai_module, 'CommandDef') and isinstance(cmd, dm_ai_module.CommandDef):
            try:
                if hasattr(dm_ai_module, 'CommandSystem') and hasattr(dm_ai_module.CommandSystem, 'execute_command'):
                    src = getattr(cmd, 'instance_id', -1)
                    pid = getattr(cmd, 'owner_id', getattr(state, 'active_player_id', 0))

                    # Call directly with the object
                    dm_ai_module.CommandSystem.execute_command(state, cmd, src, pid, {})
                    return
            except Exception as e:
                logger.exception(f'EngineCompat: exception during direct CommandDef execution: {e}')
                # Fallthrough to legacy path if direct execution fails
                pass

        # Prepare dictionary if input is a dict or has to_dict
        cmd_dict = None
        if isinstance(cmd, dict):
            cmd_dict = cmd
        elif hasattr(cmd, 'to_dict'):
            cmd_dict = cmd.to_dict()

        get_tracer().log_command(cmd_dict if cmd_dict else {"raw_cmd": str(cmd)})

        # 1. Try C++ CommandSystem if it's a wrapped Command with `to_dict`
        # and has a valid type for the engine.
        if cmd_dict:
            # Fast-path: if CommandSystem is available on the shim, call it
            # directly with the dict to allow Python-side CommandSystem stubs
            # to execute without going through the full mapping logic.
            try:
                if dm_ai_module is not None and hasattr(dm_ai_module, 'CommandSystem') and hasattr(dm_ai_module.CommandSystem, 'execute_command'):
                    try:
                        src = cmd_dict.get('instance_id', cmd_dict.get('source_instance_id', -1))
                        pid = cmd_dict.get('player_id', getattr(state, 'active_player_id', 0))
                        dm_ai_module.CommandSystem.execute_command(state, cmd_dict, src, pid, {})
                        return
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                logger.debug('EngineCompat: cmd_dict detected: %s', cmd_dict)
                type_val = cmd_dict.get('type')
                logger.debug('EngineCompat: type = %s', type_val)

                # Try to resolve CommandType from flexible inputs (name, numeric, enum)
                cmd_type_obj = None
                try:
                    # Try EngineCommandType first (for game commands)
                    if hasattr(dm_ai_module, 'EngineCommandType'):
                        # If it's already an enum-like object, try to use directly
                        if isinstance(type_val, (int,)):
                            try:
                                cmd_type_obj = dm_ai_module.EngineCommandType(int(type_val))
                            except Exception:
                                cmd_type_obj = None
                        elif isinstance(type_val, str):
                            # Numeric string?
                            try:
                                if type_val.isdigit():
                                    cmd_type_obj = dm_ai_module.EngineCommandType(int(type_val))
                            except Exception:
                                pass
                            if cmd_type_obj is None and hasattr(dm_ai_module.EngineCommandType, type_val):
                                cmd_type_obj = getattr(dm_ai_module.EngineCommandType, type_val)
                        else:
                            # Try direct assignment for enum-like objects
                            try:
                                first_attr = next(attr for attr in dir(dm_ai_module.EngineCommandType) if not attr.startswith('_'))
                                if isinstance(type_val, type(getattr(dm_ai_module.EngineCommandType, first_attr))):
                                    cmd_type_obj = type_val
                            except:
                                pass
                    # Fallback to legacy CommandType if EngineCommandType not found
                    if cmd_type_obj is None and hasattr(dm_ai_module, 'CommandType'):
                        # If it's already an enum-like object, try to use directly
                        if isinstance(type_val, (int,)):
                            try:
                                cmd_type_obj = dm_ai_module.CommandType(int(type_val))
                            except Exception:
                                cmd_type_obj = None
                        elif isinstance(type_val, str):
                            # Numeric string?
                            try:
                                if type_val.isdigit():
                                    cmd_type_obj = dm_ai_module.CommandType(int(type_val))
                            except Exception:
                                pass
                            if cmd_type_obj is None and hasattr(dm_ai_module.CommandType, type_val):
                                cmd_type_obj = getattr(dm_ai_module.CommandType, type_val)
                        else:
                            # Try direct assignment for enum-like objects
                            try:
                                first_attr = next(attr for attr in dir(dm_ai_module.CommandType) if not attr.startswith('_'))
                                if isinstance(type_val, type(getattr(dm_ai_module.CommandType, first_attr))):
                                    cmd_type_obj = type_val
                            except:
                                pass
                except Exception:
                    cmd_type_obj = None

                # Only proceed to CommandSystem mapping if we have a type object and command system available
                logger.debug(f'EngineCompat: cmd_type_obj={cmd_type_obj}, has_CommandSystem={hasattr(dm_ai_module, "CommandSystem")}')
                if cmd_type_obj is not None and hasattr(dm_ai_module, 'CommandSystem') and hasattr(dm_ai_module.CommandSystem, 'execute_command'):
                    logger.debug('EngineCompat: Entering CommandSystem mapping')
                    # Map dict to CommandDef (create instance)
                    cmd_def = dm_ai_module.CommandDef()
                    try:
                        cmd_def.type = cmd_type_obj
                    except Exception:
                        # Fallback: set raw numeric or string when enum assignment fails
                        try:
                            setattr(cmd_def, 'type', int(getattr(cmd_type_obj, 'value', cmd_type_obj)))
                        except Exception:
                            try:
                                setattr(cmd_def, 'type', str(cmd_type_obj))
                            except Exception:
                                pass

                    # Common fields (assign if present on CommandDef)
                    def _assign_if_exists(o: Any, name: str, value: Any) -> bool:
                        try:
                            if hasattr(o, name):
                                setattr(o, name, value)
                                return True
                        except Exception:
                            pass
                        return False

                    _assign_if_exists(cmd_def, 'amount', int(cmd_dict.get('amount', 0)))
                    _assign_if_exists(cmd_def, 'str_param', str(cmd_dict.get('str_param', '')))
                    _assign_if_exists(cmd_def, 'optional', bool(cmd_dict.get('optional', False)))
                    _assign_if_exists(cmd_def, 'up_to', bool(cmd_dict.get('up_to', False)))

                    # Populate instance id from several possible keys
                    for key in ('instance_id', 'source_instance_id', 'source_id', 'source'):
                        if key in cmd_dict:
                            try:
                                _assign_if_exists(cmd_def, 'instance_id', int(cmd_dict[key]))
                            except Exception:
                                _assign_if_exists(cmd_def, 'instance_id', cmd_dict[key])
                            break

                    # target instance aliases
                    for key in ('target_instance', 'target_instance_id', 'target', 'target_id'):
                        if key in cmd_dict:
                            try:
                                _assign_if_exists(cmd_def, 'target_instance', int(cmd_dict[key]))
                            except Exception:
                                _assign_if_exists(cmd_def, 'target_instance', cmd_dict[key])
                            break

                    # owner/player id aliases
                    for key in ('owner_id', 'player_id', 'owner'):
                        if key in cmd_dict:
                            try:
                                _assign_if_exists(cmd_def, 'owner_id', int(cmd_dict[key]))
                            except Exception:
                                _assign_if_exists(cmd_def, 'owner_id', cmd_dict[key])
                            break

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
                    # Guard access to TargetScope: may be missing in stubbed/native fallback
                    _TargetScope = getattr(dm_ai_module, 'TargetScope', None)
                    if scope_str and _TargetScope is not None and hasattr(_TargetScope, scope_str):
                        try:
                            cmd_def.target_group = getattr(_TargetScope, scope_str)
                        except Exception:
                            # If assignment fails, fall back to string label
                            cmd_def.target_group = str(scope_str)
                    elif scope_str:
                        # No TargetScope enum available: assign normalized string
                        cmd_def.target_group = str(scope_str)

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
                    ctx: Dict[str, Any] = {}
                    filter_dict = cmd_dict.get('target_filter') or {}
                    assigned_any = False

                    def _resolve_zone_instances(zname: str) -> List[Any]:
                        # Normalize common legacy names
                        if zname in ('BATTLE_ZONE', 'BATTLE'):
                            return cast(List[Any], getattr(state.players[player_id], 'battle_zone', []))
                        if zname in ('HAND',):
                            return cast(List[Any], getattr(state.players[player_id], 'hand', []))
                        if zname in ('MANA_ZONE', 'MANA'):
                            return cast(List[Any], getattr(state.players[player_id], 'mana_zone', []))
                        if zname in ('SHIELD_ZONE', 'SHIELD'):
                            return cast(List[Any], getattr(state.players[player_id], 'shield_zone', []))
                        if zname in ('DECK',):
                            return cast(List[Any], getattr(state.players[player_id], 'deck', []))
                        return []

                    if filter_dict and not getattr(cmd_def, 'instance_id', 0):
                        logger.debug('EngineCompat: resolving filter zones %s', filter_dict.get('zones'))
                        zones = filter_dict.get('zones') or []
                        instances = []
                        for z in zones:
                            logger.debug('EngineCompat: resolving zone %s', z)
                            for inst in _resolve_zone_instances(str(z)):
                                logger.debug('EngineCompat: found instance %s', getattr(inst, 'instance_id', None))
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
                                logger.debug('EngineCompat: executed per-instance for command')
                                return

                    # Default single-shot execute if no per-instance fallback
                    logger.debug(f'EngineCompat: calling CommandSystem.execute_command single-shot, cmd_def.type={getattr(cmd_def, "type", None)}, source_id={source_id}, player_id={player_id}')
                    dm_ai_module.CommandSystem.execute_command(state, cmd_def, source_id, player_id, ctx)
                    logger.debug('EngineCompat: called CommandSystem.execute_command successfully')
                    return # Success: Return only if executed by CommandSystem
            except Exception as e:
                logger.exception(f'EngineCompat: exception during CommandSystem mapping: {e}')
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
                        cmd.execute(state, EngineCompat._resolve_db(card_db))
                    except TypeError:
                        cmd.execute(state)
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # Last resort: no-op with warning. Emit detailed debug/exception info
        try:
            # Try to include any prepared dict representation for easier debugging
            try:
                logger.warning('ExecuteCommand could not execute given command/object; cmd=%s cmd_dict=%s', cmd, cmd_dict)
            except Exception:
                logger.warning('ExecuteCommand could not execute given command/object: %s', cmd)

            # Also log a full exception stacktrace and rich diagnostic info
            # (there may be prior logger.exception calls in the mapping path).
            try:
                import traceback as _traceback

                # Primary: recent exception text (may be 'NoneType: None' if no recent exception)
                try:
                    tb = _traceback.format_exc()
                    if tb and tb.strip() and 'NoneType' not in tb:
                        logger.debug('ExecuteCommand traceback:\n%s', tb)
                    else:
                        logger.debug('ExecuteCommand: no recent exception traceback available')
                except Exception:
                    try:
                        logger.debug('ExecuteCommand: traceback.format_exc() failed')
                    except Exception:
                        pass

                # Dump prepared command dictionary types and truncated reprs to aid diagnosis
                try:
                    if cmd_dict:
                        types_summary = {k: type(v).__name__ for k, v in cmd_dict.items()}
                        repr_summary = {k: (repr(v)[:200] + '...' if len(repr(v)) > 200 else repr(v)) for k, v in cmd_dict.items()}
                        logger.debug('ExecuteCommand: cmd_dict types=%s', types_summary)
                        logger.debug('ExecuteCommand: cmd_dict reprs=%s', repr_summary)
                    else:
                        try:
                            logger.debug('ExecuteCommand: no cmd_dict prepared; raw cmd=%s', repr(cmd))
                        except Exception:
                            logger.debug('ExecuteCommand: no cmd_dict and raw cmd repr failed')
                except Exception:
                    try:
                        logger.debug('ExecuteCommand: failed to dump cmd_dict details')
                    except Exception:
                        pass

                # If a native CommandDef was constructed, dump its common attrs
                try:
                    if 'cmd_def' in locals() and cmd_def is not None:
                        try:
                            attrs = {}
                            for a in ('type', 'instance_id', 'from_zone', 'to_zone', 'target_group', 'owner_id'):
                                if hasattr(cmd_def, a):
                                    try:
                                        attrs[a] = getattr(cmd_def, a)
                                    except Exception:
                                        attrs[a] = '<unreadable>'
                            logger.debug('ExecuteCommand: cmd_def attrs=%s', attrs)
                        except Exception:
                            logger.debug('ExecuteCommand: failed to inspect cmd_def')
                except Exception:
                    pass

                # Snapshot of game state for context (compact)
                try:
                    sd = EngineCompat.dump_state_debug(state, max_samples=2)
                    logger.debug('ExecuteCommand: state snapshot=%s', sd)
                except Exception:
                    try:
                        logger.debug('ExecuteCommand: failed to dump state snapshot')
                    except Exception:
                        pass
            except Exception:
                pass
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
        Loads cards using Python JSON loader first (preferred for robustness),
        but also attempts to load via Native loader if available to populate the cache
        for engine compatibility.
        """
        # Resolve path
        final_path = filepath
        if not os.path.exists(final_path):
            try:
                # Try finding relative to project root if path is relative
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                alt_path = os.path.join(base_dir, filepath)
                if os.path.exists(alt_path):
                    final_path = alt_path
            except Exception:
                pass

        # 1. Python Load (Data Source of Truth)
        py_data: Optional[Dict[int, Any]] = None
        try:
            import json
            with open(final_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert array format to dict format if needed
            if isinstance(data, list):
                card_dict = {}
                for card in data:
                    if isinstance(card, dict) and 'id' in card:
                        card_id = card['id']
                        card_dict[card_id] = card
                py_data = card_dict
            else:
                py_data = data
        except Exception as e:
            logger.error("Error loading cards (Python): %s", e)

        # 2. Native Side-load (for Engine Compatibility)
        if EngineCompat.is_available() and EngineCompat._native_enabled:
            try:
                native_db = EngineCompat.JsonLoader_load_cards(final_path)
                if native_db:
                    EngineCompat._native_db_cache = native_db
                    logger.info("EngineCompat: Native CardDB cached successfully.")
            except Exception:
                pass

        # 3. Return Logic
        if py_data is not None:
            return py_data

        # If Python failed but Native succeeded (rare/unlikely if file is valid JSON), return Native (wrapped/cached)
        if EngineCompat._native_db_cache:
            return cast(Dict[int, Any], EngineCompat._native_db_cache)

        return {}

    @staticmethod
    def _resolve_db(card_db: Any) -> Any:
        """Helper to swap dict with cached Native DB if available."""
        if isinstance(card_db, dict) and EngineCompat.is_available() and EngineCompat._native_enabled:
             if EngineCompat._native_db_cache:
                 return EngineCompat._native_db_cache
        return card_db

    @staticmethod
    def register_batch_inference_numpy(callback: Optional[Callable[[List[Any]], Any]]) -> None:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        if hasattr(dm_ai_module, 'register_batch_inference_numpy'):
            dm_ai_module.register_batch_inference_numpy(callback)
        else:
            logger.warning("dm_ai_module.register_batch_inference_numpy not found.")

    @staticmethod
    def TensorConverter_convert_to_tensor(state: GameState, player_id: int, card_db: CardDB) -> Union[List[float], Any]:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module, 'TensorConverter') and hasattr(dm_ai_module.TensorConverter, 'convert_to_tensor'):
            return dm_ai_module.TensorConverter.convert_to_tensor(state, player_id, real_db)
        # Return standard zero vector if native not available to prevent crashes in network modules
        return [0.0] * 856

    @staticmethod
    def get_pending_effects_info(state: GameState) -> List[Any]:
        EngineCompat._check_module()
        assert dm_ai_module is not None

        # Try module function first
        if hasattr(dm_ai_module, 'get_pending_effects_info'):
            try:
                return list(dm_ai_module.get_pending_effects_info(state))
            except Exception:
                pass

        # Fallback: manually inspect state.pending_effects
        # This mirrors the logic in the Python stub's get_pending_effects_info
        info = []
        try:
            pending = getattr(state, 'pending_effects', [])
            if pending:
                for cmd in pending:
                    t = getattr(cmd, 'type', 'UNKNOWN')
                    sid = getattr(cmd, 'source_instance_id', -1)
                    pid = getattr(cmd, 'target_player', 0)
                    info.append((str(t), sid, pid, cmd))
        except Exception:
            pass

        return info

    @staticmethod
    def create_parallel_runner(card_db: CardDB, sims: int, batch_size: int) -> Any:
        EngineCompat._check_module()
        assert dm_ai_module is not None
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module, 'ParallelRunner'):
            return dm_ai_module.ParallelRunner(real_db, sims, batch_size)
        return None

    @staticmethod
    def ParallelRunner_play_games(runner: Any, initial_states: List[GameState], evaluator_func: Callable, temperature: float, add_noise: bool, threads: int) -> List[Any]:
        if runner and hasattr(runner, 'play_games'):
            return list(runner.play_games(initial_states, evaluator_func, temperature, add_noise, threads))
        raise RuntimeError("ParallelRunner invalid or play_games not found")
