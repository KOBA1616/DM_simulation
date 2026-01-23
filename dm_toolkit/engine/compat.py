# -*- coding: utf-8 -*-
import sys
import os
import logging
from typing import Any, List, Optional, Callable, Dict, Union
import enum
from types import ModuleType

from dm_toolkit.debug.effect_tracer import get_tracer, TraceEventType

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
                return dm_ai_module.get_execution_context(state)
            except Exception:
                pass
        # Fallback if state has it directly (Python stub)
        if hasattr(state, 'execution_context'):
             # Check if it's an object with .variables or a dict
             ctx = getattr(state, 'execution_context')
             if hasattr(ctx, 'variables'):
                 return ctx.variables
             if isinstance(ctx, dict):
                 return ctx
        return {}

    @staticmethod
    def get_command_details(cmd: Any) -> str:
        """Wraps dm_ai_module.get_command_details"""
        if dm_ai_module and hasattr(dm_ai_module, 'get_command_details'):
            try:
                return dm_ai_module.get_command_details(cmd)
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

                def _sample(zone_obj):
                    samples = []
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
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module.EffectResolver, 'resume'):
            dm_ai_module.EffectResolver.resume(state, real_db, selection)
        else:
            logger.warning("dm_ai_module.EffectResolver.resume not found.")

    @staticmethod
    def EffectResolver_resolve_action(state: GameState, action: Action, card_db: CardDB) -> None:
        get_tracer().log_event(TraceEventType.EFFECT_RESOLUTION, "Resolving Action", {"action": str(action)})
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
        if hasattr(dm_ai_module.EffectResolver, 'resolve_action'):
            dm_ai_module.EffectResolver.resolve_action(state, action, real_db)
        else:
            logger.warning("dm_ai_module.EffectResolver.resolve_action not found.")

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
                    def _phase_name(x):
                        try:
                            return str(x)
                        except Exception:
                            try:
                                return getattr(x, 'name', repr(x))
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
                                    def _to_int(x):
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
                                            if before_int in sorted_values:
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

                        def _to_phase(x):
                            try:
                                if x is None:
                                    return None
                                if isinstance(x, PhaseEnum):
                                    return x
                                if hasattr(x, 'value'):
                                    return PhaseEnum(int(getattr(x, 'value')))
                                if isinstance(x, str):
                                    nm = x.split('.', 1)[1] if x.startswith('Phase.') else x
                                    nm = nm.strip()
                                    if hasattr(PhaseEnum, nm):
                                        return getattr(PhaseEnum, nm)
                                return PhaseEnum(int(x))
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
                def _phase_name(x):
                    try:
                        return str(x)
                    except Exception:
                        try:
                            return getattr(x, 'name', repr(x))
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
                    if cnt > 15:
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
    def PhaseManager_check_game_over(state: GameState):
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
        real_db = EngineCompat._resolve_db(card_db)
        from dm_toolkit.commands import generate_legal_commands
        return generate_legal_commands(state, real_db)

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

        get_tracer().log_command(cmd_dict if cmd_dict else {"raw_cmd": str(cmd)})

        # 1. Try C++ CommandSystem if it's a wrapped Command with `to_dict`
        # and has a valid type for the engine.
        if cmd_dict:
            try:
                logger.debug('EngineCompat: cmd_dict detected: %s', cmd_dict)
                type_str = cmd_dict.get('type')
                logger.debug('EngineCompat: type_str = %s', type_str)

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
                        if 'instance_id' in cmd_dict:
                            cmd_def.instance_id = int(cmd_dict['instance_id'])
                        elif 'source_instance_id' in cmd_dict:
                            cmd_def.instance_id = int(cmd_dict['source_instance_id'])

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
                        logger.debug('EngineCompat: calling CommandSystem.execute_command single-shot')
                        dm_ai_module.CommandSystem.execute_command(state, cmd_def, source_id, player_id, ctx)
                        logger.debug('EngineCompat: called CommandSystem.execute_command')
                        return # Success: Return only if executed by CommandSystem
            except Exception:
                logger.exception('EngineCompat: exception during CommandSystem mapping')
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

            # If it's an Action-like object, use EffectResolver
            if hasattr(cmd, 'type'):
                if hasattr(dm_ai_module.EffectResolver, 'resolve_action'):
                    dm_ai_module.EffectResolver.resolve_action(state, cmd, EngineCompat._resolve_db(card_db))
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
                logger.debug('EngineCompat: Python-fallback player_id=%s ctype=%s', player_id, ctype)
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
                        logger.debug('EngineCompat: REPLACE_CARD_MOVE moved %s card(s) from %s to %s', moved, (original_zone or 'UNKNOWN'), dest_zone)
                        return

                if ctype in ('TAP', 'UNTAP', 'RETURN_TO_HAND'):
                    filter_dict = cd.get('target_filter') or {}
                    zones = filter_dict.get('zones') or []
                    targets = []
                    for z in zones:
                        for inst in _resolve_zone_instances_local(str(z)):
                            targets.append(inst)

                    if targets:
                        logger.debug('EngineCompat: Python-fallback found targets count %s', len(targets))
                        if ctype == 'TAP':
                            for inst in targets:
                                try:
                                    setattr(inst, 'is_tapped', True)
                                except Exception:
                                    pass
                            return

                # Fallback for standard actions
                if ctype == 'MANA_CHARGE':
                    instance_id = cd.get('source_instance_id') or cd.get('instance_id')
                    if instance_id:
                        p = state.players[player_id]
                        for i, c in enumerate(p.hand):
                            if getattr(c, 'instance_id', -1) == instance_id:
                                card = p.hand.pop(i)
                                card.is_tapped = False
                                p.mana_zone.append(card)
                                return

                if ctype == 'PLAY_FROM_ZONE':
                    instance_id = cd.get('source_instance_id') or cd.get('instance_id')
                    from_z = str(cd.get('from_zone', '')).upper()
                    to_z = str(cd.get('to_zone', '')).upper()

                    if instance_id:
                        src_list = _resolve_zone_instances_local(from_z)
                        card = None
                        for i, c in enumerate(src_list):
                            if getattr(c, 'instance_id', -1) == instance_id:
                                card = src_list.pop(i)
                                break

                        if card:
                            dest_list = _resolve_zone_instances_local(to_z)
                            # If to_z is implied or explicitly standard
                            if 'BATTLE' in to_z:
                                card.sick = True
                                card.is_tapped = False

                            if dest_list is not None:
                                dest_list.append(card)
                            return

                if ctype == 'PLAY_CARD':
                    instance_id = cd.get('source_instance_id') or cd.get('instance_id')
                    if instance_id:
                        p = state.players[player_id]
                        # Try hand first
                        card = None
                        for i, c in enumerate(p.hand):
                            if getattr(c, 'instance_id', -1) == instance_id:
                                card = p.hand.pop(i)
                                break
                        if card:
                            # Assume Creature goes to Battle, Spell goes to Grave (simplification)
                            # Ideally check card_db but this is deep fallback
                            # Check if card has type info attached
                            is_spell = False
                            cid = getattr(card, 'card_id', -1)
                            cdef = {}

                            if card_db:
                                if hasattr(card_db, 'get_card'):
                                    cdef = card_db.get_card(cid)
                                elif isinstance(card_db, dict):
                                    cdef = card_db.get(cid) or card_db.get(str(cid))

                            # Fallback to global DB if provided DB is missing/empty and we have the module
                            if not cdef and dm_ai_module and hasattr(dm_ai_module, 'CardDatabase'):
                                try:
                                    # Try static get_card
                                    if hasattr(dm_ai_module.CardDatabase, 'get_card'):
                                        cdef = dm_ai_module.CardDatabase.get_card(cid)
                                except Exception:
                                    pass

                            if cdef and cdef.get('type') == 'SPELL':
                                is_spell = True

                            if is_spell:
                                p.graveyard.append(card)
                            else:
                                card.sick = True
                                card.is_tapped = False
                                p.battle_zone.append(card)
                            return

                if ctype == 'ATTACK_PLAYER':
                    instance_id = cd.get('source_instance_id') or cd.get('instance_id')
                    if instance_id:
                        p = state.players[player_id]
                        for c in p.battle_zone:
                            if getattr(c, 'instance_id', -1) == instance_id:
                                c.is_tapped = True
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
            logger.warning('ExecuteCommand could not execute given command/object: %s', cmd)
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
            return EngineCompat._native_db_cache

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
        real_db = EngineCompat._resolve_db(card_db)
        if hasattr(dm_ai_module, 'ParallelRunner'):
            return dm_ai_module.ParallelRunner(real_db, sims, batch_size)
        return None

    @staticmethod
    def ParallelRunner_play_games(runner: Any, initial_states: List[GameState], evaluator_func: Callable, temperature: float, verbose: bool, threads: int) -> List[Any]:
        if runner and hasattr(runner, 'play_games'):
            return list(runner.play_games(initial_states, evaluator_func, temperature, verbose, threads))
        raise RuntimeError("ParallelRunner invalid or play_games not found")
