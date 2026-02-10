from typing import Any, Dict, Optional, Protocol, runtime_checkable, List, cast
import os
from dm_toolkit.action_to_command import map_action
import warnings
import logging

# module logger
logger = logging.getLogger('dm_toolkit.commands')


def _call_native_action_generator(state: Any, card_db: Any) -> List[Any]:
    """Call the native action generator with fallbacks.

    Handles multiple possible native names for compatibility:
    - dm_ai_module.generate_commands
    - dm_ai_module.ActionGenerator.generate_legal_commands
    - dm_ai_module.ActionGenerator.generate_legal_actions
    - instance.generate(state, player_id)
    """
    # Allow disabling native engine calls when running tests or debugging
    # Set `DM_DISABLE_NATIVE=1` in the environment to force Python-only paths.
    try:
        if os.getenv('DM_DISABLE_NATIVE') in ('1', 'true', 'True'):
            logger.debug('DM_DISABLE_NATIVE is set; skipping native action generator')
            return []
    except Exception:
        pass

    try:
        import dm_ai_module
    except Exception:
        return []

    # Convert Python dict to C++ CardDatabase if needed
    try:
        from dm_toolkit.engine.compat import EngineCompat
        native_db = EngineCompat._resolve_db(card_db)
    except Exception:
        native_db = card_db  # Fallback to original if conversion fails

    # 1) Prefer a top-level generate_commands (command-first) if present
    try:
        if hasattr(dm_ai_module, 'generate_commands'):
            try:
                res = dm_ai_module.generate_commands(state, native_db) or []
                # Prefer non-empty command-first output; if empty, fall back to
                # legacy ActionGenerator paths to preserve compatibility.
                if res:
                    return res
            except Exception:
                pass
    except Exception:
        pass

    AG = getattr(dm_ai_module, 'ActionGenerator', None)
    if AG is None:
        return []

    # 2) Try static/classmethod generate_legal_actions (preferred)
    try:
        if hasattr(AG, 'generate_legal_actions'):
            try:
                return AG.generate_legal_actions(state, native_db) or []
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fallback to generate_legal_commands if it exists
    try:
        if hasattr(AG, 'generate_legal_commands'):
            try:
                return AG.generate_legal_commands(state, native_db) or []
            except Exception:
                pass
    except Exception:
        pass

    # 4) Try instance-based generator
    try:
        inst = AG()
        if hasattr(inst, 'generate'):
            try:
                return inst.generate(state, getattr(state, 'active_player_id', 0)) or []
            except Exception:
                pass
    except Exception:
        pass

    return []


@runtime_checkable
class ICommand(Protocol):
    def execute(self, state: Any, card_db: Any = None) -> Optional[Any]:
        ...

    def invert(self, state: Any) -> Optional[Any]:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...


class BaseCommand:
    """Minimal base command to serve as canonical interface for new commands."""

    def execute(self, state: Any) -> Optional[Any]:
        raise NotImplementedError()

    def invert(self, state: Any) -> Optional[Any]:
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "UNKNOWN", "kind": self.__class__.__name__}


# NOTE: Mana charge tracking moved to C++ side (turn_stats.mana_charged_this_turn).
# Python no longer maintains duplicate state.


def wrap_action(action: Any) -> Optional[ICommand]:
    """Return an `ICommand`-like object for the provided `action`.

    - If `action` already implements `execute`, return it.
    - Otherwise, returns a wrapper that implements `execute` via unified command path
      and `to_dict` via `map_action` from `action_to_command`.
    """
    if action is None:
        return None

    # If it's already command-like, return as-is
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return action  # type: ignore

    # Warn when wrapping legacy Action-like objects that do not provide a
    # precomputed `command` attribute. This helps surface places that still
    # rely on Action-only execution so they can migrate to command-first.
    try:
        has_cmd_attr = hasattr(action, 'command')
    except Exception:
        has_cmd_attr = False
    if not has_cmd_attr:
        try:
            warnings.warn(
                "Wrapping legacy Action-like object for execution; attach a 'command' attribute or migrate to ICommand/command dict to avoid this deprecation warning.",
                DeprecationWarning,
                stacklevel=3,
            )
        except Exception:
            pass

    # Unified wrapper: convert action-like object to command dict and execute via EngineCompat
    class _ActionWrapper(BaseCommand):
        def __init__(self, a: Any):
            self._action = a

        def execute(self, state: Any, card_db: Any = None) -> Optional[Any]:
            try:
                from dm_toolkit.unified_execution import ensure_executable_command
                from dm_toolkit.engine.compat import EngineCompat
                # Accept optional card_db when callers provide it (some callers
                # invoke execute(state, card_db)). Support both signatures.
                try:
                    # Try to read card_db if provided as second arg via Python's call
                    import inspect
                    sig = inspect.signature(self.execute)
                except Exception:
                    pass
                cmd = ensure_executable_command(self._action)
                # Pass through card_db when provided by caller.
                try:
                    EngineCompat.ExecuteCommand(state, cmd, card_db)
                except TypeError:
                    # Fallback if older signature expects two args
                    EngineCompat.ExecuteCommand(state, cmd)
                # NOTE: Mana charge tracking moved to C++ side (turn_stats.mana_charged_this_turn).
                # Python no longer maintains duplicate state after action execution.
            except Exception:
                return None
            return None

        def invert(self, state: Any) -> Optional[Any]:
            # Best-effort: delegate to underlying object if available
            try:
                inv = getattr(self._action, "invert", None)
                if callable(inv):
                    return inv(state)
            except Exception:
                pass
            return None

        def to_dict(self) -> Dict[str, Any]:
            # Use the unified execution mapper to preserve all normalization
            try:
                from dm_toolkit.unified_execution import to_command_dict
                cmd = to_command_dict(self._action)
                try:
                    # Fix legacy normalization where ATTACK was converted to
                    # type 'NONE' with a legacy_original_type marker.
                    if isinstance(cmd, dict):
                        orig = cmd.get('legacy_original_type') or cmd.get('legacy_type')
                        if isinstance(orig, str) and orig.upper() == 'ATTACK':
                            cmd['type'] = 'ATTACK'
                            cmd['unified_type'] = 'ATTACK'
                except Exception:
                    pass
                return cmd
            except Exception:
                # Fallback to direct mapping if unified path fails
                try:
                    return map_action(self._action)
                except Exception:
                    return {"type": "NONE"}

        def to_string(self) -> str:
            # Check if underlying action has to_string
            if hasattr(self._action, "to_string") and callable(getattr(self._action, "to_string")):
                return str(self._action.to_string())
            # Fallback to dict description
            d = self.to_dict()
            return str(d)

        def __getattr__(self, name: str) -> Any:
            # Delegate attribute access to underlying action
            return getattr(self._action, name)

    return _ActionWrapper(action)


def generate_legal_commands(state: Any, card_db: Dict[Any, Any]) -> list:
    """Compatibility helper: generate legal actions and return wrapped commands.

    Calls `dm_ai_module.ActionGenerator.generate_legal_actions` and maps each
    `Action` (or its attached `command`) to an `ICommand` via `wrap_action`.
    """
    try:
        import dm_ai_module
        # Ensure actions variable is always defined to avoid UnboundLocalError
        actions: List[Any] = []
        # Debug: report observed state phase and active player for diagnostics
        try:
            cur_phase = getattr(state, 'current_phase', None)
            phase_name = getattr(cur_phase, 'name', None) or str(cur_phase)
            logger.debug(f"Debug state_phase -> active_player={getattr(state, 'active_player_id', getattr(state, 'active_player', None))}, current_phase={phase_name}, raw={cur_phase}")
        except Exception:
            pass

            actions: List[Any] = []
            # Prefer native/module-level command generator if available
            try:
                import dm_ai_module as native
                try:
                    # dm_ai_module.generate_legal_commands is a module-level compatibility
                    # function that prefers native command-first output and falls back
                    # to a Python action-to-command mapping when native is unavailable.
                    actions = native.generate_legal_commands(state, card_db) or []
                    logger.debug(f"Using dm_ai_module.generate_legal_commands -> count={len(actions)}")
                except Exception:
                    actions = []
            except Exception:
                actions = []

            # If module-level generator returned nothing, try native action generator helper
            if not actions:
                try:
                    actions = _call_native_action_generator(state, card_db) or []
                except Exception:
                    actions = []
            # Debug: show types/reprs of first few returned actions to diagnose discrepancies
            try:
                sample = []
                for a in list(actions)[:6]:
                    try:
                        sample.append((type(a).__name__, repr(a)))
                    except Exception:
                        sample.append((type(a).__name__, str(a)))
                logger.debug(f"Debug generate_legal_actions -> count={len(actions)}, samples={sample}")

                # Additional diagnostics: when in ATTACK phase, log battle-zone creatures
                try:
                    cur_phase = getattr(state, 'current_phase', None)
                    pname = getattr(cur_phase, 'name', None) or str(cur_phase)
                    if isinstance(pname, str) and 'ATTACK' in pname.upper():
                        pid = getattr(state, 'active_player_id', 0)
                        try:
                            player = state.players[pid]
                            bz = list(getattr(player, 'battle_zone', []) or [])
                            bz_info = []
                            for c in bz:
                                try:
                                    bz_info.append({
                                        'instance_id': getattr(c, 'instance_id', None),
                                        'card_id': getattr(c, 'card_id', None),
                                        'is_tapped': getattr(c, 'is_tapped', None),
                                        'sick': getattr(c, 'sick', None),
                                    })
                                except Exception:
                                    try:
                                        bz_info.append({'repr': repr(c)})
                                    except Exception:
                                        bz_info.append({'repr': str(c)})
                            logger.debug(f"Debug battle_zone -> pid={pid}, count={len(bz)}, creatures={bz_info}")
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass

            # Normalize native actions to command dicts when possible so we can
            # reliably detect whether PLAY_CARD (or its unified equivalent)
            # is present. This absorbs differences where native bindings
            # expose enums/fields differently (int vs Enum, value1 vs amount).
            from typing import Optional
            normalized_cmds: List[Optional[Dict[str, Any]]] = []
            try:
                from dm_toolkit.unified_execution import to_command_dict
                for a in list(actions):
                    try:
                        normalized_cmds.append(to_command_dict(a))
                    except Exception:
                        try:
                            # Fallback: try map_action directly
                            normalized_cmds.append(map_action(a))
                        except Exception:
                            normalized_cmds.append(None)
            except Exception:
                # If unified path not importable, best-effort map_action attempt
                try:
                    for a in list(actions):
                        try:
                            normalized_cmds.append(map_action(a))
                        except Exception:
                            normalized_cmds.append(None)
                except Exception:
                    normalized_cmds = [None] * len(actions)

            # If native mapping preserved the legacy original type for
            # ATTACK (some older native paths return ATTACK as a legacy
            # token), normalize it so higher-level counting and UI code
            # see an explicit 'ATTACK' type instead of 'NONE' with a
            # legacy_original_type marker.
            try:
                if isinstance(normalized_cmds, list):
                    for c in normalized_cmds:
                        try:
                            if isinstance(c, dict):
                                orig = c.get('legacy_original_type') or c.get('legacy_type')
                                if isinstance(orig, str) and orig.upper() == 'ATTACK':
                                    c['type'] = 'ATTACK'
                                    c['unified_type'] = 'ATTACK'
                        except Exception:
                            pass
            except Exception:
                pass

            # Debug: dump normalized commands (safe representations)
            try:
                dump_norm = []
                for c in normalized_cmds:
                    try:
                        if isinstance(c, dict):
                            dump_norm.append(c)
                        else:
                            dump_norm.append({'_repr': repr(c)})
                    except Exception:
                        dump_norm.append({'_repr': str(c)})
                logger.debug(f"Debug normalized_cmds -> count={len(dump_norm)}, entries={dump_norm[:12]}")
            except Exception:
                pass

            # Debug: dump raw native actions (type, action.type name if present, and repr samples)
            try:
                raw_dump = []
                raw_type_names = []
                for i, a in enumerate(list(actions)):
                    try:
                        a_type = getattr(a, 'type', None)
                        try:
                            tname = getattr(a_type, 'name', None) or str(a_type)
                        except Exception:
                            tname = str(a_type)
                        raw_type_names.append(str(tname))
                        raw_dump.append((i, type(a).__name__, tname, repr(a)))
                    except Exception:
                        try:
                            raw_dump.append((i, type(a).__name__, None, str(a)))
                        except Exception:
                            raw_dump.append((i, type(a).__name__, None, '<unreprable>'))
                logger.debug(f"Debug raw_actions -> count={len(actions)}, types={raw_type_names[:12]}, samples={raw_dump[:12]}")
            except Exception:
                pass

            # Use normalized_cmds to detect presence of play-type commands
            try:
                # Robustly detect play-like commands even when `type` is an Enum
                has_play_native = False
                for c in normalized_cmds:
                    if not isinstance(c, dict):
                        continue
                    t = c.get('type') or c.get('legacy_original_type')
                    if t is None:
                        continue
                    # Prefer enum name when available
                    try:
                        tname = getattr(t, 'name', None) or str(t)
                    except Exception:
                        tname = str(t)
                    tt = tname.upper()
                    # Accept several aliases and unified indicators for play actions
                    if tt in ('PLAY_FROM_ZONE', 'PLAY_FROM_BUFFER', 'CAST_SPELL', 'PLAY_CARD', 'FRIEND_BURST', 'PLAY', 'DECLARE_PLAY', 'PUT_INTO_PLAY'):
                        has_play_native = True
                        break
                    # Also inspect other possible hints
                    if isinstance(c.get('unified_type'), str) and str(c.get('unified_type')).upper().startswith('PLAY'):
                        has_play_native = True
                        break
            except Exception:
                has_play_native = False
        except Exception as e:
            # If it fails, we may have a format mismatch
            # Log for debugging but don't fail - just return empty list
            print(f"Warning: generate_legal_actions raised: {e}")
            pass

        # If C++ returned no actions, call fast_forward to progress game
        # This handles phases like START_OF_TURN, DRAW that have no player actions
        # If C++ engine returns no actions (auto-phases like START_OF_TURN, DRAW),
        # fast-forward to next decision point
        if not actions:
            try:
                if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'fast_forward'):
                    # Convert Python dict to C++ CardDatabase if needed
                    from dm_toolkit.engine.compat import EngineCompat
                    native_db = EngineCompat._resolve_db(card_db)
                    dm_ai_module.PhaseManager.fast_forward(state, native_db)
                    # Re-query actions after fast_forward
                    actions = _call_native_action_generator(state, card_db) or []
            except Exception:
                pass  # Silent fallback - if fast_forward fails, return empty actions

        # If no native actions found, synthesize simple play candidates from
        # Python state as a best-effort fallback so tests and tools can
        # exercise play logic without the C++ engine. This helps when the
        # native engine returns an empty set for the current state.
        try:
            if not actions:
                pid = getattr(state, 'active_player_id', 0)
                hand = []
                try:
                    hand = list(getattr(state.players[pid], 'hand', []) or [])
                except Exception:
                    hand = []
                synth = []
                for c in hand:
                    try:
                        iid = getattr(c, 'instance_id', None) or getattr(c, 'id', None)
                        # Prefer creating an Action-like object when the shim exposes Action/ActionType
                        act_obj = None
                        try:
                            import dm_ai_module
                            ActionCls = getattr(dm_ai_module, 'Action', None)
                            ActionType = getattr(dm_ai_module, 'ActionType', None)
                            if ActionCls is not None:
                                act_obj = ActionCls()
                                # Set type to PLAY_CARD when ActionType exists
                                try:
                                    if ActionType is not None and hasattr(ActionType, 'PLAY_CARD'):
                                        act_obj.type = getattr(ActionType, 'PLAY_CARD')
                                    else:
                                        act_obj.type = 'PLAY_CARD'
                                except Exception:
                                    act_obj.type = 'PLAY_CARD'
                                # Attach identifiers
                                try:
                                    setattr(act_obj, 'instance_id', iid)
                                except Exception:
                                    pass
                        except Exception:
                            act_obj = None

                        if act_obj is not None:
                            synth.append(act_obj)
                        else:
                            synth.append({'type': 'PLAY_FROM_ZONE', 'instance_id': iid, 'player_id': pid, 'unified_type': 'PLAY'})
                    except Exception:
                        continue
                actions = synth
                if actions:
                    logger.debug(f"Synthesized {len(actions)} play actions as Python fallback (DM_DISABLE_NATIVE)")
        except Exception:
            pass
        
        # Trust C++ engine completely - wrap actions for GUI execution
        cmds = []
        for a in actions:
            w = wrap_action(a)
            if w is not None:
                cmds.append(w)
        return cmds
    except Exception as e:
        print(f"generate_legal_commands failed: {e}")
        import traceback
        traceback.print_exc()
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_action", "generate_legal_commands"]
