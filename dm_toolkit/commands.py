from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dm_toolkit.action_to_command import map_action

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


# Tracks whether a given player has performed a mana-charge during the current
# turn for a particular `state` object. Keyed by (id(state), player_id).
_mana_charged_by_state: Dict[tuple, bool] = {}
# Track last observed mana count per (state, player) so we can detect
# mana increases performed outside the Python wrapper (native flow commands,
# etc.) and mark _mana_charged_by_state accordingly.
_last_mana_count_by_state: Dict[tuple, int] = {}


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
                # After successful execution, if this underlying action is a
                # MANA_CHARGE, mark that the active player has charged mana for
                # this turn so future legal-action generations can enforce the
                # "once per turn" rule.
                try:
                    a_type = getattr(self._action, 'type', None)
                    tname = ''
                    if a_type is not None:
                        tname = getattr(a_type, 'name', str(a_type))
                    if isinstance(tname, str) and tname.endswith('MANA_CHARGE'):
                        sid = id(state)
                        pid = getattr(state, 'active_player_id', 0)
                        _mana_charged_by_state[(sid, pid)] = True
                except Exception:
                    pass
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
                return to_command_dict(self._action)
            except Exception:
                # Fallback to direct mapping if unified path fails
                try:
                    return map_action(self._action)
                except Exception:
                    return {"type": "NONE"}

        def to_string(self) -> str:
            # Check if underlying action has to_string
            if hasattr(self._action, "to_string") and callable(getattr(self._action, "to_string")):
                return self._action.to_string()
            # Fallback to dict description
            d = self.to_dict()
            return str(d)

        def __getattr__(self, name: str) -> Any:
            # Delegate attribute access to underlying action
            return getattr(self._action, name)

    return _ActionWrapper(action)


def generate_legal_commands(state: Any, card_db: Dict[int, Any]) -> list:
    """Compatibility helper: generate legal actions and return wrapped commands.

    Calls `dm_ai_module.ActionGenerator.generate_legal_actions` and maps each
    `Action` (or its attached `command`) to an `ICommand` via `wrap_action`.
    """
    try:
        import dm_ai_module
        # Debug: report observed state phase and active player for diagnostics
        try:
            cur_phase = getattr(state, 'current_phase', None)
            phase_name = getattr(cur_phase, 'name', None) or str(cur_phase)
            print(f"Debug state_phase -> active_player={getattr(state, 'active_player_id', getattr(state, 'active_player', None))}, current_phase={phase_name}, raw={cur_phase}")
        except Exception:
            pass

        actions = []
        try:
            # Try to use card_db directly - it could be a native CardDatabase or a dict
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db) or []
            # Debug: show types/reprs of first few returned actions to diagnose discrepancies
            try:
                sample = []
                for a in list(actions)[:6]:
                    try:
                        sample.append((type(a).__name__, repr(a)))
                    except Exception:
                        sample.append((type(a).__name__, str(a)))
                print(f"Debug generate_legal_actions -> count={len(actions)}, samples={sample}")
            except Exception:
                pass

            # Normalize native actions to command dicts when possible so we can
            # reliably detect whether PLAY_CARD (or its unified equivalent)
            # is present. This absorbs differences where native bindings
            # expose enums/fields differently (int vs Enum, value1 vs amount).
            normalized_cmds = []
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
                print(f"Debug normalized_cmds -> count={len(dump_norm)}, entries={dump_norm[:12]}")
            except Exception:
                pass

            # Debug: dump raw native actions (type and repr samples)
            try:
                raw_dump = []
                for i, a in enumerate(list(actions)):
                    try:
                        raw_dump.append((i, type(a).__name__, repr(a)))
                    except Exception:
                        raw_dump.append((i, type(a).__name__, str(a)))
                print(f"Debug raw_actions -> count={len(actions)}, samples={raw_dump[:12]}")
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

        # Reset per-turn tracking when a new start-of-turn phase is observed
        try:
            sid = id(state)
            pid = getattr(state, 'active_player_id', 0)
            cur_phase = getattr(state, 'current_phase', None)
            pstr = ''
            if cur_phase is not None:
                pstr = getattr(cur_phase, 'name', str(cur_phase))
            # Treat phases that end with START_OF_TURN as the reset point
            # Read current observed mana for the active player (best-effort)
            try:
                player = state.players[pid]
                cur_mana = getattr(player, 'mana_count', None)
                if cur_mana is None:
                    cur_mana = len(getattr(player, 'mana_zone', []) or [])
            except Exception:
                cur_mana = None

            # Reset flag at start-of-turn for the active player and initialize
            # the last-observed mana count to avoid false positives.
            if isinstance(pstr, str) and pstr.endswith('START_OF_TURN'):
                _mana_charged_by_state[(sid, pid)] = False
                if cur_mana is not None:
                    _last_mana_count_by_state[(sid, pid)] = int(cur_mana)

            # If we observe that the player's mana count has increased since
            # the last observation, mark that they have effectively charged
            # mana (covers native FlowCommand paths that bypass our wrapper).
            try:
                if cur_mana is not None:
                    last = _last_mana_count_by_state.get((sid, pid))
                    if last is None:
                        _last_mana_count_by_state[(sid, pid)] = int(cur_mana)
                    else:
                        if int(cur_mana) > int(last):
                            _mana_charged_by_state[(sid, pid)] = True
                            _last_mana_count_by_state[(sid, pid)] = int(cur_mana)
            except Exception:
                pass
        except Exception:
            pass

        # If the engine returned no actions, provide a conservative Python-side
        # fallback so the UI / AI can continue progressing while we debug
        # the native ActionGenerator implementation.
        # If engine returned nothing, or only a PASS (native may return PASS in
        # a start-of-turn subphase), provide a conservative Python-side
        # fallback so the UI / AI can continue progressing while we debug
        # the native ActionGenerator implementation.
        pass_only = False
        try:
            if actions and len(actions) == 1:
                # Detect PASS-only by inspecting normalized command when available
                if normalized_cmds and len(normalized_cmds) >= 1 and isinstance(normalized_cmds[0], dict):
                    t = normalized_cmds[0].get('type') or normalized_cmds[0].get('legacy_original_type')
                    if isinstance(t, str) and t.upper() == 'PASS':
                        pass_only = True
                else:
                    a = actions[0]
                    tname = getattr(a, 'type', None)
                    # For native objects the type may be an Enum; compare by name/str
                    if tname is not None and (str(tname).endswith('PASS') or getattr(tname, 'name', '') == 'PASS'):
                        pass_only = True
        except Exception:
            pass

        if not actions or pass_only:
            try:
                from dm_ai_module import Action, ActionType

                pid = getattr(state, 'active_player_id', 0)
                player = state.players[pid]

                # PASS is always legal
                pass_act = Action()
                pass_act.type = ActionType.PASS
                actions.append(pass_act)

                # Mana charge actions (allow charging any card in hand) -
                # only propose them if the player hasn't already charged mana
                # this turn (runtime-tracked). We don't mark the tracker here;
                # it's marked when an actual MANA_CHARGE is executed.
                try:
                    sid = id(state)
                    pid = getattr(state, 'active_player_id', 0)
                    already_charged = _mana_charged_by_state.get((sid, pid), False)
                    if not already_charged:
                        for c in list(getattr(player, 'hand', [])):
                            act = Action()
                            act.type = ActionType.MANA_CHARGE
                            act.card_id = getattr(c, 'card_id', c)
                            act.source_instance_id = getattr(c, 'instance_id', -1)
                            actions.append(act)
                except Exception:
                    pass

                # Playable cards heuristics: if enough untapped mana, propose PLAY_CARD
                try:
                    usable_mana = sum(1 for m in getattr(player, 'mana_zone', []) if not getattr(m, 'is_tapped', False))
                    for c in list(getattr(player, 'hand', [])):
                        cid = getattr(c, 'card_id', c)
                        # Robust card_db lookup (try multiple interfaces/keys)
                        cost = 9999
                        cdef = None
                        try:
                            if card_db is None:
                                cdef = None
                            elif hasattr(card_db, 'get_card'):
                                try:
                                    cdef = card_db.get_card(cid)
                                except Exception:
                                    # Some implementations may expect int keys
                                    cdef = card_db.get_card(int(cid)) if isinstance(cid, str) and cid.isdigit() else None
                            elif isinstance(card_db, dict):
                                # try int and str keys
                                try:
                                    cdef = card_db.get(cid)
                                except Exception:
                                    cdef = None
                                if cdef is None:
                                    cdef = card_db.get(str(cid)) if isinstance(cid, (int, str)) else None
                                if cdef is None and isinstance(cid, str) and cid.isdigit():
                                    cdef = card_db.get(int(cid))
                            elif hasattr(card_db, 'cards'):
                                try:
                                    cards_attr = getattr(card_db, 'cards')
                                    if isinstance(cards_attr, dict):
                                        cdef = cards_attr.get(cid) or cards_attr.get(str(cid))
                                except Exception:
                                    cdef = None
                            else:
                                cdef = None
                            if cdef and isinstance(cdef, dict) and 'cost' in cdef:
                                try:
                                    cost = int(cdef.get('cost', cost))
                                except Exception:
                                    cost = cost
                        except Exception:
                            cdef = None
                            cost = 9999

                        # Debug lookup result
                        try:
                            print(f"Debug play_lookup -> pid={pid}, card_id={cid}, cdef_present={bool(cdef)}, assumed_cost={cost}")
                        except Exception:
                            pass

                        if cost <= usable_mana:
                            act = Action()
                            act.type = ActionType.PLAY_CARD
                            act.card_id = cid
                            act.source_instance_id = getattr(c, 'instance_id', -1)
                            actions.append(act)
                            try:
                                print(f"Debug play_heuristic1 -> pid={pid}, usable_mana={usable_mana}, card_id={cid}, cdef_present={bool(cdef)}, cost={cost}")
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception:
                # If we cannot construct fallback actions, keep actions empty
                pass

            # If native generator did not include any PLAY_CARD actions, attempt
            # to add PLAY_CARD candidates using the same heuristic we use in the
            # pure-Python ActionGenerator. This covers cases where the native
            # generator omits play options due to mismatched card_db formats or
            # other minor interoperability issues.
            try:
                # Only propose play candidates when we're in the Main phase
                cur_phase = getattr(state, 'current_phase', None)
                p_name = getattr(cur_phase, 'name', None) or str(cur_phase)
                in_main_phase = False
                try:
                    if isinstance(p_name, str) and 'MAIN' in p_name.upper():
                        in_main_phase = True
                    elif isinstance(cur_phase, int) and int(cur_phase) == 3:
                        in_main_phase = True
                except Exception:
                    in_main_phase = False

                # If normalized detection found a play, skip heuristic
                if not ('has_play_native' in locals() and has_play_native) and in_main_phase:
                    has_play = False
                    for a in list(actions):
                        try:
                            t = getattr(a, 'type', None)
                            tname = getattr(t, 'name', str(t)) if t is not None else str(t)
                            if isinstance(tname, str) and tname.endswith('PLAY_CARD'):
                                has_play = True
                                break
                        except Exception:
                            continue
                    if not has_play:
                        from dm_ai_module import Action, ActionType
                        pid = getattr(state, 'active_player_id', 0)
                        player = state.players[pid]
                        usable_mana = sum(1 for m in getattr(player, 'mana_zone', []) if not getattr(m, 'is_tapped', False))
                        for c in list(getattr(player, 'hand', [])):
                            cid = getattr(c, 'card_id', c)
                            cost = 9999
                            try:
                                if isinstance(card_db, dict):
                                    cdef = card_db.get(cid) or card_db.get(str(cid))
                                elif hasattr(card_db, 'get_card'):
                                    cdef = card_db.get_card(cid)
                                else:
                                    cdef = None
                                if cdef:
                                    cost = int(cdef.get('cost', 9999))
                            except Exception:
                                cost = 9999
                            if cost <= usable_mana:
                                act = Action()
                                act.type = ActionType.PLAY_CARD
                                act.card_id = cid
                                act.source_instance_id = getattr(c, 'instance_id', -1)
                                actions.append(act)
                                try:
                                    print(f"Debug play_heuristic2 -> pid={pid}, usable_mana={usable_mana}, card_id={cid}, cdef_present={bool(cdef)}, cost={cost}")
                                except Exception:
                                    pass
            except Exception:
                pass

        # Filter out MANA_CHARGE actions after they have been performed this
        # turn (based on the runtime tracker). We only present a single
        # candidate MANA_CHARGE to the UI/AI when possible to avoid
        # multi-charge loops; the executed wrapper marks the tracker so
        # subsequent generations will omit further MANA_CHARGE options.
        filtered = []
        try:
            sid = id(state)
            pid = getattr(state, 'active_player_id', 0)
            charged = _mana_charged_by_state.get((sid, pid), False)
            mana_added = False
            for a in actions:
                try:
                    a_type = getattr(a, 'type', None)
                    tname = ''
                    if a_type is not None:
                        tname = getattr(a_type, 'name', str(a_type))
                    if isinstance(tname, str) and tname.endswith('MANA_CHARGE'):
                        if charged:
                            continue
                        if mana_added:
                            continue
                        mana_added = True
                except Exception:
                    pass
                filtered.append(a)
        except Exception:
            filtered = actions

        cmds = []
        for a in filtered:
            w = wrap_action(a)
            if w is not None:
                cmds.append(w)
        return cmds
    except Exception:
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_action", "generate_legal_commands"]
