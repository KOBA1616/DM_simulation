from typing import Any, Dict, Optional, Protocol, runtime_checkable, List, cast
import os
# 再発防止: action_to_command は削除済み。command_to_dict で代替。
# 再発防止: map_action は command_to_dict に改名済み。旧名 map_action はエイリアスとして保持。
def command_to_dict(cmd: Any) -> Dict[str, Any]:
    """CommandDef / dict をシリアライズ可能な辞書に変換するユーティリティ。"""
    if isinstance(cmd, dict):
        return cmd
    if hasattr(cmd, 'to_dict'):
        try:
            return cmd.to_dict()
        except Exception:
            pass
    return {'type': str(getattr(cmd, 'type', 'UNKNOWN')), 'source_instance_id': getattr(cmd, 'source_instance_id', getattr(cmd, 'instance_id', -1))}

# 後方互換エイリアス
map_action = command_to_dict
import logging

# module logger
logger = logging.getLogger('dm_toolkit.commands')


def _call_native_command_generator(state: Any, card_db: Any) -> List[Any]:
    """C++ IntentGenerator.generate_legal_commands を呼び出し CommandDef リストを返す。

    再発防止: ActionGenerator（旧API）は削除済み。
    再発防止: _call_native_action_generator は _call_native_command_generator に改名済み。
    C++ IntentGenerator.generate_legal_commands のみを使用する。
    """
    try:
        import dm_ai_module
    except Exception:
        return []

    try:
        from dm_toolkit.engine.compat import EngineCompat
        native_db = EngineCompat._resolve_db(card_db)
    except Exception:
        native_db = card_db

    # IntentGenerator.generate_legal_commands が唯一の正規 API
    try:
        IG = getattr(dm_ai_module, 'IntentGenerator', None)
        if IG is not None and hasattr(IG, 'generate_legal_commands'):
            return IG.generate_legal_commands(state, native_db) or []
    except Exception:
        pass

    return []

# 後方互換エイリアス
_call_native_action_generator = _call_native_command_generator


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


def wrap_command(action: Any) -> Optional[ICommand]:
    """CommandDef を ICommand 互換オブジェクトとして返す。

    再発防止: _ActionWrapper（レガシーActionラッパー）は削除済み。
    再発防止: wrap_action は wrap_command に改名済み。後方互換エイリアス wrap_action を末尾に保持。
    CommandDef または execute 付きオブジェクトはそのまま返す。
    それ以外は EngineCompat.ExecuteCommand に委譲する薄いラッパーを返す。
    """
    if action is None:
        return None

    # 既に execute を持つ場合はそのまま返す（CommandDef / ICommand）
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return action  # type: ignore

    # 薄いラッパー: EngineCompat.ExecuteCommand に委譲
    class _CommandWrapper(BaseCommand):
        def __init__(self, a: Any):
            self._action = a

        def execute(self, state: Any, card_db: Any = None) -> Optional[Any]:
            try:
                from dm_toolkit.engine.compat import EngineCompat
                try:
                    EngineCompat.ExecuteCommand(state, self._action, card_db)
                except TypeError:
                    EngineCompat.ExecuteCommand(state, self._action)
            except Exception:
                return None
            return None

        def to_dict(self) -> Dict[str, Any]:
            return command_to_dict(self._action)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._action, name)

    return _CommandWrapper(action)

# 後方互換エイリアス: 既存の zone_widget.py / test_zone_display.py が import している
wrap_action = wrap_command


def generate_legal_commands(state: Any, card_db: Dict[Any, Any], strict: bool = False, skip_wrapper: bool = False) -> list:
    """Legal commands を生成するヘルパー。

    C++ IntentGenerator.generate_legal_commands を呼び出し CommandDef を返す。
    skip_wrapper=True の場合はネイティブオブジェクトをそのまま返す。

    Args:
        state: GameState object
        card_db: CardDatabase object or dict
        strict: True の場合、ネイティブジェネレーター不在時に RuntimeError。
        skip_wrapper: True の場合、生の CommandDef を返す。
    """
    dm_ai_module: Any = None
    try:
        # Ensure continuous effects / active modifiers are up-to-date before
        # delegating to the native intent generator. Use EngineCompat to
        # attempt recalc in both Python-fallback and native-backed GameState.
        try:
            from dm_toolkit.engine.compat import EngineCompat
            try:
                EngineCompat.ensure_recalculated(state)
            except Exception:
                pass
        except Exception:
            pass
        import dm_ai_module as dm_ai_module
    except Exception:
        if strict:
            raise RuntimeError("Native dm_ai_module not available")
        pass

    # strict 時: IntentGenerator の存在確認
    if strict:
        try:
            import dm_ai_module as dm_ai_module
            if not hasattr(dm_ai_module, 'IntentGenerator'):
                raise RuntimeError("IntentGenerator not found in dm_ai_module (strict mode)")
        except Exception:
            if strict: raise

    try:
        # Debug: report observed state phase and active player for diagnostics
        try:
            cur_phase = getattr(state, 'current_phase', None)
            phase_name = getattr(cur_phase, 'name', None) or str(cur_phase)
            logger.debug(f"Debug state_phase -> active_player={getattr(state, 'active_player_id', getattr(state, 'active_player', None))}, current_phase={phase_name}, raw={cur_phase}")
        except Exception:
            pass

        actions: List[Any] = []
        try:
            # Prefer command-first generator when available (returns command dicts)
            # Use robust native action generator helper that supports several
            # historical names and generator shapes (generate_commands, 
            # generate_legal_commands, generate_legal_commands, instance.generate).
            try:
                actions = _call_native_command_generator(state, card_db) or []
            except Exception:
                if strict:
                    raise
                actions = []
            # Debug: show types/reprs of first few returned actions to diagnose discrepancies
            try:
                sample = []
                for a in list(actions)[:6]:
                    try:
                        sample.append((type(a).__name__, repr(a)))
                    except Exception:
                        sample.append((type(a).__name__, str(a)))
                logger.debug(f"Debug generate_legal_commands -> count={len(actions)}, samples={sample}")

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

            # 再発防止: unified_execution.to_command_dict は削除済み。command_to_dict で代替。
            from typing import Optional
            normalized_cmds: List[Optional[Dict[str, Any]]] = []
            for a in list(actions):
                try:
                    normalized_cmds.append(command_to_dict(a))
                except Exception:
                    normalized_cmds.append(None)

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
            logger.warning(f"generate_legal_commands raised: {e}")
            pass

        # 再発防止: commands.generate_legal_commands 内で暗黙 fast_forward すると
        # game_session 側の進行責務（step/fast_forward）と二重化して状態競合を招く。
        # 既定では自動進行しない。必要時のみ DM_COMMANDS_AUTOFORWARD=1 で許可する。
        if (not actions) and dm_ai_module is not None and os.environ.get("DM_COMMANDS_AUTOFORWARD", "0") == "1":
            try:
                if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'fast_forward'):
                    # Convert Python dict to C++ CardDatabase if needed
                    from dm_toolkit.engine.compat import EngineCompat
                    native_db = EngineCompat._resolve_db(card_db)
                    dm_ai_module.PhaseManager.fast_forward(state, native_db)
                    # Re-query commands after fast_forward
                    actions = _call_native_command_generator(state, card_db) or []
            except Exception as e:
                logger.warning(f"commands auto-forward failed: {e}")

        # Trust C++ engine completely - wrap actions for GUI execution
        if skip_wrapper:
            return actions

        cmds = []
        for a in actions:
            w = wrap_command(a)
            if w is not None:
                cmds.append(w)
        return cmds
    except Exception as e:
        if strict:
            raise
        logger.exception(f"generate_legal_commands failed: {e}")
        import traceback
        traceback.print_exc()
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_command", "wrap_action", "generate_legal_commands", "command_to_dict"]
# 再発防止: wrap_action / map_action は後方互換エイリアス。新規コードでは wrap_command / command_to_dict を使用すること。
