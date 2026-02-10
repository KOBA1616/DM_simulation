from typing import Any, Dict, List, Optional

from dm_toolkit import commands_v2 as commands

try:
    import dm_ai_module
except Exception:
    # allow tests/tooling to import without native module
    dm_ai_module = None


def generate_legal_commands(state: Any, card_db: Dict[int, Any], strict: bool = False) -> List[Any]:
    """Prefer command-first generation, fall back to legacy shims.

    Returns a list of command-like objects (dict-like or objects with `to_dict`).
    """
    # Try the preferred v2 generator first
    try:
        cmds = commands.generate_legal_commands(state, card_db, strict=strict) or []
        return cmds
    except Exception:
        pass

    # Fallback to legacy commands module (if present)
    try:
        from dm_toolkit import commands as legacy_commands

        return legacy_commands.generate_legal_commands(state, card_db, strict=strict) or []
    except Exception:
        pass

    # As a last resort, return empty list
    return []


def normalize_to_command(obj: Any) -> Dict[str, Any]:
    """Return a plain dict representation for a command-like object.

    - If obj is a dict, return copy
    - If obj has `to_dict()`, call it
    - Otherwise, try to build from attributes
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'to_dict'):
        try:
            return obj.to_dict()
        except Exception:
            pass

    # Generic attribute extraction (best-effort)
    out: Dict[str, Any] = {}
    for k in ('type', 'action_type', 'source_instance_id', 'instance_id', 'target'):
        if hasattr(obj, k):
            try:
                out[k] = getattr(obj, k)
            except Exception:
                pass
    return out


def command_to_index(cmd_obj: Any) -> Optional[int]:
    """Map a command-like object to encoder index, if `dm_ai_module` available.

    Returns None on failure.
    """
    if dm_ai_module is None:
        return None
    try:
        d = normalize_to_command(cmd_obj)
        return dm_ai_module.CommandEncoder.command_to_index(d)
    except Exception:
        try:
            if hasattr(cmd_obj, 'to_dict'):
                return dm_ai_module.CommandEncoder.command_to_index(cmd_obj.to_dict())
        except Exception:
            return None
    return None
