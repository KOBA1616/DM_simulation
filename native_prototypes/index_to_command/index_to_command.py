"""Python wrapper for the native index_to_command prototype.

Tries to import the compiled `index_to_command_native` module; if unavailable,
falls back to a pure-Python implementation that mirrors the C++ prototype.
"""

try:
    import index_to_command_native as _native
except Exception:
    _native = None


def _py_index_to_command(idx: int) -> dict:
    if idx == 0:
        return {"type": "PASS"}
    elif 0 < idx < 20:
        return {"type": "MANA_CHARGE", "slot_index": idx}
    else:
        return {"type": "PLAY_FROM_ZONE", "slot_index": idx - 20}


def index_to_command(idx: int) -> dict:
    """Map an action index to a command dict using native implementation when available.

    Args:
        idx: action index from model/MCTS output
    Returns:
        dict: command-like dict (keys: 'type', optional 'slot_index')
    """
    if _native is not None:
        try:
            return _native.index_to_command(int(idx))
        except Exception:
            pass
    return _py_index_to_command(int(idx))
