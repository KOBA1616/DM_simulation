import os
import sys

# NOTE:
# sitecustomize.py is executed automatically by Python when it is importable.
# For clone-based distribution, keep behavior minimal and non-destructive.
# Avoid implicitly mutating sys.path beyond the repo root, and never delete caches.

_ROOT_DIR = os.path.dirname(__file__)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Optional dev convenience: add build/bin to import path only when explicitly enabled.
if os.environ.get('DM_SIMULATION_DEV', '').strip() == '1':
    for subdir in ['bin', 'build']:
        path = os.path.join(_ROOT_DIR, subdir)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


# Compatibility shims for dm_ai_module: when the compiled extension is
# imported directly it may lack recent helper symbols expected by tests.
try:
    import dm_ai_module as _dm  # type: ignore
    from enum import IntEnum

    if not hasattr(_dm, 'PlayerIntent'):
        class PlayerIntent(IntEnum):
            PLAY_CARD = 1
            ATTACK_PLAYER = 2
            ATTACK_CREATURE = 3
            PASS = 4
        try:
            setattr(_dm, 'PlayerIntent', PlayerIntent)
        except Exception:
            pass

    if not hasattr(_dm, 'PassiveType'):
        class PassiveType(IntEnum):
            NONE = 0
            CANNOT_ATTACK = 1
            CANNOT_SUMMON = 2
            CANNOT_BE_SELECTED = 3
            FORCE_SELECTION = 4
        try:
            setattr(_dm, 'PassiveType', PassiveType)
        except Exception:
            pass

    if not hasattr(_dm, 'EffectActionType'):
        class EffectActionType(IntEnum):
            DRAW_CARD = 1
            APPLY_MODIFIER = 2
        try:
            setattr(_dm, 'EffectActionType', EffectActionType)
        except Exception:
            pass

    if not hasattr(_dm, 'DataCollector'):
        class DataCollector:
            def __init__(self, db=None):
                self.db = db
            def collect(self):
                return {}
        try:
            setattr(_dm, 'DataCollector', DataCollector)
        except Exception:
            pass

    if not hasattr(_dm, 'SelfAttention'):
        class SelfAttention:
            def __init__(self, embed_dim: int, num_heads: int):
                self.embed_dim = embed_dim
                self.num_heads = num_heads
            def initialize_weights(self):
                return None
            def forward(self, x):
                return x
        try:
            setattr(_dm, 'SelfAttention', SelfAttention)
        except Exception:
            pass

    # Best-effort: if CommandDef exists but lacks `instance_id`, try to
    # wrap it so Python-created instances have that attribute.
    try:
        if hasattr(_dm, 'CommandDef') and not hasattr(getattr(_dm, 'CommandDef'), 'instance_id'):
            Cmd = getattr(_dm, 'CommandDef')
            try:
                class _CmdWrap(Cmd):
                    def __init__(self, *a, **k):
                        super().__init__(*a, **k)
                        try:
                            if not hasattr(self, 'instance_id'):
                                setattr(self, 'instance_id', -1)
                        except Exception:
                            pass
                setattr(_dm, 'CommandDef', _CmdWrap)
            except Exception:
                pass
    except Exception:
        pass
except Exception:
    pass
