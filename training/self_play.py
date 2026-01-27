# Backwards-compat shim for legacy imports: `import training.self_play`
# Delegates to `dm_toolkit.training.self_play` implementation.
from dm_toolkit.training.self_play import *
__all__ = [name for name in globals() if not name.startswith('_')]
