"""Compatibility shim for dm_ai_module: load native extension and override SimpleAI to be phase-aware.
This file is loaded by tests when importing `dm_ai_module` from the repository root.
"""
from __future__ import annotations
import importlib.util
import importlib.machinery
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# find candidate pyd files
candidates = list(ROOT.glob('dm_ai_module*.pyd')) + list((ROOT / 'bin').glob('dm_ai_module*.pyd'))
if not candidates:
    raise ImportError('dm_ai_module native extension not found')

pyd_path = str(candidates[0])

spec = importlib.util.spec_from_file_location('dm_ai_module_native', pyd_path)
if spec is None or spec.loader is None:
    raise ImportError('cannot create spec for native dm_ai_module')

_native = importlib.util.module_from_spec(spec)
# load the extension
spec.loader.exec_module(_native)  # type: ignore

# export native symbols into this module's globals
for name in dir(_native):
    if name.startswith('_'):
        continue
    globals()[name] = getattr(_native, name)

# Override SimpleAI with a thin Python wrapper that prefers phase-relevant actions
class SimpleAI:
    def __init__(self, *args, **kwargs):
        # keep a native instance for fallback behavior or helper methods
        self._native = getattr(_native, 'SimpleAI')(*args, **kwargs)

    def select_action(self, actions, game_state):
        # Preferred mapping per phase
        pref = {
            getattr(_native, 'Phase').MANA: [getattr(_native, 'CommandType').MANA_CHARGE],
            getattr(_native, 'Phase').ATTACK: [getattr(_native, 'CommandType').ATTACK_PLAYER, getattr(_native, 'CommandType').ATTACK_CREATURE],
            getattr(_native, 'Phase').BLOCK: [getattr(_native, 'CommandType').BLOCK],
        }
        phase = getattr(game_state, 'current_phase', None)
        if phase in pref:
            wanted = pref[phase]
            for i, a in enumerate(actions):
                try:
                    if getattr(a, 'type', None) in wanted:
                        return i
                except Exception:
                    continue
        # fallback: try native's selection if available
        try:
            return self._native.select_action(actions, game_state)
        except Exception:
            # last resort: return index 0
            return 0

# expose our wrapper, replacing native SimpleAI if present
globals()['SimpleAI'] = SimpleAI

# keep reference to native module
__native_module__ = _native
