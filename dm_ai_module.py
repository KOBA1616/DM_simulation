"""Minimal Python wrapper for dm_ai_module native extension.

This file loads the C++ extension. If the extension is not available,
it raises an ImportError, enforcing the use of the native engine.
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
import importlib.machinery
from enum import IntEnum
from typing import Any, List, Optional
import copy
import math
import uuid

# Try to load native extension if present in build output (prefer native C++ implementation)
# unless explicitly disabled via DM_DISABLE_NATIVE environment variable.
_disable_native = os.environ.get('DM_DISABLE_NATIVE', '').lower() in ('1', 'true', 'yes')
IS_NATIVE = False

if not _disable_native:
    try:
        # On Windows, proactively preload the onnxruntime DLL that ships with the
        # installed Python package. This avoids accidentally binding to a system-wide
        # onnxruntime.dll (e.g. under System32) which can be an older version and
        # trigger ORT API version mismatches when importing the native extension.
        try:
            if os.name == 'nt':
                import ctypes
                from pathlib import Path

                try:
                    import onnxruntime as _ort  # type: ignore

                    _capi_dir = Path(getattr(_ort, '__file__', '')).resolve().parent / 'capi'
                    _ort_dll = _capi_dir / 'onnxruntime.dll'
                    if _ort_dll.exists():
                        ctypes.WinDLL(str(_ort_dll))
                except Exception:
                    pass
        except Exception:
            pass
        
        _root = os.path.dirname(__file__)
        native_override = os.environ.get('DM_AI_MODULE_NATIVE')
        _candidates = []

        if native_override:
            try:
                if os.path.isdir(native_override):
                    for name in os.listdir(native_override):
                        if name.startswith('dm_ai_module') and (name.endswith('.pyd') or name.endswith('.so')):
                            _candidates.append(os.path.join(native_override, name))
                elif os.path.exists(native_override):
                    _candidates.append(native_override)
            except Exception:
                pass

        _candidates += [
            os.path.join(_root, 'bin', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'bin', 'dm_ai_module.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(_root, 'build', 'dm_ai_module.cpython-312-x86_64-linux-gnu.so'),
        ]
        _loaded_native = False
        for _p in _candidates:
            try:
                if _p and os.path.exists(_p):
                    loader = importlib.machinery.ExtensionFileLoader('dm_ai_module', _p)
                    spec = importlib.util.spec_from_loader('dm_ai_module', loader)
                    mod = importlib.util.module_from_spec(spec)
                    loader.exec_module(mod)
                    for _k, _v in mod.__dict__.items():
                        if _k.startswith('__'):
                            continue
                        globals()[_k] = _v
                    IS_NATIVE = True
                    _loaded_native = True
                    break
            except Exception:
                continue
    except Exception:
        IS_NATIVE = False

if not IS_NATIVE:
    # DM_DISABLE_NATIVE=1 のときはPythonフォールバックで続行する。
    # ネイティブが期待されている（DM_DISABLE_NATIVEが未設定）が見つからない場合のみ例外を投げる。
    # 再発防止: _disable_native チェックを必ず入れること。ネイティブが無効化されている場合は
    # ImportError を投げずにPythonフォールバック実装にフォールスルーさせる。
    if not _disable_native:
        raise ImportError("Native module dm_ai_module not found or failed to load. Please build the C++ extension (cmake).")


try:
    import torch
    import numpy as np
except ImportError:
    pass

# Expose a Python CommandEncoder fallback from native_prototypes if available
# or use the one from native module if it exports it.
# Note: Native module likely exports CommandEncoder if built with AI support.
if 'CommandEncoder' not in globals():
    try:
        from native_prototypes.index_to_command.command_encoder import CommandEncoder
    except Exception:
         # If not in native and not in prototypes, we might be in trouble for AI tasks,
         # but for engine tasks it's fine.
         pass


if 'CardStub' not in globals():
    class CardStub:
        _iid = 1000

        def __init__(self, card_id: int, instance_id: Optional[int] = None, cost: int = 1, card_type: str = 'CREATURE'):
            if instance_id is None:
                CardStub._iid += 1
                instance_id = CardStub._iid
            self.card_id = card_id
            self.instance_id = instance_id
            self.is_tapped = False
            self.sick = False
            self.cost = cost
            self.card_type = card_type  # 'CREATURE' or 'SPELL'


# Helper: shim imports for other classes (Zone, DevTools etc) can be left at module level
# if they are not core engine classes that native replaces.

if 'Zone' not in globals():
    from enum import IntEnum
    class Zone(IntEnum):
        DECK = 0
        HAND = 1
        MANA = 2
        BATTLE = 3
        GRAVEYARD = 4
        SHIELD = 5

if 'DevTools' not in globals():
    class DevTools:
        @staticmethod
        def move_cards(*args, **kwargs):
            return 0
        @staticmethod
        def trigger_loop_detection(state: Any):
            pass

if 'ParallelRunner' not in globals() and not IS_NATIVE:
    # This block is unreachable now if IS_NATIVE enforces True, but kept for logic structure
    pass

if 'JsonLoader' not in globals():
    # DM_DISABLE_NATIVE=1 時のPythonフォールバック実装
    # 再発防止: JsonLoader は GUI/テストで必ず使われるため、必ずフォールバックを提供すること。
    class JsonLoader:
        @staticmethod
        def load_cards(path: str) -> dict:
            """cards.jsonをロードしてid->カードデータの辞書を返す。"""
            import json as _json
            import os as _os
            # 相対パスの場合はワークスペースルートから解決を試みる
            if not _os.path.isabs(path):
                _root = _os.path.dirname(_os.path.abspath(__file__))
                path = _os.path.join(_root, path)
            with open(path, 'r', encoding='utf-8') as _f:
                _cards = _json.load(_f)
            if isinstance(_cards, list):
                return {c['id']: c for c in _cards if 'id' in c}
            return _cards

if 'ExecuteActionCompat' not in globals():
    def ExecuteActionCompat(target: Any, action: Any, player_id: int = 0, ctx: Any = None) -> bool:
        try:
            if hasattr(target, 'execute_command'):
                target.execute_command(action)
                return True
        except Exception:
            pass
        return False

# Ensure core symbols exist on the module object even if a native extension
# was loaded earlier and didn't export them (Shim injection into Native module object)
try:
    import sys as _sys
    _mod = _sys.modules.get('dm_ai_module')
    if _mod is not None and IS_NATIVE:
        # If we are here, IS_NATIVE is True, so we imported native.
        # But we might want to inject pure-python helpers like `Zone` if native didn't have them.
        for _name, _obj in list(globals().items()):
             if _name.startswith('_'): continue
             if not hasattr(_mod, _name):
                 setattr(_mod, _name, _obj)
except Exception:
    pass
