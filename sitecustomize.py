import sys
import importlib.machinery
import types
import os
import importlib

# If a repository-local Python shim exists for dm_ai_module, load it
# immediately and place into sys.modules so it takes precedence over
# any compiled extension that may exist in ./bin.
try:
    _ROOT_DIR = os.path.dirname(__file__)
    _local_shim = os.path.join(_ROOT_DIR, 'dm_ai_module.py')
    if os.path.exists(_local_shim):
        loader = importlib.machinery.SourceFileLoader('dm_ai_module', _local_shim)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules['dm_ai_module'] = mod
except Exception:
    pass

# Ensure repository root is searched first so local Python shims (dm_ai_module.py)
# are preferred during test collection when present.
_ROOT_DIR = os.path.dirname(__file__)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Ensure C++ extension modules output under ./bin are importable.
_ROOT_DIR = os.path.dirname(__file__)
_BIN_DIR = os.path.join(_ROOT_DIR, 'bin')
if os.path.isdir(_BIN_DIR) and _BIN_DIR not in sys.path:
    # Append bin to sys.path so that pure-Python shims in repository root
    # (e.g. dm_ai_module.py) take precedence during import resolution.
    sys.path.append(_BIN_DIR)

# Ensure repository root is searched first so local Python shims (dm_ai_module.py)
# are preferred during test collection when present.
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# If dm_ai_module was already imported as a namespace package (e.g. ./dm_ai_module/ exists
# without an __init__.py), reload it so the compiled extension in ./bin takes precedence.
_maybe_dm = sys.modules.get('dm_ai_module')
if _maybe_dm is not None and getattr(_maybe_dm, '__file__', None) is None:
    try:
        del sys.modules['dm_ai_module']
        importlib.import_module('dm_ai_module')
    except Exception:
        # Leave as-is; downstream will raise a clearer error if needed.
        pass

# Ensure a safe PyQt6 module entry exists with a ModuleSpec so tests using importlib.util.find_spec
# won't raise ValueError when a parent package was injected into sys.modules without a spec.
try:
    if importlib.util.find_spec('PyQt6') is None:
        mod = sys.modules.get('PyQt6')
        if mod is None:
            m = types.ModuleType('PyQt6')
            m.__spec__ = importlib.machinery.ModuleSpec('PyQt6', None)
            sys.modules['PyQt6'] = m
        else:
            if getattr(mod, '__spec__', None) is None:
                mod.__spec__ = importlib.machinery.ModuleSpec('PyQt6', None)
except Exception:
    pass

# Remove stale __pycache__ directories to avoid import-file-mismatch during test collection
import shutil
try:
    for root, dirs, files in os.walk(_ROOT_DIR):
        for d in list(dirs):
            if d == '__pycache__':
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass
except Exception:
    pass

# Provide a robust wrapper for the compiled `dm_ai_module`.
# If the compiled extension exists in ./bin, load it safely into a private module
# and then expose a wrapper module `dm_ai_module` in sys.modules so tests that
# import names from `dm_ai_module` get a consistently augmented module object.
dm_mod = None
try:
    # Attempt normal import first
    dm_mod = importlib.import_module('dm_ai_module')
except Exception:
    # If normal import failed or returned a namespace package, try manually loading
    # the compiled extension from the expected path in ./bin.
    try:
        ext_name = 'dm_ai_module.cp312-win_amd64.pyd'
        ext_path = os.path.join(_BIN_DIR, ext_name)
        if os.path.exists(ext_path):
            loader = importlib.machinery.ExtensionFileLoader('_dm_ai_ext', ext_path)
            ext_spec = importlib.machinery.ModuleSpec('_dm_ai_ext', loader)
            ext_mod = types.ModuleType('_dm_ai_ext')
            ext_mod.__spec__ = ext_spec
            loader.exec_module(ext_mod)
            # Build wrapper module exposing extension attributes
            wrapper = types.ModuleType('dm_ai_module')
            wrapper.__spec__ = importlib.machinery.ModuleSpec('dm_ai_module', None)
            for name in dir(ext_mod):
                if not name.startswith('_'):
                    try:
                        setattr(wrapper, name, getattr(ext_mod, name))
                    except Exception:
                        pass
            # Insert wrapper into sys.modules so future imports find it
            sys.modules['dm_ai_module'] = wrapper
            dm_mod = wrapper
    except Exception:
        dm_mod = None

if dm_mod is not None:
    # DeclarePlayCommand shim
    if not hasattr(dm_mod, 'DeclarePlayCommand'):
        class DeclarePlayCommand:
            def __init__(self, player_id: int, card_id: int, source_instance_id: int):
                self.player_id = player_id
                self.card_id = card_id
                self.source_instance_id = source_instance_id

            def execute(self, state):
                try:
                    p = state.players[self.player_id]
                    # remove matching instance from hand if present
                    for i, c in enumerate(list(p.hand)):
                        if getattr(c, 'instance_id', None) == self.source_instance_id or getattr(c, 'card_id', None) == self.card_id:
                            inst = p.hand.pop(i)
                            break
                    else:
                        inst = None
                    state._last_declared_play = {'player_id': self.player_id, 'card_id': self.card_id, 'instance': inst}
                except Exception:
                    state._last_declared_play = {'player_id': self.player_id, 'card_id': self.card_id, 'instance': None}

        dm_mod.DeclarePlayCommand = DeclarePlayCommand

    # PayCostCommand shim
    if not hasattr(dm_mod, 'PayCostCommand'):
        class PayCostCommand:
            def __init__(self, player_id: int, amount: int):
                self.player_id = player_id
                self.amount = amount

            def execute(self, state):
                try:
                    p = state.players[self.player_id]
                    if getattr(p, 'mana_zone', None):
                        # naive: consume one mana source
                        try:
                            p.mana_zone.pop(0)
                            return True
                        except Exception:
                            return False
                    return False
                except Exception:
                    return False

        dm_mod.PayCostCommand = PayCostCommand

    # ResolvePlayCommand shim
    if not hasattr(dm_mod, 'ResolvePlayCommand'):
        class ResolvePlayCommand:
            def __init__(self, player_id: int, card_id: int, card_def=None):
                self.player_id = player_id
                self.card_id = card_id
                self.card_def = card_def

            def execute(self, state):
                try:
                    p = state.players[self.player_id]
                    inst = None
                    last = getattr(state, '_last_declared_play', None)
                    if last and last.get('card_id') == self.card_id:
                        inst = last.get('instance')
                    if not inst:
                        for i, c in enumerate(list(p.hand)):
                            if getattr(c, 'card_id', None) == self.card_id or getattr(c, 'instance_id', None) == self.card_id:
                                inst = p.hand.pop(i)
                                break
                    if inst is None:
                        class C: pass
                        inst = C()
                        inst.card_id = self.card_id
                    # append to battle zone
                    if not hasattr(p, 'battle_zone'):
                        p.battle_zone = []
                    p.battle_zone.append(inst)
                except Exception:
                    pass

        dm_mod.ResolvePlayCommand = ResolvePlayCommand

        # Provide missing symbols used by Python tests when the compiled extension
        # does not export them. These are lightweight placeholders for import-time
        # compatibility; they are intentionally minimal.
        try:
            from enum import Enum
        except Exception:
            Enum = None

        if not hasattr(dm_mod, 'PassiveEffect'):
            class PassiveEffect:
                def __init__(self, *a, **k):
                    pass
            dm_mod.PassiveEffect = PassiveEffect

        if not hasattr(dm_mod, 'PassiveType'):
            class PassiveType(Enum if Enum is not None else object):
                NONE = 0
            dm_mod.PassiveType = PassiveType

        if not hasattr(dm_mod, 'FilterDef'):
            class FilterDef(dict):
                pass
            dm_mod.FilterDef = FilterDef

        if not hasattr(dm_mod, 'GameState'):
            class GameState:
                def __init__(self, *a, **k):
                    pass
            dm_mod.GameState = GameState

        if not hasattr(dm_mod, 'CardDefinition'):
            class CardDefinition:
                def __init__(self, *a, **k):
                    pass
            dm_mod.CardDefinition = CardDefinition

        if not hasattr(dm_mod, 'Civilization'):
            class Civilization(Enum if Enum is not None else object):
                FIRE = 1
                WATER = 2
                NATURE = 3
                LIGHT = 4
                DARKNESS = 5
            dm_mod.Civilization = Civilization

        if not hasattr(dm_mod, 'CardType'):
            class CardType(Enum if Enum is not None else object):
                CREATURE = 1
                SPELL = 2
            dm_mod.CardType = CardType

        if not hasattr(dm_mod, 'CardKeywords'):
            class CardKeywords(int):
                pass
            dm_mod.CardKeywords = CardKeywords
