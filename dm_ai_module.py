"""Project-local dm_ai_module loader.

This repository builds a native extension module named `dm_ai_module`.
To keep imports consistent across GUI / scripts / tests, this file acts as the
canonical import target for `import dm_ai_module`.

Policy:
1.  **Prioritize Source/Build Artifacts**: We check `build/`, `bin/`, and `release/` directories relative to the repo root first.
    This ensures that when developers or CI modify C++ code and rebuild, the new version is used immediately without needing `pip install`.
2.  **Fallback to Installed/System**: If no local artifact is found, we fall back to standard import logic (which might find a system-installed version).
3.  **Fallback to Stub**: If no native module is found at all, we use a lightweight pure-Python stub.
    Tests that strictly require the native module must check `dm_ai_module.IS_NATIVE` or use the `require_native` fixture.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os
import sys
from enum import Enum
from types import ModuleType
from typing import Any, Optional


def _ensure_windows_dll_search_path() -> None:
    """Ensure dependent DLLs resolve from the active Python environment.

    On Windows, native extensions (like dm_ai_module.pyd) may depend on DLLs such
    as onnxruntime.dll. If an older DLL is found earlier on PATH, it can cause
    C-API version mismatches at import time.
    """
    if os.name != "nt":
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return

    try:
        import onnxruntime as ort  # type: ignore

        ort_pkg_dir = os.path.dirname(os.path.abspath(ort.__file__))
        capi_dir = os.path.join(ort_pkg_dir, "capi")
        for p in (ort_pkg_dir, capi_dir):
            if os.path.isdir(p):
                try:
                    add_dir(p)
                except Exception:
                    pass
    except Exception:
        # If onnxruntime isn't installed (or import fails), just skip.
        pass


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_native_paths(root: str) -> list[str]:
    # Prioritize build directories to catch fresh builds.
    patterns = [
        # Explicit bin directory often used for output
        os.path.join(root, "bin", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "dm_ai_module*.so"),
        # CMake build directories
        os.path.join(root, "build", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build", "**", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.dylib"),
        # Fallback to general patterns if specific ones miss
        os.path.join(root, "build*", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build*", "**", "dm_ai_module*.so"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    # Prefer Release artifacts when multiple exist (e.g., Debug and Release builds).
    def _score(p: str) -> tuple[int, int]:
        p_norm = p.replace("/", "\\").lower()
        return (
            0 if "\\release\\" in p_norm or "/release/" in p_norm else 1,
            0 if "\\bin\\" in p_norm or "/bin/" in p_norm else 1,
        )

    uniq = sorted({os.path.normpath(p) for p in paths}, key=_score)
    return uniq


def _load_native_in_place(module_name: str, path: str) -> ModuleType:
    try:
        loader = importlib.machinery.ExtensionFileLoader(module_name, path)
        spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
        if spec is None:
            raise ImportError(f"Could not create spec for native module at: {path}")
        module = importlib.util.module_from_spec(spec)

        # Pre-register to support recursive imports during module init.
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        # If loading fails (e.g. missing DLLs), clean up sys.modules and re-raise
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e


def _try_load_native() -> Optional[ModuleType]:
    root = _repo_root()

    _ensure_windows_dll_search_path()

    # Allow explicit override (useful for debugging specific builds).
    override = os.environ.get("DM_AI_MODULE_NATIVE")
    candidates = [override] if override else _candidate_native_paths(root)

    for p in candidates:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                mod = _load_native_in_place(__name__, p)
                return mod
            except Exception:
                # Log or ignore? For now ignore and try next candidate.
                continue
    return None


_native = _try_load_native()

if _native is not None:
    # Expose native symbols from this module.
    globals().update(_native.__dict__)
    IS_NATIVE = True

    # Minimal shims for compatibility when native builds lack some helpers.
    if "DeclarePlayCommand" not in globals():
        class DeclarePlayCommand:  # type: ignore
            def __init__(self, player_id: int, card_id: int, source_instance_id: int):
                self.player_id = player_id
                self.card_id = card_id
                self.source_instance_id = source_instance_id

            def execute(self, state: Any) -> None:
                setattr(state, "_last_declared_play", {
                    "player_id": self.player_id,
                    "card_id": self.card_id,
                    "source_instance_id": self.source_instance_id,
                })
        globals()["DeclarePlayCommand"] = DeclarePlayCommand

    if "PayCostCommand" not in globals():
        class PayCostCommand:  # type: ignore
            def __init__(self, player_id: int, amount: int):
                self.player_id = player_id
                self.amount = amount

            def execute(self, state: Any) -> bool:
                return True
        globals()["PayCostCommand"] = PayCostCommand

    if "ResolvePlayCommand" not in globals():
        class ResolvePlayCommand:  # type: ignore
            def __init__(self, player_id: int, card_id: int, card_def: Any = None):
                self.player_id = player_id
                self.card_id = card_id
                self.card_def = card_def

            def execute(self, state: Any) -> None:
                return
        globals()["ResolvePlayCommand"] = ResolvePlayCommand

else:
    # -----------------
    # Pure-Python fallback (Stub for tests/lint)
    # -----------------
    IS_NATIVE = False

    class Civilization(Enum):
        FIRE = 1
        WATER = 2
        NATURE = 3
        LIGHT = 4
        DARKNESS = 5

    class CardType(Enum):
        CREATURE = 1
        SPELL = 2

    class CardKeywords(int):
        pass

    class PassiveType(Enum):
        NONE = 0

    class PassiveEffect:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class FilterDef(dict):
        pass

    class EffectDef:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class ActionDef:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class Action:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = None
            self.target_player = 0
            self.source_instance_id = 0
            self.card_id = 0
            self.slot_index = 0
            self.value1 = 0

    class ActionType(Enum):
        PLAY_CARD = 1
        ATTACK_PLAYER = 2
        ATTACK_CREATURE = 3
        BLOCK_CREATURE = 4
        PASS = 5
        USE_SHIELD_TRIGGER = 6
        MANA_CHARGE = 7
        RESOLVE_EFFECT = 8
        SELECT_TARGET = 9

    class ConditionDef:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class EffectActionType(Enum):
        DRAW_CARD = 1
        SEND_SHIELD_TO_GRAVE = 2
        SEARCH_DECK_BOTTOM = 3
        CAST_SPELL = 4
        PUT_CREATURE = 5
        TAP_CREATURE = 6
        DESTROY_CREATURE = 7
        HAND_DISCARD = 8
        MANA_CHARGE = 9
        MOVE_CARD = 10
        SELECT_TARGET = 11
        RESOLVE_EFFECT = 12
        BREAK_SHIELD = 13

    class JsonLoader:
        @staticmethod
        def load_cards(path: str) -> dict[int, Any]:
            return {}

    class LethalSolver:
        @staticmethod
        def is_lethal(state: Any, card_db: Any) -> bool:
            return False

    class PlayerStub:
        def __init__(self) -> None:
            self.hand: list[Any] = []
            self.deck: list[Any] = []
            self.battle_zone: list[Any] = []
            self.graveyard: list[Any] = []
            self.mana_zone: list[Any] = []
            self.shield_zone: list[Any] = []

    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            self.turn_number = 0
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0

        def setup_test_duel(self) -> None:
            return

    class CardDefinition:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    _batch_callback: Optional[Any] = None

    def set_batch_callback(cb: Any) -> None:
        global _batch_callback
        _batch_callback = cb

    def has_batch_callback() -> bool:
        return _batch_callback is not None

    def clear_batch_callback() -> None:
        global _batch_callback
        _batch_callback = None

    class ActionEncoder:
        TOTAL_ACTION_SIZE = 10

    class NeuralEvaluator:
        def __init__(self, card_db: Any):
            self.card_db = card_db

        def evaluate(self, batch: list[Any]):
            if _batch_callback is not None:
                return _batch_callback(batch)
            policies = [[0.0] * ActionEncoder.TOTAL_ACTION_SIZE for _ in batch]
            values = [0.0 for _ in batch]
            return policies, values

    class CommandDef:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class GameCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            pass
        def execute(self, state: Any) -> None:
            pass

    class MutateCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            pass
        def execute(self, state: Any) -> None:
            pass

    class FlowCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            pass
        def execute(self, state: Any) -> None:
            pass

    class MutationType(Enum):
        ADD_MODIFIER = 1
        ADD_PASSIVE = 2

    class CardData:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, *args: Any, **kwargs: Any) -> None:
            return

    class EffectResolver:
        @staticmethod
        def resolve_action(state: Any, action: Any, card_db: Any) -> None:
            return

    class PhaseManager:
        @staticmethod
        def next_phase(state: Any, card_db: Any) -> None:
            return

    def register_card_data(data: Any) -> None:
        return

    class CardRegistry:
        @staticmethod
        def get_all_cards() -> dict[int, Any]:
            return {}
        @staticmethod
        def get_all_definitions() -> dict[int, Any]:
            return {}
        @staticmethod
        def get_card_data(card_id: int) -> Any:
            return None

    class GenericCardSystem:
        @staticmethod
        def resolve_action_with_db(state: Any, action: Any, source_id: int, card_db: Any, ctx: Any = None) -> Any:
            return None
        @staticmethod
        def resolve_effect_with_db(state: Any, eff: Any, source_id: int, card_db: Any) -> None:
            return

    class CommandType(Enum):
        TRANSITION = 1
        MUTATE = 2
        FLOW = 3
        QUERY = 4
        DRAW_CARD = 5
        DISCARD = 6
        DESTROY = 7
        MANA_CHARGE = 8
        TAP = 9
        UNTAP = 10
        POWER_MOD = 11
        ADD_KEYWORD = 12
        RETURN_TO_HAND = 13
        BREAK_SHIELD = 14
        SEARCH_DECK = 15
        SHIELD_TRIGGER = 16
        NONE = 17

    class TargetScope(Enum):
        PLAYER_SELF = 1
        SELF = 1

    class Zone(Enum):
        DECK = 1
        HAND = 2
        GRAVEYARD = 3
        MANA = 4
        BATTLE_ZONE = 5
        SHIELD_ZONE = 6
        BATTLE = 5
        SHIELD = 6
